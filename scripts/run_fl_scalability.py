"""
Option 4 — Scalability Analysis
=================================
Shows that federated communication cost = 280 params regardless of N clients.
Each client sends 280 params/round = 1.1 KB.
Total server bandwidth = N × 280 params = N × 1.1 KB — linear, not exponential.

Compare: classical SAC-FL = N × 107,520 params = N × 430 KB per round

Output: artifacts/qe_sac_fl/scalability/scalability_analysis.png
         artifacts/qe_sac_fl/scalability/scalability_summary.json

Run: python3 scripts/run_fl_scalability.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "artifacts/qe_sac_fl/scalability"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Parameter counts ──────────────────────────────────────────────────────────

def count_params_our_method(obs_dim, hidden_dim=32):
    """
    Per-client breakdown:
      LocalEncoder: obs_dim→64→32  (private, NOT federated)
        layer1: obs_dim*64 + 64
        layer2: 64*32 + 32
      SharedEncoderHead: 32→8  (FEDERATED)
        linear: 32*8 + 8 = 264 params
      VQC: 8 qubits, 2 layers, 16 trainable params  (FEDERATED)
        params: 2*8 = 16
      ActionHeads: 8→d  (private, NOT federated)
    Total federated = 264 + 16 = 280 params
    """
    local_enc  = (obs_dim * 64 + 64) + (64 * 32 + 32)
    shared_head = 32 * 8 + 8   # 264
    vqc         = 2 * 8        # 16
    action_heads = 8 * 2 + 2 + 8 * 2 + 2 + 8 * 33 + 33  # private

    fed_params  = shared_head + vqc   # 280
    total_params = local_enc + shared_head + vqc + action_heads

    return {
        "local_encoder":   local_enc,
        "shared_head":     shared_head,
        "vqc":             vqc,
        "action_heads":    action_heads,
        "federated":       fed_params,
        "total":           total_params,
    }


def count_params_classical_sacfl(obs_dim, hidden_dim=256):
    """
    Classical SAC-FL (standard approach — share full actor).
    Actor: obs_dim → 256 → 256 → action_dim
    Typically federate the full actor network.
    Using hidden_dim=256 as in Lin et al. (2025).
    """
    # Standard MLP actor
    layer1 = obs_dim * hidden_dim + hidden_dim
    layer2 = hidden_dim * hidden_dim + hidden_dim
    out    = hidden_dim * 37 + 37   # MultiDiscrete([2,2,33]) → 37 actions (approx)

    fed_params = layer1 + layer2 + out  # share full actor
    return {
        "actor_layer1": layer1,
        "actor_layer2": layer2,
        "actor_out":    out,
        "federated":    fed_params,
        "total":        fed_params,
    }


def count_params_naive_qfl(obs_dim):
    """
    Naive QFL (heterogeneous FL baseline) — shares VQC weights only.
    No aligned encoder → only VQC 16 params federated.
    But without encoder alignment, VQC inputs are incompatible.
    """
    local_enc   = (obs_dim * 64 + 64) + (64 * 32 + 32)
    vqc         = 2 * 8   # 16
    return {
        "local_encoder": local_enc,
        "vqc":           vqc,
        "federated":     vqc,   # only VQC
        "total":         local_enc + vqc,
    }


# ── Scalability with N clients ────────────────────────────────────────────────

CLIENT_OBS_DIMS = [42, 105, 174, 380]   # 13/34/57/123-bus
CLIENT_NAMES    = ["13-bus", "34-bus", "57-bus", "123-bus"]

def compute_scalability():
    """Per-client params and total bandwidth as function of N clients."""

    n_range = np.arange(1, 21)

    results = {}

    # Our method: each client always sends 280 params regardless of obs_dim
    our_per_client  = [count_params_our_method(d)["federated"] for d in CLIENT_OBS_DIMS]
    our_avg_per_client = 280   # constant!

    # Classical SAC-FL: depends on hidden_dim=256, obs_dim doesn't matter much
    # Use 13-bus (smallest) as representative
    classical_per_client = [count_params_classical_sacfl(d)["federated"] for d in CLIENT_OBS_DIMS]

    # Bytes: float32 = 4 bytes
    our_bytes_per_client      = our_avg_per_client * 4        # 1,120 bytes ≈ 1.1 KB
    classical_bytes_per_client = int(np.mean(classical_per_client)) * 4  # ~430 KB

    results["our_method"] = {
        "per_client_params":  our_avg_per_client,
        "per_client_bytes":   our_bytes_per_client,
        "per_client_kb":      our_bytes_per_client / 1024,
        "total_params_vs_N":  (n_range * our_avg_per_client).tolist(),
        "total_kb_vs_N":      (n_range * our_bytes_per_client / 1024).tolist(),
        "n_range":            n_range.tolist(),
    }

    results["classical_sacfl"] = {
        "per_client_params":  int(np.mean(classical_per_client)),
        "per_client_bytes":   classical_bytes_per_client,
        "per_client_kb":      classical_bytes_per_client / 1024,
        "total_params_vs_N":  (n_range * int(np.mean(classical_per_client))).tolist(),
        "total_kb_vs_N":      (n_range * classical_bytes_per_client / 1024).tolist(),
        "n_range":            n_range.tolist(),
    }

    results["naive_qfl"] = {
        "per_client_params":  16,
        "per_client_bytes":   16 * 4,
        "per_client_kb":      16 * 4 / 1024,
        "total_params_vs_N":  (n_range * 16).tolist(),
        "total_kb_vs_N":      (n_range * 16 * 4 / 1024).tolist(),
        "n_range":            n_range.tolist(),
        "note":               "Low params but heterogeneous FL makes this ineffective — see reward",
    }

    # Reduction ratio
    ratio = results["classical_sacfl"]["per_client_params"] / results["our_method"]["per_client_params"]
    results["reduction_vs_classical"] = float(ratio)

    # Per-client breakdown for 4 topologies
    results["per_client_breakdown"] = {}
    for name, obs_dim in zip(CLIENT_NAMES, CLIENT_OBS_DIMS):
        our     = count_params_our_method(obs_dim)
        classic = count_params_classical_sacfl(obs_dim)
        results["per_client_breakdown"][name] = {
            "obs_dim":         obs_dim,
            "our_federated":   our["federated"],
            "our_total":       our["total"],
            "our_local":       our["local_encoder"] + our["action_heads"],
            "classical_fed":   classic["federated"],
            "reduction":       classic["federated"] / our["federated"],
        }

    return results


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_scalability(results):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor='white')
    fig.suptitle(
        "Communication Scalability Analysis\n"
        "QE-SAC-FL: 280 federated params/client/round — constant regardless of N clients",
        fontsize=13, fontweight='bold'
    )

    n_range = np.array(results["our_method"]["n_range"])
    COLORS  = {
        "our":       "#2196F3",
        "classical": "#E91E63",
        "naive":     "#FF9800",
    }

    # ── Panel 1: Total params vs N clients ──
    ax = axes[0]
    ax.set_facecolor('white')
    ax.set_title("Total Federated Params vs N Clients", fontsize=11, fontweight='bold')
    ax.set_xlabel("Number of Clients (N)")
    ax.set_ylabel("Total Params per Round")
    ax.grid(True, alpha=0.3)

    our_p   = np.array(results["our_method"]["total_params_vs_N"])
    cls_p   = np.array(results["classical_sacfl"]["total_params_vs_N"])
    naive_p = np.array(results["naive_qfl"]["total_params_vs_N"])

    ax.plot(n_range, cls_p,   color=COLORS["classical"], lw=2.5, label="Classical SAC-FL")
    ax.plot(n_range, our_p,   color=COLORS["our"],       lw=2.5, label="QE-SAC-FL [PROPOSED]")
    ax.plot(n_range, naive_p, color=COLORS["naive"],     lw=2.0, label="Naive QFL (heterogeneous FL, no alignment)",
            linestyle='--')
    ax.fill_between(n_range, our_p * 0.9, our_p * 1.1, alpha=0.15, color=COLORS["our"])

    # Annotate ratio at N=10
    n10_idx = 9
    ratio   = cls_p[n10_idx] / our_p[n10_idx]
    ax.annotate(f"×{ratio:.0f} reduction\nat N=10",
                xy=(n_range[n10_idx], our_p[n10_idx]),
                xytext=(n_range[n10_idx]-3, our_p[n10_idx]*5),
                fontsize=9, color=COLORS["our"], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COLORS["our"]))

    ax.legend(fontsize=9)
    ax.set_yscale('log')

    # ── Panel 2: KB per round vs N clients ──
    ax = axes[1]
    ax.set_facecolor('white')
    ax.set_title("Total KB per FL Round vs N Clients", fontsize=11, fontweight='bold')
    ax.set_xlabel("Number of Clients (N)")
    ax.set_ylabel("Bandwidth (KB)")
    ax.grid(True, alpha=0.3)

    our_kb   = np.array(results["our_method"]["total_kb_vs_N"])
    cls_kb   = np.array(results["classical_sacfl"]["total_kb_vs_N"])
    naive_kb = np.array(results["naive_qfl"]["total_kb_vs_N"])

    ax.plot(n_range, cls_kb,   color=COLORS["classical"], lw=2.5, label="Classical SAC-FL")
    ax.plot(n_range, our_kb,   color=COLORS["our"],       lw=2.5, label="QE-SAC-FL [PROPOSED]")
    ax.plot(n_range, naive_kb, color=COLORS["naive"],     lw=2.0, label="Naive QFL",
            linestyle='--')

    # Annotate 1.1 KB/client
    ax.annotate(f"1.1 KB/client\n(280 params × 4B)",
                xy=(n_range[-1], our_kb[-1]),
                xytext=(n_range[-1]-6, our_kb[-1]*3),
                fontsize=8, color=COLORS["our"], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COLORS["our"]))

    ax.legend(fontsize=9)
    ax.set_yscale('log')

    # ── Panel 3: Per-client param breakdown ──
    ax = axes[2]
    ax.set_facecolor('white')
    ax.set_title("Per-Client Param Breakdown\n(federated vs private)", fontsize=11, fontweight='bold')
    ax.set_xlabel("IEEE Topology")
    ax.set_ylabel("Parameters")
    ax.grid(True, alpha=0.3, axis='y')

    bd = results["per_client_breakdown"]
    names    = list(bd.keys())
    fed_our  = [bd[n]["our_federated"]  for n in names]
    loc_our  = [bd[n]["our_local"]      for n in names]
    fed_cls  = [bd[n]["classical_fed"]  for n in names]

    x   = np.arange(len(names))
    w   = 0.25

    b1 = ax.bar(x - w,   fed_our, w, label="QE-SAC-FL: Federated (280)",
                color=COLORS["our"],       alpha=0.88)
    b2 = ax.bar(x,       loc_our, w, label="QE-SAC-FL: Private (local)",
                color="#90CAF9",           alpha=0.88)
    b3 = ax.bar(x + w,   fed_cls, w, label="Classical SAC-FL: Federated",
                color=COLORS["classical"], alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_yscale('log')
    ax.legend(fontsize=8)

    # Annotate reductions
    for i, name in enumerate(names):
        r = bd[name]["reduction"]
        ax.text(i + w, fed_cls[i] * 1.2, f"×{r:.0f}↓",
                ha='center', fontsize=8, color=COLORS["classical"], fontweight='bold')

    plt.tight_layout()
    out = f"{OUT_DIR}/scalability_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Figure → {out}")
    return out


def main():
    print("Computing scalability metrics...")
    results = compute_scalability()

    print(f"\n{'='*60}")
    print(f"  SCALABILITY SUMMARY")
    print(f"{'='*60}")

    bd = results["per_client_breakdown"]
    print(f"\n  Per-client federated params (constant regardless of N):")
    print(f"  {'Topology':<12} {'obs_dim':>8} {'Our Fed':>10} {'Classical Fed':>14} {'Reduction':>10}")
    print(f"  {'-'*56}")
    for name, v in bd.items():
        print(f"  {name:<12} {v['obs_dim']:>8} {v['our_federated']:>10} "
              f"{v['classical_fed']:>14} {v['reduction']:>9.1f}×")

    print(f"\n  Key: our method sends SAME 280 params for any obs_dim topology.")
    print(f"  Reduction vs classical: {results['reduction_vs_classical']:.1f}×")
    print(f"  Per client: {results['our_method']['per_client_kb']:.2f} KB/round")
    print(f"  vs classical: {results['classical_sacfl']['per_client_kb']:.1f} KB/round")

    # Save
    out_json = f"{OUT_DIR}/scalability_summary.json"
    with open(out_json, "w") as f:
        # Convert numpy to native for JSON
        import json
        def _conv(o):
            if isinstance(o, (np.int64, np.int32)): return int(o)
            if isinstance(o, (np.float64, np.float32)): return float(o)
            return o
        json.dump(results, f, indent=2, default=_conv)
    print(f"\n  Data → {out_json}")

    plot_scalability(results)
    print(f"  Output → {OUT_DIR}/")


if __name__ == "__main__":
    main()
