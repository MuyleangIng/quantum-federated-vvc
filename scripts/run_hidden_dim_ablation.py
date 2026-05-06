"""
Hidden Dim Ablation — justifies hidden_dim=32 in SharedEncoderHead.

Runs aligned FL with hidden_dim ∈ {16, 32, 64, 128} and compares:
  - Final reward per client
  - SharedHead parameter count
  - Communication cost per round

Usage:
    python scripts/run_hidden_dim_ablation.py > logs/ablation.log 2>&1 &
Output:
    artifacts/qe_sac_fl/ablation/hidden_dim_results.json
    artifacts/qe_sac_fl/ablation/hidden_dim_ablation.png
"""

import sys
import os
import json

sys.path.insert(0, "/root/power-system")
os.makedirs("artifacts/qe_sac_fl/ablation", exist_ok=True)


def head_param_count(hidden_dim: int) -> int:
    """SharedEncoderHead: Linear(hidden_dim, 8) + bias = hidden_dim*8 + 8."""
    return hidden_dim * 8 + 8


def bytes_per_round(hidden_dim: int, n_clients: int = 3) -> int:
    """Upload + download of SharedHead + VQC per FL round."""
    params = head_param_count(hidden_dim) + 16  # head + VQC
    return n_clients * 2 * params * 4            # float32


def run_one(hidden_dim: int, n_rounds: int = 30) -> dict:
    """Run aligned FL with a specific hidden_dim. Returns summary dict."""
    import torch
    from src.qe_sac_fl.fed_config import FedConfig, ClientConfig
    from src.qe_sac_fl.federated_trainer import FederatedTrainer

    n_gpus  = torch.cuda.device_count()
    devices = [f"cuda:{i}" if i < n_gpus else "cpu" for i in range(3)]

    cfg = FedConfig()
    cfg.n_rounds         = n_rounds
    cfg.local_steps      = 500
    cfg.warmup_steps     = 500
    cfg.batch_size       = 256
    cfg.lr               = 3e-4
    cfg.buffer_size      = 100_000
    cfg.log_interval     = 10
    cfg.parallel_clients = (n_gpus >= 3)
    cfg.hidden_dim       = hidden_dim
    cfg.clients = [
        ClientConfig(name="Utility_A_13bus",  env_id="13bus_fl",  obs_dim=43,  n_actions=132, seed=0, device=devices[0], reward_scale=50.0),
        ClientConfig(name="Utility_B_34bus",  env_id="34bus_fl",  obs_dim=113, n_actions=132, seed=1, device=devices[1], reward_scale=10.0),
        ClientConfig(name="Utility_C_123bus", env_id="123bus_fl", obs_dim=349, n_actions=132, seed=2, device=devices[2], reward_scale=750.0),
    ]

    trainer = FederatedTrainer(cfg)
    results = trainer.run_aligned()

    hp    = head_param_count(hidden_dim)
    bpr   = bytes_per_round(hidden_dim)
    rewards = results.final_rewards()

    summary = {
        "hidden_dim":       hidden_dim,
        "head_params":      hp,
        "total_fed_params": hp + 16,
        "bytes_per_round":  bpr,
        "final_rewards":    rewards,
        "mean_reward":      float(sum(rewards.values()) / len(rewards)) if rewards else 0.0,
    }
    print(f"  hidden_dim={hidden_dim}  head_params={hp}  bytes/round={bpr:,}  "
          f"rewards={[f'{v:.1f}' for v in rewards.values()]}")
    return summary


def plot_results(all_results: list):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plot")
        return

    dims    = [r["hidden_dim"] for r in all_results]
    clients = list(all_results[0]["final_rewards"].keys())
    colors  = ["#e74c3c", "#3498db", "#2ecc71"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Plot 1: reward per client
    ax = axes[0]
    for client, color in zip(clients, colors):
        rewards = [r["final_rewards"].get(client, 0) for r in all_results]
        ax.plot(dims, rewards, "o-", color=color,
                label=client.split("_")[1], linewidth=2, markersize=7)
    ax.axvline(x=32, color="gray", linestyle="--", alpha=0.5, label="current (32)")
    ax.set_xlabel("hidden_dim")
    ax.set_ylabel("Final Reward")
    ax.set_title("Reward vs hidden_dim")
    ax.legend(fontsize=8)
    ax.set_xticks(dims)

    # Plot 2: SharedHead param count
    ax = axes[1]
    hp = [r["head_params"] for r in all_results]
    bars = ax.bar([str(d) for d in dims], hp, color=["#e74c3c" if d==32 else "#9b59b6" for d in dims], alpha=0.8)
    ax.set_xlabel("hidden_dim")
    ax.set_ylabel("SharedHead Params")
    ax.set_title("Federated Params vs hidden_dim")
    for bar, p in zip(bars, hp):
        ax.text(bar.get_x() + bar.get_width()/2, p + 2, str(p), ha="center", fontsize=9)

    # Plot 3: mean reward across all clients
    ax = axes[2]
    mean_rewards = [r["mean_reward"] for r in all_results]
    ax.plot(dims, mean_rewards, "s-", color="#e67e22", linewidth=2, markersize=8)
    ax.axvline(x=32, color="gray", linestyle="--", alpha=0.5, label="current (32)")
    best_dim = dims[mean_rewards.index(max(mean_rewards))]
    ax.axvline(x=best_dim, color="green", linestyle=":", alpha=0.7, label=f"best ({best_dim})")
    ax.set_xlabel("hidden_dim")
    ax.set_ylabel("Mean Reward (all clients)")
    ax.set_title("Overall Performance vs hidden_dim")
    ax.set_xticks(dims)
    ax.legend(fontsize=8)

    plt.suptitle("hidden_dim Ablation — SharedEncoderHead", fontsize=13)
    plt.tight_layout()
    out = "artifacts/qe_sac_fl/ablation/hidden_dim_ablation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {out}")
    plt.close()


def main():
    print("\n" + "="*60)
    print("  HIDDEN DIM ABLATION")
    print("  Testing: hidden_dim ∈ {16, 32, 64, 128}")
    print("  100 rounds × 500 steps = 50K steps/client")
    print("="*60)

    hidden_dims = [16, 32, 64, 128]
    all_results = []

    for hd in hidden_dims:
        print(f"\n--- hidden_dim = {hd} ---")
        result = run_one(hd, n_rounds=100)
        all_results.append(result)

    # Save
    out_path = "artifacts/qe_sac_fl/ablation/hidden_dim_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary table
    print(f"\n{'='*72}")
    print(f"  {'hidden_dim':>10} {'head_params':>12} {'bytes/round':>12} "
          f"{'13-bus':>10} {'34-bus':>10} {'123-bus':>10}")
    print(f"  {'-'*70}")
    best_mean = max(r["mean_reward"] for r in all_results)
    for r in all_results:
        rewards = list(r["final_rewards"].values())
        marker = " ← best" if r["mean_reward"] == best_mean else ""
        marker += " ← current" if r["hidden_dim"] == 32 else ""
        print(f"  {r['hidden_dim']:>10} {r['head_params']:>12,} "
              f"{r['bytes_per_round']:>12,} "
              f"{rewards[0] if len(rewards)>0 else 0:>10.1f} "
              f"{rewards[1] if len(rewards)>1 else 0:>10.1f} "
              f"{rewards[2] if len(rewards)>2 else 0:>10.1f}{marker}")
    print(f"{'='*72}")
    print(f"\n  Results saved → {out_path}")

    plot_results(all_results)


if __name__ == "__main__":
    main()
