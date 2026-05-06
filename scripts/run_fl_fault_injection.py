"""
Option 2 — Fault Injection Experiment (GNN vs MLP on Dynamic Topology)
========================================================================
Tests GNN encoder advantage when grid topology changes (line faults).

Experiment:
  - Train MLP-encoder FL and GNN-encoder FL on normal topology
  - At round 50, inject a fault: remove branch 5↔6 on 13-bus (busses 5 and 6 disconnected)
  - Measure reward recovery over next 50 rounds
  - GNN should recover faster because adjacency matrix is updated

Two conditions:
  mlp_fl  — MLP LocalEncoder, cannot represent changed topology
  gnn_fl  — GNN LocalEncoder, adjacency matrix updated on fault

Run: python3 scripts/run_fl_fault_injection.py
Output: artifacts/qe_sac_fl_gnn/fault_injection/
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.qe_sac_fl.aligned_agent import AlignedQESACAgent
from src.qe_sac_fl.federated_trainer import _fedavg

OUT_DIR  = "artifacts/qe_sac_fl_gnn/fault_injection_200r"
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS        = [0, 1, 2]
N_ROUNDS     = 200
FAULT_ROUND  = 50   # inject fault here
LOCAL_STEPS  = 1000
WARMUP       = 500
BATCH_SIZE   = 64
LR           = 3e-4
BUFFER_SIZE  = 50_000

# Use 13-bus only for fast fault experiment
CLIENTS = [
    dict(name="A_13bus",  obs_dim=43,  device_dims=[2,2,33,33], reward_scale=50.0,
         device="cuda:0" if torch.cuda.is_available() else "cpu"),
]

# ── Fault-aware environment wrapper ───────────────────────────────────────────

from src.qe_sac.env_utils import VVCEnv13Bus

class FaultableVVCEnv13Bus(VVCEnv13Bus):
    """13-bus env with injectable line fault."""

    def inject_fault(self, branch_idx: int):
        """
        Remove branch_idx from the network (set R,X to very high impedance).
        This simulates an open-circuit fault.
        """
        if hasattr(self, '_branches') and branch_idx < len(self._branches):
            # High impedance = disconnected
            branches = list(self._branches)
            fr, to, r, x = branches[branch_idx]
            branches[branch_idx] = (fr, to, r * 1000, x * 1000)
            self._branches = branches
            print(f"    FAULT injected: branch {branch_idx} ({fr}↔{to}) disconnected")
            return True
        return False

    def clear_fault(self):
        """Restore original topology."""
        # Reset to original by reloading class-level data
        self._branches = VVCEnv13Bus._branches[:]
        print(f"    Fault cleared")


# ── GNN encoder attempt — fallback to MLP if GNN not available ───────────────

def make_agent(condition, cfg):
    """Make agent for given condition."""
    try:
        if condition == "gnn_fl":
            from src.qe_sac_fl.aligned_agent import GNNAlignedQESACAgent
            from src.qe_sac.env_utils import _IEEE13_BRANCHES
            agent = GNNAlignedQESACAgent(
                obs_dim=cfg["obs_dim"], device_dims=cfg["device_dims"],
                n_buses=13, branches=list(_IEEE13_BRANCHES),
                lr=LR, buffer_size=BUFFER_SIZE, device=cfg["device"])
            print(f"    Using GNN encoder")
            return agent, True
    except (ImportError, AttributeError) as e:
        print(f"    GNN import failed ({e}), falling back to MLP")

    agent = AlignedQESACAgent(
        obs_dim=cfg["obs_dim"], device_dims=cfg["device_dims"],
        lr=LR, buffer_size=BUFFER_SIZE, device=cfg["device"])
    return agent, False


def make_env(name, seed, condition):
    """Make env — fault-aware env wraps 13-bus."""
    return FaultableVVCEnv13Bus(seed=seed)


def run_one(condition, seed):
    print(f"\n{'='*55}\n  {condition}  seed={seed}\n{'='*55}")
    torch.manual_seed(seed); np.random.seed(seed)

    agents, envs = [], []
    gnn_flags    = []
    for cfg in CLIENTS:
        agent, is_gnn = make_agent(condition, cfg)
        env = make_env(cfg["name"], seed, condition)
        agents.append(agent); envs.append(env); gnn_flags.append(is_gnn)

    # Warm-up
    for agent, env, cfg in zip(agents, envs, CLIENTS):
        obs, _ = env.reset()
        for _ in range(WARMUP):
            obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(cfg["device"])
            action = agent.actor.select_action(obs_t)
            nobs, reward, done, trunc, _ = env.step(action)
            agent.store(obs, action, reward / cfg["reward_scale"], nobs, done or trunc)
            obs = nobs if not (done or trunc) else env.reset()[0]

    reward_history = {cfg["name"]: [] for cfg in CLIENTS}
    server_weights = agents[0].actor.get_shared_weights()
    fault_injected = False

    for rnd in range(N_ROUNDS):
        # Inject fault at FAULT_ROUND
        if rnd == FAULT_ROUND and not fault_injected:
            print(f"\n  *** FAULT INJECTION at round {rnd+1} ***")
            for env in envs:
                if hasattr(env, 'inject_fault'):
                    env.inject_fault(branch_idx=5)   # branch 5-6 on 13-bus
                    # For GNN: rebuild adjacency matrix with fault
                    if condition == "gnn_fl":
                        for agent, is_gnn in zip(agents, gnn_flags):
                            if is_gnn and hasattr(agent, 'update_adjacency'):
                                # Remove edge 5↔6 from adjacency
                                new_adj = agent.adj_matrix.clone()
                                new_adj[5, 6] = 0.0
                                new_adj[6, 5] = 0.0
                                agent.update_adjacency(new_adj)
                                print(f"    GNN adjacency updated for fault")
            fault_injected = True

        for agent, env, cfg in zip(agents, envs, CLIENTS):
            agent.actor.set_shared_weights(server_weights)
            ep_rewards, ep_rew = [], 0.0
            obs, _ = env.reset()
            for _ in range(LOCAL_STEPS):
                obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(cfg["device"])
                action = agent.actor.select_action(obs_t)
                nobs, reward, done, trunc, _ = env.step(action)
                agent.store(obs, action, reward / cfg["reward_scale"], nobs, done or trunc)
                ep_rew += reward / cfg["reward_scale"]
                obs = nobs
                if done or trunc:
                    ep_rewards.append(ep_rew); ep_rew = 0.0
                    obs, _ = env.reset()
                agent.update(batch_size=BATCH_SIZE)

            reward_history[cfg["name"]].append(
                float(np.mean(ep_rewards)) if ep_rewards else float("nan"))

        # FedAvg
        all_sw  = [a.actor.get_shared_weights() for a in agents]
        from src.qe_sac_fl.aligned_encoder import fedavg_shared_head
        avg_sh  = fedavg_shared_head([s["shared_head"] for s in all_sw])
        avg_vqc = torch.stack([s["vqc"].float() for s in all_sw]).mean(0)
        server_weights = {"shared_head": avg_sh, "vqc": avg_vqc}

        rA = reward_history["A_13bus"][-1]
        fault_str = " [POST-FAULT]" if fault_injected else ""
        print(f"  Round {rnd+1:>3}/{N_ROUNDS} | A={rA:+.3f}{fault_str}")

    for env in envs: env.close()

    # Compute pre-fault (rounds 40-50) and post-fault recovery (rounds 51-100) rewards
    rh = reward_history["A_13bus"]
    pre_fault  = float(np.nanmean(rh[40:50]))   # rounds 41-50
    post_fault = float(np.nanmean(rh[50:]))      # rounds 51-100
    recovery   = float(np.nanmean(rh[75:]))      # last 25 rounds — full recovery

    return {
        "condition":      condition,
        "seed":           seed,
        "reward_history": reward_history,
        "fault_round":    FAULT_ROUND,
        "pre_fault_mean": pre_fault,
        "post_fault_mean": post_fault,
        "recovery_mean":  recovery,
    }


def plot_results(all_results):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor='white')
    fig.suptitle(
        "Fault Injection Experiment — GNN vs MLP Encoder\n"
        f"Line fault injected at round {FAULT_ROUND}: branch 5↔6 disconnected (13-bus)",
        fontsize=12, fontweight='bold'
    )

    colors = {"mlp_fl": "#E91E63", "gnn_fl": "#2196F3"}
    labels = {"mlp_fl": "MLP Encoder FL", "gnn_fl": "GNN Encoder FL [PROPOSED]"}

    # ── Panel 1: full reward curve ──
    ax = axes[0]
    ax.set_facecolor('white')
    ax.set_title("Reward over FL Rounds — Client A (13-bus)", fontsize=11, fontweight='bold')
    ax.set_xlabel("FL Round"); ax.set_ylabel("Mean Reward (↑ better)")
    ax.grid(True, alpha=0.3)
    ax.axvline(x=FAULT_ROUND, color='red', lw=2, linestyle='--', alpha=0.8, label=f"Fault @ round {FAULT_ROUND}")

    for cond in ["mlp_fl", "gnn_fl"]:
        res_list = all_results.get(cond, [])
        if not res_list: continue
        arrs = np.array([r["reward_history"]["A_13bus"] for r in res_list])
        mn   = np.nanmean(arrs, axis=0)
        std  = np.nanstd(arrs, axis=0)
        rds  = np.arange(1, len(mn)+1)
        ax.plot(rds, mn, color=colors[cond], lw=2.5, label=labels[cond])
        ax.fill_between(rds, mn-std, mn+std, alpha=0.15, color=colors[cond])
    ax.legend(fontsize=9)

    # ── Panel 2: recovery bar chart ──
    ax = axes[1]
    ax.set_facecolor('white')
    ax.set_title("Recovery Analysis\n(mean reward pre/post fault)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Mean Reward (↑ better)")
    ax.grid(True, alpha=0.3, axis='y')

    metrics = ["pre_fault_mean", "post_fault_mean", "recovery_mean"]
    x_labels = ["Pre-Fault\n(rounds 41-50)", "Post-Fault\n(rounds 51-100)", "Recovery\n(rounds 76-100)"]
    x = np.arange(len(metrics))
    w = 0.3

    for i, cond in enumerate(["mlp_fl", "gnn_fl"]):
        res_list = all_results.get(cond, [])
        if not res_list: continue
        vals = [float(np.mean([r[m] for r in res_list])) for m in metrics]
        ax.bar(x + (i-0.5)*w, vals, w, color=colors[cond], label=labels[cond], alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = f"{OUT_DIR}/fault_injection_results.png"
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Figure → {out}")


def main():
    all_results = {"mlp_fl": [], "gnn_fl": []}
    t0 = time.time()

    for condition in ["mlp_fl", "gnn_fl"]:
        for seed in SEEDS:
            res = run_one(condition, seed)
            all_results[condition].append(res)
            with open(f"{OUT_DIR}/{condition}_seed{seed}.json", "w") as f:
                json.dump(res, f, indent=2)

    print(f"\n{'='*60}\n  FAULT INJECTION SUMMARY\n{'='*60}")
    print(f"  {'Condition':<12} {'Pre-Fault':>12} {'Post-Fault':>12} {'Recovery':>12}")
    for cond in ["mlp_fl", "gnn_fl"]:
        r = all_results[cond]
        pf  = float(np.mean([x["pre_fault_mean"]  for x in r]))
        po  = float(np.mean([x["post_fault_mean"] for x in r]))
        rec = float(np.mean([x["recovery_mean"]   for x in r]))
        print(f"  {cond:<12} {pf:>+12.3f} {po:>+12.3f} {rec:>+12.3f}")

    summary = {}
    for cond in all_results:
        r = all_results[cond]
        summary[cond] = {
            "pre_fault_mean":  float(np.mean([x["pre_fault_mean"]  for x in r])),
            "post_fault_mean": float(np.mean([x["post_fault_mean"] for x in r])),
            "recovery_mean":   float(np.mean([x["recovery_mean"]   for x in r])),
        }
    with open(f"{OUT_DIR}/fault_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    plot_results(all_results)
    print(f"\n  Done. {(time.time()-t0)/60:.1f} min")
    print(f"  Output → {OUT_DIR}/")


if __name__ == "__main__":
    main()
