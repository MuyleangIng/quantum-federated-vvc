"""
4-Topology Federated QE-SAC-FL
================================
Extends the 3-client experiment to 4 heterogeneous IEEE topologies:
  Client A — 13-bus   (obs=42,  radial distribution)
  Client B — 34-bus   (obs=105, meshed distribution)
  Client C — 57-bus   (obs=174, sub-transmission)
  Client D — 123-bus  (obs=372, large distribution)

All share the same 280 federated params (SharedHead 264 + VQC 16).
Demonstrates scalability across topology diversity.

Run: python3 scripts/run_fl_4topology.py
Output: artifacts/qe_sac_fl/four_topology/
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.qe_sac.env_utils import VVCEnv13Bus
from src.qe_sac_fl.env_34bus import VVCEnv34BusFL, VVCEnv123BusFL
from src.qe_sac_fl.env_57bus import VVCEnv57BusFL
from src.qe_sac_fl.aligned_encoder import fedavg_shared_head
from src.qe_sac_fl.aligned_agent import AlignedQESACAgent

SEEDS       = [0, 1, 2]
N_ROUNDS    = 100
LOCAL_STEPS = 1000
WARMUP      = 500
BATCH_SIZE  = 64
LR          = 3e-4
BUFFER_SIZE = 50_000
OUT_DIR     = "artifacts/qe_sac_fl/four_topology"
os.makedirs(OUT_DIR, exist_ok=True)

CLIENTS = [
    dict(name="A_13bus",  obs_dim=43,  device_dims=[2,2,33,33], reward_scale=50.0,
         device="cuda:0" if torch.cuda.is_available() else "cpu"),
    dict(name="B_34bus",  obs_dim=113, device_dims=[2,2,33,33,33], reward_scale=10.0,
         device="cuda:0" if torch.cuda.is_available() else "cpu"),
    dict(name="C_57bus",  obs_dim=174, device_dims=[2,2,33], reward_scale=30.0,
         device="cuda:0" if torch.cuda.is_available() else "cpu"),
    dict(name="D_123bus", obs_dim=349, device_dims=[2,2,33], reward_scale=750.0,
         device="cuda:0" if torch.cuda.is_available() else "cpu"),
]


def make_env(name, seed):
    if "13"  in name: return VVCEnv13Bus(seed=seed)
    if "34"  in name: return VVCEnv34BusFL(seed=seed)
    if "57"  in name: return VVCEnv57BusFL(seed=seed)
    return VVCEnv123BusFL(seed=seed)


def run_one(seed):
    print(f"\n{'='*55}\n  4-topology FL  seed={seed}\n{'='*55}")
    torch.manual_seed(seed); np.random.seed(seed)

    agents, envs = [], []
    for cfg in CLIENTS:
        agent = AlignedQESACAgent(
            obs_dim=cfg["obs_dim"], device_dims=cfg["device_dims"],
            lr=LR, buffer_size=BUFFER_SIZE, device=cfg["device"])
        env = make_env(cfg["name"], seed)
        agents.append(agent); envs.append(env)

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

    for rnd in range(N_ROUNDS):
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
        avg_sh  = fedavg_shared_head([s["shared_head"] for s in all_sw])
        avg_vqc = torch.stack([s["vqc"].float() for s in all_sw]).mean(0)
        server_weights = {"shared_head": avg_sh, "vqc": avg_vqc}

        print(f"  Round {rnd+1:>3}/{N_ROUNDS} | "
              + "  ".join(f"{cfg['name'].split('_')[0]}={reward_history[cfg['name']][-1]:+.2f}"
                          for cfg in CLIENTS))

    for env in envs: env.close()

    final = {cfg["name"]: float(np.nanmean(reward_history[cfg["name"]][-10:]))
             for cfg in CLIENTS}
    return {"seed": seed, "reward_history": reward_history, "final": final}


def plot_results(all_results):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor='white')
    fig.suptitle(
        "4-Topology Federated QE-SAC-FL\n"
        "280 federated params shared across 13/34/57/123-bus clients",
        fontsize=13, fontweight='bold'
    )
    colors  = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
    clients = [cfg["name"] for cfg in CLIENTS]
    titles  = ["Client A (13-bus)", "Client B (34-bus)",
               "Client C (57-bus)", "Client D (123-bus)"]

    for ax, cname, title, color in zip(axes.flat, clients, titles, colors):
        ax.set_facecolor('white')
        ax.set_title(title, fontsize=11, fontweight='bold', color=color)
        ax.set_xlabel("FL Round"); ax.set_ylabel("Mean Reward (↑ better)")
        ax.grid(True, alpha=0.3)

        arrs = np.array([r["reward_history"][cname] for r in all_results])
        mn   = np.nanmean(arrs, axis=0)
        std  = np.nanstd(arrs, axis=0)
        rds  = np.arange(1, len(mn)+1)
        ax.plot(rds, mn,  color=color, lw=2.5, label=f"{title}")
        ax.fill_between(rds, mn-std, mn+std, alpha=0.18, color=color)

        final = float(np.nanmean(arrs[:, -10:]))
        ax.axhline(final, color=color, lw=1, linestyle='--', alpha=0.6)
        ax.text(0.97, 0.05, f"final={final:+.3f}", transform=ax.transAxes,
                ha='right', fontsize=9, color=color, fontweight='bold')

    plt.tight_layout()
    out = f"{OUT_DIR}/four_topology_rewards.png"
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Figure → {out}")


def main():
    all_results = []
    t0 = time.time()

    for seed in SEEDS:
        res = run_one(seed)
        all_results.append(res)
        with open(f"{OUT_DIR}/seed{seed}.json", "w") as f:
            json.dump(res, f, indent=2)

    print(f"\n{'='*60}\n  4-TOPOLOGY SUMMARY\n{'='*60}")
    print(f"  {'Client':<12} {'Mean Final Reward':>18}")
    for cfg in CLIENTS:
        vals = [r["final"][cfg["name"]] for r in all_results]
        print(f"  {cfg['name']:<12} {float(np.mean(vals)):>+18.3f}  ±{float(np.std(vals)):.3f}")

    summary = {cfg["name"]: {
        "mean": float(np.mean([r["final"][cfg["name"]] for r in all_results])),
        "std":  float(np.std( [r["final"][cfg["name"]] for r in all_results])),
    } for cfg in CLIENTS}
    with open(f"{OUT_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    plot_results(all_results)
    print(f"\n  Done. {(time.time()-t0)/60:.1f} min")
    print(f"  Output → {OUT_DIR}/")


if __name__ == "__main__":
    main()
