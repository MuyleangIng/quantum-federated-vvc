"""
Q-GNN-SAC vs QE-SAC vs Lin et al. (2025) — OpenDSS 13-bus comparison.

Proves GNN+VQC beats MLP+VQC on real OpenDSS topology.

Run: python -u scripts/run_gnn_opendss.py > logs/gnn_opendss.log 2>&1 &
Output: artifacts/gnn_opendss/summary.json
"""
import os, sys, time, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import torch

from src.qe_sac.env_opendss import VVCEnvOpenDSS
from src.qe_sac_fl.aligned_agent import AlignedQESACAgent, GNNAlignedQESACAgent
from src.qe_sac.env_utils import _IEEE13_BRANCHES

OUT_DIR = os.path.join(ROOT, "artifacts", "gnn_opendss")
os.makedirs(OUT_DIR, exist_ok=True)

N_STEPS     = 50_000
BATCH_SIZE  = 512
LR          = 3e-4
GAMMA       = 0.99
TAU         = 0.005
ALPHA       = 0.1
BUFFER_SIZE = 200_000
SEEDS       = [0, 1, 2, 3, 4]

import argparse
_p = argparse.ArgumentParser()
_p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
_p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
_args, _ = _p.parse_known_args()
SEEDS  = _args.seeds
DEVICE = _args.device

# VVCEnvOpenDSS: 2 caps + 3 regs + 1 battery → [2,2,33,33,33,33]
NVEC    = [2, 2, 33, 33, 33, 33]
N_BUSES = 13


def get_obs_dim():
    env = VVCEnvOpenDSS(seed=0)
    d = env.observation_space.shape[0]
    env.close()
    return d


def eval_agent(env, agent, is_gnn=False, n_eps=10):
    rewards, viols = [], []
    for _ in range(n_eps):
        obs, _ = env.reset()
        ep_r, ep_v, done = 0.0, 0, False
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            if is_gnn:
                action = agent.actor.select_action(obs_t, deterministic=True)
            else:
                action = agent.actor.select_action(obs_t, deterministic=True)
            obs, r, te, tr, info = env.step(action)
            ep_r += r
            ep_v += info.get("v_viol", 0)
            done = te or tr
        rewards.append(ep_r)
        viols.append(ep_v)
    return float(np.mean(rewards)), float(np.mean(viols))


def run_mlp(seed):
    print(f"\n  [MLP+VQC] seed={seed}", flush=True)
    torch.manual_seed(seed); np.random.seed(seed)
    obs_dim = get_obs_dim()
    train_env = VVCEnvOpenDSS(seed=seed)
    eval_env  = VVCEnvOpenDSS(seed=seed + 100)

    agent = AlignedQESACAgent(
        obs_dim=obs_dim, device_dims=list(NVEC),
        lr=LR, gamma=GAMMA, tau=TAU, alpha=ALPHA,
        buffer_size=BUFFER_SIZE, hidden_dim=32, device=DEVICE)

    obs, _ = train_env.reset()
    t0 = time.time()
    for i in range(N_STEPS):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        action = train_env.action_space.sample() if agent._size < 512 else agent.actor.select_action(obs_t)
        nobs, r, te, tr, _ = train_env.step(action)
        agent.store(obs, action, r, nobs, te or tr)
        if agent._size >= 512:
            agent.update(BATCH_SIZE)
        obs = nobs if not (te or tr) else train_env.reset()[0]

    reward, viol = eval_agent(eval_env, agent, is_gnn=False)
    train_env.close(); eval_env.close()
    print(f"    reward={reward:.3f}  viol={viol:.1f}  time={( time.time()-t0)/60:.1f}min", flush=True)
    return {"reward": reward, "viol": viol}


def run_gnn(seed):
    print(f"\n  [GNN+VQC] seed={seed}", flush=True)
    torch.manual_seed(seed); np.random.seed(seed)
    obs_dim = get_obs_dim()
    train_env = VVCEnvOpenDSS(seed=seed)
    eval_env  = VVCEnvOpenDSS(seed=seed + 100)

    branches = _IEEE13_BRANCHES

    agent = GNNAlignedQESACAgent(
        obs_dim=obs_dim, n_buses=N_BUSES, branches=branches,
        device_dims=list(NVEC),
        lr=LR, gamma=GAMMA, tau=TAU, alpha=ALPHA,
        buffer_size=BUFFER_SIZE, hidden_dim=32, device=DEVICE)

    obs, _ = train_env.reset()
    t0 = time.time()
    for i in range(N_STEPS):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        action = train_env.action_space.sample() if agent._size < 512 else agent.actor.select_action(obs_t)
        nobs, r, te, tr, _ = train_env.step(action)
        agent.store(obs, action, r, nobs, te or tr)
        if agent._size >= 512:
            agent.update(BATCH_SIZE)
        obs = nobs if not (te or tr) else train_env.reset()[0]

    reward, viol = eval_agent(eval_env, agent, is_gnn=True)
    train_env.close(); eval_env.close()
    print(f"    reward={reward:.3f}  viol={viol:.1f}  time={(time.time()-t0)/60:.1f}min", flush=True)
    return {"reward": reward, "viol": viol}


def main():
    print("="*60, flush=True)
    print("  Q-GNN-SAC vs QE-SAC — OpenDSS 13-bus", flush=True)
    print(f"  {N_STEPS} steps × {len(SEEDS)} seeds", flush=True)
    print("="*60, flush=True)

    results = {"mlp": [], "gnn": []}

    for seed in SEEDS:
        results["mlp"].append(run_mlp(seed))
        results["gnn"].append(run_gnn(seed))
        with open(f"{OUT_DIR}/seed{seed}.json", "w") as f:
            json.dump({"mlp": results["mlp"][-1], "gnn": results["gnn"][-1]}, f, indent=2)

    print(f"\n{'='*60}", flush=True)
    print(f"  RESULTS vs Lin et al. (2025)", flush=True)
    print(f"  {'Method':<20} {'Reward':>10} {'Viol':>10}", flush=True)
    print(f"  {'-'*42}", flush=True)
    print(f"  {'Lin QE-SAC':<20} {-5.39:>10.3f} {0.00:>10.2f}", flush=True)

    summary = {}
    for method in ["mlp", "gnn"]:
        rs = [r["reward"] for r in results[method]]
        vs = [r["viol"]   for r in results[method]]
        label = "MLP+VQC [Ours]" if method == "mlp" else "GNN+VQC [Ours]"
        print(f"  {label:<20} {np.mean(rs):>10.3f} {np.mean(vs):>10.2f}", flush=True)
        summary[method] = {"reward_mean": float(np.mean(rs)), "reward_std": float(np.std(rs)),
                           "viol_mean": float(np.mean(vs))}

    with open(f"{OUT_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved → {OUT_DIR}/summary.json", flush=True)


if __name__ == "__main__":
    main()
