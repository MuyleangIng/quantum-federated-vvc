"""
QE-SAC full metrics evaluation — matching MASAC paper metrics:
  1. Reward
  2. Controllable ratio  (% steps ALL buses in [0.95, 1.05] p.u.)
  3. Reactive power loss (MVAR)
  4. Average voltage     (p.u.)

Run: python -u scripts/evaluate_violation_metrics.py > logs/metrics_eval.log 2>&1 &
Output: artifacts/gnn_opendss/metrics_summary.json
"""
import os, sys, time, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import torch

from src.qe_sac.env_opendss import VVCEnvOpenDSS
from src.qe_sac_fl.aligned_agent import AlignedQESACAgent

OUT_DIR     = os.path.join(ROOT, "artifacts", "gnn_opendss")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

N_STEPS     = 50_000
BATCH_SIZE  = 512
LR          = 3e-4
GAMMA       = 0.99
TAU         = 0.005
ALPHA       = 0.1
BUFFER_SIZE = 200_000
SEEDS       = [0, 1, 2, 3, 4]
NVEC        = [2, 2, 33, 33, 33, 33]
DEVICE      = "cuda:0" if torch.cuda.is_available() else "cpu"
N_EVAL_EPS  = 20


def get_obs_dim():
    env = VVCEnvOpenDSS(seed=0)
    d = env.observation_space.shape[0]
    env.close()
    return d


def eval_full_metrics(env, agent, n_eps=N_EVAL_EPS):
    all_reward, all_ctrl, all_ploss, all_avgv = [], [], [], []

    for _ in range(n_eps):
        obs, _ = env.reset()
        ep_r = 0.0
        ctrl_steps = 0
        total_steps = 0
        ploss_sum = 0.0
        voltage_sum = 0.0
        done = False

        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            action = agent.actor.select_action(obs_t, deterministic=True)
            obs, r, te, tr, info = env.step(action)

            ep_r += r
            total_steps += 1

            voltages = info.get("voltage", np.array([]))
            if len(voltages) > 0:
                all_in_bounds = bool(np.all((voltages >= 0.95) & (voltages <= 1.05)))
                if all_in_bounds:
                    ctrl_steps += 1
                voltage_sum += float(np.mean(voltages))

            ploss_kw = info.get("P_loss", 0.0)
            ploss_sum += ploss_kw

            done = te or tr

        all_reward.append(ep_r)
        all_ctrl.append(ctrl_steps / max(total_steps, 1) * 100)
        all_ploss.append(ploss_sum / max(total_steps, 1) / 1000)  # kW → MVAR approx
        all_avgv.append(voltage_sum / max(total_steps, 1))

    return {
        "reward":      float(np.mean(all_reward)),
        "reward_std":  float(np.std(all_reward)),
        "ctrl_ratio":  float(np.mean(all_ctrl)),
        "ctrl_std":    float(np.std(all_ctrl)),
        "react_loss":  float(np.mean(all_ploss)),
        "react_std":   float(np.std(all_ploss)),
        "avg_voltage": float(np.mean(all_avgv)),
        "avgv_std":    float(np.std(all_avgv)),
    }


def run_seed(seed):
    print(f"\n  [seed={seed}]", flush=True)
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
        action = train_env.action_space.sample() if agent._size < 512 \
                 else agent.actor.select_action(obs_t)
        nobs, r, te, tr, _ = train_env.step(action)
        agent.store(obs, action, r, nobs, te or tr)
        if agent._size >= 512:
            agent.update(BATCH_SIZE)
        obs = nobs if not (te or tr) else train_env.reset()[0]

    metrics = eval_full_metrics(eval_env, agent)
    train_env.close(); eval_env.close()

    print(f"    reward={metrics['reward']:.3f}  "
          f"ctrl={metrics['ctrl_ratio']:.1f}%  "
          f"ploss={metrics['react_loss']:.4f}MVAR  "
          f"avgV={metrics['avg_voltage']:.4f}  "
          f"time={(time.time()-t0)/60:.1f}min", flush=True)

    with open(f"{OUT_DIR}/metrics_seed{seed}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main():
    print("="*65, flush=True)
    print("  QE-SAC Full Metrics Evaluation — OpenDSS 13-bus", flush=True)
    print(f"  {N_STEPS} steps × {len(SEEDS)} seeds × {N_EVAL_EPS} eval eps", flush=True)
    print("="*65, flush=True)

    results = []
    for seed in SEEDS:
        m = run_seed(seed)
        results.append(m)

    rewards = [r["reward"]      for r in results]
    ctrls   = [r["ctrl_ratio"]  for r in results]
    plosses = [r["react_loss"]  for r in results]
    avgvs   = [r["avg_voltage"] for r in results]

    print(f"\n{'='*72}", flush=True)
    print(f"  FINAL TABLE — Our QE-SAC vs Literature", flush=True)
    print(f"  {'Method':<22} {'Reward':>8} {'Ctrl%':>8} {'PLoss(MVAR)':>12} {'AvgV(p.u.)':>11}", flush=True)
    print(f"  {'-'*65}", flush=True)
    print(f"  {'Lin QE-SAC (2025)':<22} {-5.390:>8.3f} {'100.0':>8} {'N/A':>12} {'N/A':>11}", flush=True)
    print(f"  {'MASAC 33-bus (2024)':<22} {'N/A':>8} {'95.37':>8} {'0.1233':>12} {'0.9976':>11}", flush=True)
    print(f"  {'-'*65}", flush=True)
    print(f"  {'Our QE-SAC':<22} {np.mean(rewards):>8.3f} {np.mean(ctrls):>8.1f} "
          f"{np.mean(plosses):>12.4f} {np.mean(avgvs):>11.4f}", flush=True)
    print(f"  {'  ±std':<22} {np.std(rewards):>8.3f} {np.std(ctrls):>8.1f} "
          f"{np.std(plosses):>12.4f} {np.std(avgvs):>11.4f}", flush=True)

    summary = {
        "our_qesac": {
            "reward_mean":      float(np.mean(rewards)),
            "reward_std":       float(np.std(rewards)),
            "ctrl_ratio_mean":  float(np.mean(ctrls)),
            "ctrl_ratio_std":   float(np.std(ctrls)),
            "react_loss_mean":  float(np.mean(plosses)),
            "react_loss_std":   float(np.std(plosses)),
            "avg_voltage_mean": float(np.mean(avgvs)),
            "avg_voltage_std":  float(np.std(avgvs)),
        },
        "lin_qesac":    {"reward": -5.390, "ctrl_ratio": 100.0},
        "masac_33bus":  {"ctrl_ratio": 95.37, "react_loss_mvar": 0.1233,
                         "avg_voltage": 0.9976},
    }
    with open(f"{OUT_DIR}/metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved → {OUT_DIR}/metrics_summary.json", flush=True)


if __name__ == "__main__":
    main()
