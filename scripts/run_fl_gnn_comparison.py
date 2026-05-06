"""
MLP vs GNN LocalEncoder comparison on 3-topology FL.

Two encoder variants, same FL framework, same federated params (288):
  - mlp: AlignedQESACAgent   (current approach)
  - gnn: GNNAlignedQESACAgent (topology-aware, handles dynamic faults)

Conditions per variant:
  local_only | naive_fl | aligned_fl

Topologies: 13-bus (obs=42), 34-bus (obs=105), 123-bus (obs=372)
Seeds: 0, 1, 2
Rounds: 50 × 1000 steps = 50K steps/client

Run:
    cd /root/power-system
    python -u scripts/run_fl_gnn_comparison.py > logs/fl_gnn_comparison.log 2>&1 &

Output:
    artifacts/qe_sac_fl_gnn/{encoder}_{condition}_seed{N}.json
    artifacts/qe_sac_fl_gnn/comparison_summary.json
"""

import sys, os, json, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

OUT_DIR = os.path.join(ROOT, "artifacts", "qe_sac_fl_gnn")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

import numpy as np
import torch
import torch.nn.functional as F

from src.qe_sac_fl.fed_config import FedConfig, ClientConfig
from src.qe_sac_fl.aligned_agent import AlignedQESACAgent, GNNAlignedQESACAgent
from src.qe_sac_fl.aligned_encoder import fedavg_shared_head
from src.qe_sac_fl.federated_trainer import _make_env
from src.qe_sac.env_utils import _VVCEnvBase

N_ROUNDS     = 50
LOCAL_STEPS  = 1_000
WARMUP_STEPS = 500
BATCH_SIZE   = 256
LR           = 1e-4
HIDDEN_DIM   = 32
SERVER_MOM   = 0.3
SEEDS        = [0, 1, 2]
LOG_INTERVAL = 10
N_GPUS       = torch.cuda.device_count()
DEVICES      = [f"cuda:{i}" if i < N_GPUS else "cpu" for i in range(3)]

CLIENTS = [
    ClientConfig("Utility_A_13bus",  "13bus_fl",  43,  132, reward_scale=50.0),
    ClientConfig("Utility_B_34bus",  "34bus_fl",  113, 132, reward_scale=10.0),
    ClientConfig("Utility_C_123bus", "123bus_fl", 349, 132, reward_scale=750.0),
]


def log(msg): print(msg, flush=True)


def make_mlp_agent(cfg, device):
    env = _make_env(cfg.env_id, cfg.seed, cfg.reward_scale)
    nvec = list(map(int, env.action_space.nvec))
    env.close()
    return AlignedQESACAgent(
        obs_dim=cfg.obs_dim, device_dims=nvec,
        lr=LR, hidden_dim=HIDDEN_DIM, device=device,
        buffer_size=100_000,
    )


def make_gnn_agent(cfg, device):
    env = _make_env(cfg.env_id, cfg.seed, cfg.reward_scale)
    nvec     = list(map(int, env.action_space.nvec))
    n_buses  = env.unwrapped._n_buses if hasattr(env, 'unwrapped') else env._n_buses
    branches = env.unwrapped._branches if hasattr(env, 'unwrapped') else env._branches
    env.close()
    return GNNAlignedQESACAgent(
        obs_dim=cfg.obs_dim, n_buses=n_buses, branches=branches,
        device_dims=nvec, lr=LR, hidden_dim=HIDDEN_DIM, device=device,
        buffer_size=100_000,
    )


def rollout(env, agent, n_steps, warmup=False):
    obs, _ = env.reset()
    ep_rewards, ep_r = [], 0.0
    grad_norms = []
    for _ in range(n_steps):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(agent.device)
        if warmup or agent._size < WARMUP_STEPS:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs_t)
        next_obs, reward, te, tr, _ = env.step(action)
        agent.store(obs, action, reward, next_obs, te or tr)
        if not warmup and agent._size >= WARMUP_STEPS:
            logs = agent.update(BATCH_SIZE)
            # Track VQC gradient norm
            if logs and agent._grad_steps % 50 == 0:
                vqc_grad = agent.actor.vqc.weights.grad
                if vqc_grad is not None:
                    grad_norms.append(float(vqc_grad.norm().item()))
        ep_r += reward
        obs = next_obs
        if te or tr:
            ep_rewards.append(ep_r)
            ep_r = 0.0
            obs, _ = env.reset()
    mean_r = float(np.mean(ep_rewards)) if ep_rewards else float('nan')
    mean_gn = float(np.mean(grad_norms)) if grad_norms else float('nan')
    return mean_r, mean_gn


def do_fedavg(agents, server_w):
    all_shared = [a.actor.cae.get_shared_weights() for a in agents]
    all_vqc    = [a.actor.vqc.weights.data.clone().cpu() for a in agents]
    avg_shared = fedavg_shared_head(all_shared, aggregation="uniform")
    avg_vqc    = torch.stack(all_vqc).mean(0)
    if server_w is None:
        ns, nv = avg_shared, avg_vqc
    else:
        ns = {k: SERVER_MOM*server_w["s"][k] + (1-SERVER_MOM)*avg_shared[k]
              for k in avg_shared}
        nv = SERVER_MOM*server_w["v"] + (1-SERVER_MOM)*avg_vqc
    for a in agents:
        a.actor.cae.set_shared_weights(ns)
        with torch.no_grad():
            a.actor.vqc.weights.copy_(nv.to(a.device))
    return {"s": ns, "v": nv}


def run_one(encoder_type, condition, seed):
    tag = f"{encoder_type}_{condition}_seed{seed}"
    out_path = os.path.join(OUT_DIR, f"{tag}.json")
    if os.path.exists(out_path):
        log(f"  skip {tag} (exists)")
        with open(out_path) as f: return json.load(f)

    log(f"\n  [{tag}]")
    torch.manual_seed(seed); np.random.seed(seed)

    envs   = [_make_env(c.env_id, seed + i, c.reward_scale) for i, c in enumerate(CLIENTS)]
    agents = []
    for i, c in enumerate(CLIENTS):
        c.seed = seed + i
        if encoder_type == "mlp":
            agents.append(make_mlp_agent(c, DEVICES[i]))
        else:
            agents.append(make_gnn_agent(c, DEVICES[i]))

    # Warmup
    for env, agent in zip(envs, agents):
        rollout(env, agent, WARMUP_STEPS, warmup=True)
    for env, agent in zip(envs, agents):
        agent.pretrain_cae(env, n_collect=1000, n_train_steps=50)

    server_w = None
    names = [c.name for c in CLIENTS]
    hist  = {n: {"reward": [], "grad_norm": []} for n in names}

    for rnd in range(1, N_ROUNDS + 1):
        for env, agent, name in zip(envs, agents, names):
            r, gn = rollout(env, agent, LOCAL_STEPS)
            hist[name]["reward"].append(r)
            hist[name]["grad_norm"].append(gn)

        if condition in ("naive_fl", "aligned_fl"):
            server_w = do_fedavg(agents, server_w)

        if rnd % LOG_INTERVAL == 0:
            rstr = "  ".join(
                f"{n.split('_')[1]}:{hist[n]['reward'][-1]:+.1f}"
                for n in names)
            log(f"    round {rnd:>3}/{N_ROUNDS}  |  {rstr}")

    final = {n: float(np.mean([x for x in hist[n]["reward"][-10:]
                               if not np.isnan(x)])) for n in names}
    for env in envs: env.close()

    payload = {
        "encoder": encoder_type, "condition": condition, "seed": seed,
        "final_rewards": final, "history": hist,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    log(f"  saved → {out_path}")
    return payload


def summarise():
    from scipy import stats as sc
    encoders   = ["mlp", "gnn"]
    conditions = ["local_only", "naive_fl", "aligned_fl"]
    names      = [c.name for c in CLIENTS]
    short      = ["A(13)", "B(34)", "C(123)"]

    # Collect all results
    data = {enc: {cond: {n: [] for n in names} for cond in conditions}
            for enc in encoders}
    for enc in encoders:
        for cond in conditions:
            for seed in SEEDS:
                p = os.path.join(OUT_DIR, f"{enc}_{cond}_seed{seed}.json")
                if not os.path.exists(p): continue
                with open(p) as f: d = json.load(f)
                for n in names:
                    if n in d["final_rewards"]:
                        data[enc][cond][n].append(d["final_rewards"][n])

    log("\n" + "="*72)
    log("  MLP vs GNN LocalEncoder — Final Reward Comparison")
    log("="*72)
    log(f"  {'':22}  {'A (13bus)':>16}  {'B (34bus)':>16}  {'C (123bus)':>16}")
    log("  " + "-"*68)

    for enc in encoders:
        for cond in conditions:
            row = f"  {enc+'/'+cond:<22}"
            for n in names:
                v = data[enc][cond][n]
                row += f"  {np.mean(v):+.2f}±{np.std(v):.2f}      " if v else "  —              "
            log(row)
        log("  " + "-"*68)

    log("\n  Effect: GNN aligned_fl vs MLP aligned_fl")
    for n, s in zip(names, short):
        gnn_v = data["gnn"]["aligned_fl"][n]
        mlp_v = data["mlp"]["aligned_fl"][n]
        if len(gnn_v) >= 2 and len(mlp_v) >= 2:
            delta = np.mean(gnn_v) - np.mean(mlp_v)
            mn = min(len(gnn_v), len(mlp_v))
            d  = delta / (np.std(np.array(gnn_v[:mn]) - np.array(mlp_v[:mn])) + 1e-9)
            log(f"    {s}: Δ={delta:+.3f}  d={d:+.2f}  "
                f"({'GNN better' if delta > 0 else 'MLP better'})")

    # Save summary
    summary = {enc: {cond: {n: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                                "n": len(v), "seeds": v}
                             for n, v in vd.items() if v}
                     for cond, vd in cd.items()}
               for enc, cd in data.items()}
    out = os.path.join(OUT_DIR, "comparison_summary.json")
    with open(out, "w") as f: json.dump(summary, f, indent=2)
    log(f"\n  summary → {out}")
    log("="*72)


def main():
    log("="*60)
    log("  MLP vs GNN Encoder — QE-SAC-FL Comparison")
    log(f"  rounds: {N_ROUNDS} × {LOCAL_STEPS} = {N_ROUNDS*LOCAL_STEPS:,} steps")
    log(f"  seeds:  {SEEDS}  |  GPUs: {N_GPUS}")
    log(f"  both variants federate: 288 params = 1.1 KB/round")
    log("="*60)

    t0 = time.time()
    for enc in ["mlp", "gnn"]:
        for cond in ["local_only", "naive_fl", "aligned_fl"]:
            for seed in SEEDS:
                run_one(enc, cond, seed)

    summarise()
    log(f"\n  total: {(time.time()-t0)/3600:.1f} h  — DONE")


if __name__ == "__main__":
    main()
