"""
FL validation on REAL OpenDSS 13-bus environment.

All 3 clients use VVCEnvOpenDSS (real 3-phase AC power flow).
Different seeds simulate different load/renewable profiles per utility.

This is the proper validation for the paper — real physics, not linearized.

Conditions:  local_only  |  naive_fl  |  aligned_fl
Clients:     seed_offset=0  |  7  |  14   (different load profiles)
Rounds:      50 × 1000 steps = 50K steps per client
Seeds:       0, 1, 2   (3 seeds — feasible on OpenDSS)

Run:
    cd /root/power-system
    python -u scripts/run_fl_opendss_real.py > logs/fl_opendss_real.log 2>&1 &

Output:
    artifacts/qe_sac_fl_opendss/seed{N}_{condition}.json
    artifacts/qe_sac_fl_opendss/verification/summary.json
    logs/fl_opendss_real.log
"""

import sys, os, json, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)   # ensure relative imports work

ARTIFACT_DIR = os.path.join(ROOT, "artifacts", "qe_sac_fl_opendss")
VERIF_DIR    = os.path.join(ARTIFACT_DIR, "verification")
LOG_DIR      = os.path.join(ROOT, "logs")
os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(VERIF_DIR,    exist_ok=True)
os.makedirs(LOG_DIR,      exist_ok=True)

import numpy as np
import torch

from src.qe_sac.env_opendss import VVCEnvOpenDSS
from src.qe_sac_fl.aligned_agent import AlignedQESACAgent
from src.qe_sac_fl.aligned_encoder import fedavg_shared_head

# ── Config ──────────────────────────────────────────────────────────────────
N_ROUNDS      = 50        # 50 rounds × 1000 steps = 50K steps/client
LOCAL_STEPS   = 1_000
WARMUP_STEPS  = 500       # shorter warmup — OpenDSS is slow
BATCH_SIZE    = 256
LR            = 1e-4      # paper-exact
GAMMA         = 0.99
TAU           = 0.005
ALPHA         = 0.2
BUFFER_SIZE   = 100_000
HIDDEN_DIM    = 32
SERVER_MOM    = 0.3
REWARD_SCALE  = 1.0       # no scaling — raw OpenDSS rewards, comparison is relative
LOG_INTERVAL  = 10
SEEDS         = [0, 1, 2]

OBS_DIM       = 48
NVEC          = [2, 2, 33, 33, 33, 33]

CLIENT_SEED_OFFSETS = [0, 7, 14]   # different load profiles per client
CLIENT_NAMES        = ["Utility_A_13bus", "Utility_B_13bus_v2", "Utility_C_13bus_v3"]

N_GPUS  = torch.cuda.device_count()
DEVICES = [f"cuda:{i}" if i < N_GPUS else "cpu" for i in range(3)]


def log(msg): print(msg, flush=True)


def make_agent(device):
    return AlignedQESACAgent(
        obs_dim=OBS_DIM, device_dims=NVEC,
        lr=LR, gamma=GAMMA, tau=TAU, alpha=ALPHA,
        buffer_size=BUFFER_SIZE, hidden_dim=HIDDEN_DIM, device=device,
    )


def rollout(env, agent, n_steps, warmup=False):
    obs, _ = env.reset()
    ep_rewards, ep_r = [], 0.0
    for _ in range(n_steps):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(agent.device)
        if warmup or agent._size < WARMUP_STEPS:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs_t, deterministic=False)
        next_obs, reward, te, tr, _ = env.step(action)
        scaled_r = reward / REWARD_SCALE
        agent.store(obs, action, scaled_r, next_obs, te or tr)
        if not warmup and agent._size >= WARMUP_STEPS:
            agent.update(BATCH_SIZE, cae_update_interval=500, cae_steps=30)
        ep_r += scaled_r
        obs = next_obs
        if te or tr:
            ep_rewards.append(ep_r)
            ep_r = 0.0
            obs, _ = env.reset()
    return float(np.mean(ep_rewards)) if ep_rewards else float('nan')


def do_fedavg(agents, server_weights):
    all_shared = [a.actor.cae.get_shared_weights() for a in agents]
    all_vqc    = [a.actor.vqc.weights.data.clone().cpu() for a in agents]
    avg_shared = fedavg_shared_head(all_shared, aggregation="uniform")
    avg_vqc    = torch.stack(all_vqc).mean(0)
    if server_weights is None:
        new_s, new_v = avg_shared, avg_vqc
    else:
        new_s = {k: SERVER_MOM * server_weights["s"][k] + (1-SERVER_MOM) * avg_shared[k]
                 for k in avg_shared}
        new_v = SERVER_MOM * server_weights["v"] + (1-SERVER_MOM) * avg_vqc
    for a in agents:
        a.actor.cae.set_shared_weights(new_s)
        with torch.no_grad():
            a.actor.vqc.weights.copy_(new_v.to(a.device))
    return {"s": new_s, "v": new_v}


def run_condition(condition, client_seeds, agent_seed):
    log(f"\n  [{condition}]  agent_seed={agent_seed}  env_seeds={client_seeds}")
    torch.manual_seed(agent_seed)
    np.random.seed(agent_seed)

    envs   = [VVCEnvOpenDSS(seed=s) for s in client_seeds]
    agents = [make_agent(DEVICES[i]) for i in range(3)]

    # Warmup
    log(f"    warmup {WARMUP_STEPS} steps each...")
    for env, agent in zip(envs, agents):
        rollout(env, agent, WARMUP_STEPS, warmup=True)

    # CAE pretrain
    for env, agent in zip(envs, agents):
        agent.pretrain_cae(env, n_collect=1000, n_train_steps=50)

    server_w   = None
    round_hist = {n: [] for n in CLIENT_NAMES}

    for rnd in range(1, N_ROUNDS + 1):
        rewards = [rollout(env, agent, LOCAL_STEPS) for env, agent in zip(envs, agents)]

        for n, r in zip(CLIENT_NAMES, rewards):
            round_hist[n].append(r)

        if condition in ("naive_fl", "aligned_fl"):
            server_w = do_fedavg(agents, server_w)

        if rnd % LOG_INTERVAL == 0:
            rstr = "  ".join(f"{n.split('_')[1]}:{r:+.1f}" for n, r in zip(CLIENT_NAMES, rewards))
            log(f"    round {rnd:>3}/{N_ROUNDS}  |  {rstr}")

    # Final = mean of last 10 rounds
    results = {}
    for n in CLIENT_NAMES:
        vals = [x for x in round_hist[n][-10:] if not np.isnan(x)]
        results[n] = float(np.mean(vals)) if vals else float('nan')

    for env in envs: env.close()
    return results, round_hist


def run_seed(seed_idx):
    log(f"\n{'='*60}")
    log(f"  SEED {seed_idx}")
    log(f"{'='*60}")
    client_seeds = [seed_idx + off for off in CLIENT_SEED_OFFSETS]

    seed_data = {}
    for condition in ["local_only", "naive_fl", "aligned_fl"]:
        # Skip if already saved (resume support)
        out_path = os.path.join(ARTIFACT_DIR, f"seed{seed_idx}_{condition}.json")
        if os.path.exists(out_path):
            log(f"  {condition}: already done — skipping")
            with open(out_path) as f:
                seed_data[condition] = json.load(f)["results"]
            continue

        t0 = time.time()
        results, curves = run_condition(condition, client_seeds, agent_seed=seed_idx*10)
        elapsed = (time.time() - t0) / 60

        log(f"\n  {condition} done ({elapsed:.0f} min):")
        for n, r in results.items(): log(f"    {n}: {r:+.3f}")

        payload = {
            "condition": condition, "seed": seed_idx,
            "client_seeds": client_seeds,
            "env": "VVCEnvOpenDSS (real 3-phase AC, IEEE 13-bus)",
            "n_rounds": N_ROUNDS, "local_steps": LOCAL_STEPS,
            "reward_scale": REWARD_SCALE,
            "results": results,
            "curves": {n: [float(x) for x in v] for n, v in curves.items()},
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        log(f"  saved → {out_path}")
        seed_data[condition] = results

    return seed_data


def summarise():
    from scipy import stats as sc
    conditions = ["local_only", "naive_fl", "aligned_fl"]
    all_data   = {c: {n: [] for n in CLIENT_NAMES} for c in conditions}

    for si in SEEDS:
        for cond in conditions:
            p = os.path.join(ARTIFACT_DIR, f"seed{si}_{cond}.json")
            if not os.path.exists(p): continue
            with open(p) as f: d = json.load(f)
            for n in CLIENT_NAMES:
                if n in d["results"]: all_data[cond][n].append(d["results"][n])

    log("\n" + "="*70)
    log("  FINAL — QE-SAC-FL on REAL OpenDSS  (n={})".format(
        len(all_data["local_only"][CLIENT_NAMES[0]])))
    log("="*70)
    log(f"  {'Condition':<22}  {'A':<20}  {'B':<20}  {'C'}")
    log("  " + "-"*70)
    for cond in conditions:
        row = f"  {cond:<22}"
        for n in CLIENT_NAMES:
            v = all_data[cond][n]
            row += f"  {np.mean(v):+.2f}±{np.std(v):.2f}         " if v else "  —                   "
        log(row)

    log("\n  Effect sizes (aligned vs naive):")
    for n in CLIENT_NAMES:
        a = np.array(all_data["aligned_fl"][n])
        nv = np.array(all_data["naive_fl"][n])
        if len(a) >= 2 and len(nv) >= 2:
            d  = (np.mean(a) - np.mean(nv)) / (np.std(a - nv[:len(a)]) + 1e-9)
            _, p = sc.ttest_rel(a, nv[:len(a)], alternative="greater")
            log(f"    {n.split('_')[1]}: Δ={np.mean(a)-np.mean(nv):+.3f}  d={d:+.2f}  p={p:.4f}")

    summary = {c: {n: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v), "seeds": v}
                   for n, v in vd.items() if v}
               for c, vd in all_data.items()}
    out = os.path.join(VERIF_DIR, "summary.json")
    with open(out, "w") as f: json.dump(summary, f, indent=2)
    log(f"\n  summary → {out}")
    log("="*70)


def main():
    log("="*60)
    log("  QE-SAC-FL — REAL OpenDSS Validation")
    log(f"  env:     VVCEnvOpenDSS (real 3-phase AC)")
    log(f"  rounds:  {N_ROUNDS} × {LOCAL_STEPS} = {N_ROUNDS*LOCAL_STEPS:,} steps/client")
    log(f"  seeds:   {SEEDS}")
    log(f"  GPUs:    {N_GPUS}  {DEVICES}")
    log("="*60)

    t0 = time.time()
    for si in SEEDS:
        run_seed(si)
    summarise()
    log(f"\n  total: {(time.time()-t0)/3600:.1f} h  — DONE")


if __name__ == "__main__":
    main()
