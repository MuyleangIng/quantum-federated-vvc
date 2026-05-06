"""Re-run seed0 local_only condition for OpenDSS FL validation."""
import sys, os, json, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT); os.chdir(ROOT)

import numpy as np
import torch
from src.qe_sac.env_opendss import VVCEnvOpenDSS
from src.qe_sac_fl.aligned_agent import AlignedQESACAgent
from src.qe_sac_fl.aligned_encoder import fedavg_shared_head

N_ROUNDS=50; LOCAL_STEPS=1_000; WARMUP_STEPS=500; BATCH_SIZE=256
LR=1e-4; GAMMA=0.99; TAU=0.005; ALPHA=0.2; BUFFER_SIZE=100_000
HIDDEN_DIM=32; REWARD_SCALE=1.0; SEED=0
CLIENT_SEED_OFFSETS=[0,7,14]
CLIENT_NAMES=["Utility_A_13bus","Utility_B_13bus_v2","Utility_C_13bus_v3"]
OBS_DIM=48; NVEC=[2,2,33,33,33,33]
N_GPUS=torch.cuda.device_count()
DEVICES=[f"cuda:{i}" if i<N_GPUS else "cpu" for i in range(3)]
OUT=os.path.join(ROOT,"artifacts","qe_sac_fl_opendss",f"seed{SEED}_local_only.json")

def log(m): print(m, flush=True)

def make_agent(device):
    return AlignedQESACAgent(obs_dim=OBS_DIM, device_dims=NVEC,
        lr=LR, gamma=GAMMA, tau=TAU, alpha=ALPHA,
        buffer_size=BUFFER_SIZE, hidden_dim=HIDDEN_DIM, device=device)

def rollout(env, agent, n_steps, warmup=False):
    obs, _ = env.reset(); ep_rewards, ep_r = [], 0.0
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
        ep_r += scaled_r; obs = next_obs
        if te or tr:
            ep_rewards.append(ep_r); ep_r = 0.0; obs, _ = env.reset()
    return float(np.mean(ep_rewards)) if ep_rewards else float('nan')

t0 = time.time()
torch.manual_seed(SEED*10); np.random.seed(SEED*10)
client_seeds = [SEED + off for off in CLIENT_SEED_OFFSETS]
log(f"seed0 local_only  env_seeds={client_seeds}")
envs   = [VVCEnvOpenDSS(seed=s) for s in client_seeds]
agents = [make_agent(DEVICES[i]) for i in range(3)]
log(f"  warmup {WARMUP_STEPS} steps...")
for env, agent in zip(envs, agents):
    rollout(env, agent, WARMUP_STEPS, warmup=True)
for env, agent in zip(envs, agents):
    agent.pretrain_cae(env, n_collect=1000, n_train_steps=50)
round_hist = {n: [] for n in CLIENT_NAMES}
for rnd in range(1, N_ROUNDS+1):
    rewards = [rollout(env, agent, LOCAL_STEPS) for env, agent in zip(envs, agents)]
    for n, r in zip(CLIENT_NAMES, rewards): round_hist[n].append(r)
    if rnd % 10 == 0:
        rstr = "  ".join(f"{n.split('_')[1]}:{r:+.1f}" for n,r in zip(CLIENT_NAMES,rewards))
        log(f"  round {rnd:>2}/{N_ROUNDS}  |  {rstr}")
for env in envs: env.close()
results = {}
for n in CLIENT_NAMES:
    vals = [x for x in round_hist[n][-10:] if not np.isnan(x)]
    results[n] = float(np.mean(vals)) if vals else float('nan')
payload = {"condition":"local_only","seed":SEED,"client_seeds":client_seeds,
    "env":"VVCEnvOpenDSS (real 3-phase AC, IEEE 13-bus)",
    "n_rounds":N_ROUNDS,"local_steps":LOCAL_STEPS,"reward_scale":REWARD_SCALE,
    "results":results,"curves":{n:[float(x) for x in v] for n,v in round_hist.items()}}
with open(OUT,"w") as f: json.dump(payload, f, indent=2)
log(f"saved → {OUT}")
for n,r in results.items(): log(f"  {n}: {r:+.3f}")
log(f"done in {(time.time()-t0)/60:.0f} min")
