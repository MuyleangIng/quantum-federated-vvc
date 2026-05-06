import os, sys, json, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if '__file__' in dir() else '/root/power-system'
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import torch

VW = float(os.environ.get("VW", "50"))

# Patch voltage weight
import src.qe_sac.env_opendss as env_mod
env_mod._V_W = VW

from src.qe_sac.env_opendss import VVCEnvOpenDSS
from src.qe_sac_fl.aligned_agent import AlignedQESACAgent

NVEC = [2, 2, 33, 33, 33, 33]
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

torch.manual_seed(0); np.random.seed(0)
env = VVCEnvOpenDSS(seed=0)
obs_dim = env.observation_space.shape[0]
agent = AlignedQESACAgent(obs_dim=obs_dim, device_dims=list(NVEC),
    lr=3e-4, gamma=0.99, tau=0.005, alpha=0.1,
    buffer_size=100_000, hidden_dim=32, device=DEVICE)

obs, _ = env.reset()
for i in range(30_000):
    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
    action = env.action_space.sample() if agent._size < 512 else agent.actor.select_action(obs_t)
    nobs, r, te, tr, _ = env.step(action)
    agent.store(obs, action, r, nobs, te or tr)
    if agent._size >= 512: agent.update(256)
    obs = nobs if not (te or tr) else env.reset()[0]
env.close()

# Eval
eval_env = VVCEnvOpenDSS(seed=100)
rewards, ctrls = [], []
for _ in range(10):
    obs, _ = eval_env.reset()
    ep_r, ctrl, total, done = 0, 0, 0, False
    while not done:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        action = agent.actor.select_action(obs_t, deterministic=True)
        obs, r, te, tr, info = eval_env.step(action)
        ep_r += r; total += 1
        v = info.get("voltage", np.array([]))
        if len(v) > 0 and np.all((v>=0.95)&(v<=1.05)): ctrl += 1
        done = te or tr
    rewards.append(ep_r)
    ctrls.append(ctrl/max(total,1)*100)
eval_env.close()

print(f"VW={VW}  reward={np.mean(rewards):.3f}  ctrl={np.mean(ctrls):.1f}%")
