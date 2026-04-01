"""
Task 4.6 — VQC qubit/layer ablation study.

Sweeps n_qubits ∈ {4, 8, 12, 16} (n_layers=2 fixed)
     and n_layers ∈ {1, 2, 3, 4}  (n_qubits=8 fixed)

For each config:
    - Train QE-SAC with VQCLayerAblation (custom n_qubits/n_layers)
    - CAE encoder output dim must match n_qubits → use custom CAE
    - 3 seeds × 50K steps on VVCEnv13Bus
    - Record: final reward, vviol, VQC gradient norm (barren plateau check)

Results saved to:
    artifacts/qe_sac/vqc_ablation_qubits.json
    artifacts/qe_sac/vqc_ablation_layers.json

NOTE: This script is SLOW (PennyLane sequential per config).
      Run overnight or use n_steps=10_000 for a quick check.
"""

import sys
import json
import os

sys.path.insert(0, "/root/power-system")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.qe_sac.env_utils import VVCEnv13Bus
from src.qe_sac.autoencoder import CAE, train_cae
from src.qe_sac.vqc import VQCLayerAblation
from src.qe_sac.sac_baseline import ClassicalSACAgent
from src.qe_sac.metrics import count_parameters

SEEDS      = [0, 1, 2]
N_STEPS    = 50_000
BATCH_SIZE = 256
WARMUP     = 1_000
CAE_INTERVAL = 500
SAVE_DIR   = "artifacts/qe_sac"
os.makedirs(SAVE_DIR, exist_ok=True)


class AblationActorNetwork(nn.Module):
    """QE-SAC actor with configurable n_qubits/n_layers."""

    def __init__(self, obs_dim: int, n_actions: int, n_qubits: int, n_layers: int):
        super().__init__()
        self.cae = CAE(obs_dim, latent_dim=n_qubits)   # CAE latent matches VQC input
        self.vqc = VQCLayerAblation(n_qubits=n_qubits, n_layers=n_layers)
        self.head = nn.Linear(n_qubits, n_actions)
        self._n_qubits = n_qubits

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        latent  = self.cae.encode(obs)
        vqc_out = self.vqc(latent)
        return F.softmax(self.head(vqc_out), dim=-1)

    def vqc_gradient_norm(self) -> float:
        """Returns L2 norm of VQC weight gradients (barren plateau indicator)."""
        if self.vqc.weights.grad is None:
            return 0.0
        return float(self.vqc.weights.grad.norm(p=2).item())


class AblationAgent:
    """Minimal SAC agent for ablation (wraps AblationActorNetwork)."""

    def __init__(self, obs_dim, n_actions, n_qubits, n_layers, buffer_size=200_000):
        from src.qe_sac.qe_sac_policy import _MLPCritic

        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self._grad_steps = 0
        self._n_qubits = n_qubits

        self.actor   = AblationActorNetwork(obs_dim, n_actions, n_qubits, n_layers)
        self.critic1 = _MLPCritic(obs_dim, n_actions)
        self.critic2 = _MLPCritic(obs_dim, n_actions)
        self.target1 = _MLPCritic(obs_dim, n_actions)
        self.target2 = _MLPCritic(obs_dim, n_actions)
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())

        self.actor_opt   = optim.Adam(self.actor.parameters(),   lr=1e-4)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=1e-4)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=1e-4)

        self._buf_obs  = np.zeros((buffer_size, obs_dim),   dtype=np.float32)
        self._buf_act  = np.zeros((buffer_size, n_actions), dtype=np.float32)
        self._buf_rew  = np.zeros(buffer_size,               dtype=np.float32)
        self._buf_next = np.zeros((buffer_size, obs_dim),   dtype=np.float32)
        self._buf_done = np.zeros(buffer_size,               dtype=np.float32)
        self._ptr  = 0
        self._size = 0
        self._max  = buffer_size

        self._vqc_grad_norms = []   # track gradient norms

    def select_action(self, obs, deterministic=False):
        probs = self.actor(obs)
        if deterministic:
            return probs.argmax(dim=-1).cpu().numpy()
        return torch.multinomial(probs, 1).squeeze(-1).cpu().numpy()

    def store(self, obs, action, reward, next_obs, done, v_viol=0.0):
        oh = np.zeros(self.n_actions, dtype=np.float32)
        oh[int(action)] = 1.0
        self._buf_obs[self._ptr]  = obs
        self._buf_act[self._ptr]  = oh
        self._buf_rew[self._ptr]  = reward
        self._buf_next[self._ptr] = next_obs
        self._buf_done[self._ptr] = float(done)
        self._ptr  = (self._ptr + 1) % self._max
        self._size = min(self._size + 1, self._max)

    def update(self, batch_size=256, cae_update_interval=500, cae_steps=50):
        if self._size < batch_size:
            return {}
        gamma, tau, alpha = 0.99, 0.005, 0.2
        idx = np.random.randint(0, self._size, batch_size)
        obs  = torch.tensor(self._buf_obs[idx])
        act  = torch.tensor(self._buf_act[idx])
        rew  = torch.tensor(self._buf_rew[idx]).unsqueeze(1)
        nobs = torch.tensor(self._buf_next[idx])
        done = torch.tensor(self._buf_done[idx]).unsqueeze(1)

        with torch.no_grad():
            np_ = self.actor(nobs)
            q1n = self.target1(nobs, np_)
            q2n = self.target2(nobs, np_)
            ent = -(np_ * torch.log(np_ + 1e-8)).sum(-1, keepdim=True)
            tq  = rew + (1-done) * gamma * (torch.min(q1n,q2n) + alpha*ent)

        c1l = F.mse_loss(self.critic1(obs, act), tq)
        c2l = F.mse_loss(self.critic2(obs, act), tq)
        self.critic1_opt.zero_grad(); c1l.backward(); self.critic1_opt.step()
        self.critic2_opt.zero_grad(); c2l.backward(); self.critic2_opt.step()

        pr = self.actor(obs)
        qp = torch.min(self.critic1(obs,pr), self.critic2(obs,pr))
        en = -(pr * torch.log(pr + 1e-8)).sum(-1, keepdim=True)
        al = -(qp + alpha*en).mean()
        self.actor_opt.zero_grad(); al.backward(); self.actor_opt.step()

        # Record VQC gradient norm
        self._vqc_grad_norms.append(self.actor.vqc_gradient_norm())

        for p,tp in zip(self.critic1.parameters(), self.target1.parameters()):
            tp.data.copy_(tau*p.data + (1-tau)*tp.data)
        for p,tp in zip(self.critic2.parameters(), self.target2.parameters()):
            tp.data.copy_(tau*p.data + (1-tau)*tp.data)

        self._grad_steps += 1
        if self._grad_steps % cae_update_interval == 0:
            train_cae(self.actor.cae, self._buf_obs[:self._size],
                      n_steps=cae_steps, latent_dim=self._n_qubits)

        return {"actor_loss": float(al), "critic_loss": float((c1l+c2l)/2)}

    def param_count(self):
        return count_parameters(self.actor)

    def save(self, path):
        torch.save({"actor": self.actor.state_dict()}, path)


def run_config(n_qubits, n_layers, seeds, n_steps, tag):
    """Run ablation for one (n_qubits, n_layers) config across seeds."""
    print(f"\n{'='*60}")
    print(f"  n_qubits={n_qubits}  n_layers={n_layers}  tag={tag}")
    print(f"{'='*60}")

    all_rewards, all_vviols, all_grad_norms = [], [], []
    n_actions = 2 * 2 * 33  # fixed action space

    for seed in seeds:
        env = VVCEnv13Bus(seed=seed)
        obs_dim = env.observation_space.shape[0]
        torch.manual_seed(seed)
        np.random.seed(seed)

        agent = AblationAgent(obs_dim, n_actions, n_qubits, n_layers)
        nvec = env.action_space.nvec

        ep_reward = ep_vviol = 0.0
        ep_rewards, ep_vviols_list = [], []
        obs, _ = env.reset()
        total_steps = 0

        while total_steps < n_steps:
            if total_steps < WARMUP:
                env_action = env.action_space.sample()
                scalar = 0
                for a, n in zip(env_action, nvec):
                    scalar = scalar * int(n) + int(a)
            else:
                with torch.no_grad():
                    scalar = int(agent.select_action(torch.tensor(obs)))
                env_action = np.zeros(len(nvec), dtype=np.int64)
                remainder = scalar % int(nvec.prod())
                for i in range(len(nvec)-1, -1, -1):
                    env_action[i] = remainder % nvec[i]
                    remainder //= nvec[i]

            nobs, rew, term, trunc, info = env.step(env_action)
            done = term or trunc
            agent.store(obs, scalar, rew, nobs, done)
            ep_reward += rew
            ep_vviol  += info.get("v_viol", 0)
            obs = nobs
            total_steps += 1

            if total_steps >= WARMUP:
                agent.update(BATCH_SIZE, CAE_INTERVAL)

            if done:
                ep_rewards.append(ep_reward)
                ep_vviols_list.append(ep_vviol)
                obs, _ = env.reset()
                ep_reward = ep_vviol = 0.0

        mean_grad = float(np.mean(agent._vqc_grad_norms)) if agent._vqc_grad_norms else 0.0
        all_rewards.append(float(np.mean(ep_rewards)))
        all_vviols.append(float(np.mean(ep_vviols_list)))
        all_grad_norms.append(mean_grad)
        print(f"  seed={seed}: reward={all_rewards[-1]:.2f}  vviol={all_vviols[-1]:.2f}  grad_norm={mean_grad:.6f}")

    return {
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "n_params": n_qubits * n_layers,   # VQC params only
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward":  float(np.std(all_rewards)),
        "mean_vviol":  float(np.mean(all_vviols)),
        "mean_vqc_grad_norm": float(np.mean(all_grad_norms)),
        "barren_plateau": bool(np.mean(all_grad_norms) < 1e-5),
        "seeds": all_rewards,
    }


def main():
    # Qubit sweep (n_layers=2 fixed)
    qubit_results = {}
    for nq in [4, 8, 12, 16]:
        tag = f"q{nq}_l2"
        r = run_config(nq, n_layers=2, seeds=SEEDS, n_steps=N_STEPS, tag=tag)
        qubit_results[tag] = r

    out = os.path.join(SAVE_DIR, "vqc_ablation_qubits.json")
    with open(out, "w") as f:
        json.dump(qubit_results, f, indent=2)
    print(f"\nQubit ablation saved → {out}")

    # Layer sweep (n_qubits=8 fixed)
    layer_results = {}
    for nl in [1, 2, 3, 4]:
        tag = f"q8_l{nl}"
        if tag in qubit_results:   # q8_l2 already done
            layer_results[tag] = qubit_results[tag]
            continue
        r = run_config(n_qubits=8, n_layers=nl, seeds=SEEDS, n_steps=N_STEPS, tag=tag)
        layer_results[tag] = r

    out = os.path.join(SAVE_DIR, "vqc_ablation_layers.json")
    with open(out, "w") as f:
        json.dump(layer_results, f, indent=2)
    print(f"Layer ablation saved → {out}")

    # Print summary
    print("\n=== QUBIT SWEEP (n_layers=2) ===")
    print(f"  {'Config':10s}  {'Params':8s}  {'Reward':10s}  {'Std':8s}  {'GradNorm':12s}  Barren?")
    for tag, r in qubit_results.items():
        print(f"  {tag:10s}  {r['n_params']:8d}  {r['mean_reward']:10.2f}  "
              f"{r['std_reward']:8.3f}  {r['mean_vqc_grad_norm']:12.6f}  "
              f"{'YES' if r['barren_plateau'] else 'no'}")

    print("\n=== LAYER SWEEP (n_qubits=8) ===")
    for tag, r in layer_results.items():
        print(f"  {tag:10s}  {r['n_params']:8d}  {r['mean_reward']:10.2f}  "
              f"{r['std_reward']:8.3f}  {r['mean_vqc_grad_norm']:12.6f}  "
              f"{'YES' if r['barren_plateau'] else 'no'}")


if __name__ == "__main__":
    main()
