"""
QFL Trainer — Federated Learning loop for QFL agents.

Each round:
  1. All clients train locally (SAC on private data)
  2. Clients upload QuantumEncoder weights (280 params = 1.1 KB)
  3. Server FedAvg → global shared_head + VQC weights
  4. Broadcast back to all clients
"""

from __future__ import annotations

import json
import os
import time
import numpy as np
import torch

from src.qfl.agent import QFLAgent
from src.qfl.config import QFLConfig, ClientConfig
from src.qe_sac_fl.federated_trainer import _make_env


# ---------------------------------------------------------------------------
# FedAvg
# ---------------------------------------------------------------------------

def _fedavg(weight_list: list[dict]) -> dict:
    """Uniform FedAvg of QuantumEncoder shared_head + VQC weights across clients."""
    # All clients have identical shared_head shapes (64→32→8)
    head_keys = weight_list[0]["shared_head"].keys()
    avg_head = {
        k: torch.stack([w["shared_head"][k].float() for w in weight_list]).mean(0)
        for k in head_keys
    }
    # VQC weights shape [2, 8] — same for all clients
    avg_vqc = torch.stack([w["vqc"].float() for w in weight_list]).mean(0)
    return {"shared_head": avg_head, "vqc": avg_vqc}


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class QFLTrainer:
    """
    Runs QFL federation across multiple utility clients.

    Conditions:
      local_only — each client trains independently (no FL)
      qfl        — full QFL with FedAvg on QuantumEncoder
    """

    def __init__(self, cfg: QFLConfig):
        self.cfg = cfg

    def _build_agents_and_envs(self, seed_offset: int):
        agents, envs = [], []
        for cc in self.cfg.clients:
            env = _make_env(cc.env_id, cc.seed + seed_offset, cc.reward_scale)
            act_dims = list(env.action_space.nvec)
            agent = QFLAgent(
                obs_dim     = cc.obs_dim,
                device_dims = act_dims,
                lr          = self.cfg.lr,
                device      = cc.device,
            )
            agents.append(agent)
            envs.append(env)
        return agents, envs

    def run(self, seed: int = 0) -> dict:
        cfg = self.cfg
        results = {}

        for condition in cfg.conditions:
            print(f"\n{'='*60}", flush=True)
            print(f"  QFL — {condition.upper()} — seed {seed}", flush=True)
            print(f"  {cfg.n_rounds} rounds × {cfg.local_steps} steps", flush=True)
            print(f"{'='*60}", flush=True)

            agents, envs = self._build_agents_and_envs(seed)
            logs = []
            t0   = time.time()

            for rnd in range(cfg.n_rounds):
                round_rewards = []

                for agent, env, cc in zip(agents, envs, cfg.clients):
                    reward = agent.train_round(env, steps=cfg.local_steps)
                    vqc_grad = 0.0
                    round_rewards.append(reward)
                    logs.append({
                        "client":        cc.name,
                        "round":         rnd,
                        "reward":        reward,
                        "vqc_grad_norm": vqc_grad,
                        "steps":         cfg.local_steps,
                    })

                # FedAvg (skip if local_only)
                if condition == "qfl":
                    global_w = _fedavg([a.get_federated_weights() for a in agents])
                    for agent in agents:
                        agent.set_federated_weights(global_w)

                if (rnd + 1) % cfg.log_interval == 0 or rnd == cfg.n_rounds - 1:
                    parts = "  ".join(
                        f"{cc.name.split('_')[1]}:{r:.6f}"
                        for cc, r in zip(cfg.clients, round_rewards)
                    )
                    print(f"  round {rnd+1:3d}/{cfg.n_rounds}  |  {parts}", flush=True)

            wall = time.time() - t0
            bytes_comm = 0
            if condition == "qfl":
                bytes_comm = len(cfg.clients) * cfg.n_rounds * 280 * 4 * 2

            results[condition] = {
                "condition":          condition,
                "seed":               seed,
                "n_rounds":           cfg.n_rounds,
                "wall_time_seconds":  wall,
                "bytes_communicated": bytes_comm,
                "logs":               logs,
            }

            path = os.path.join(cfg.save_dir, f"seed{seed}_{condition}.json")
            with open(path, "w") as f:
                json.dump(results[condition], f, indent=2)
            print(f"  saved → {path}", flush=True)

        return results
