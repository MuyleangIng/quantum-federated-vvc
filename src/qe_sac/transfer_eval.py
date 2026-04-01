"""
Transfer Learning Evaluation for QE-SAC (Task 5).

Tests whether a policy trained on a small feeder (13-bus) can control
a larger unseen feeder (123-bus) — with or without GNN adaptation.

Three transfer conditions:
    A. Zero-shot    : freeze both VQC and GNN, evaluate directly
    B. GNN-adapt    : freeze VQC, re-train GNN for N steps, then evaluate
    C. Full retrain : train from scratch on target feeder (baseline)

Key insight:
    VQC = feeder-agnostic quantum policy (16 params learned from 13-bus)
    GNN = feeder-specific encoder (re-adapts in ~500 steps)

Usage
-----
from src.qe_sac.transfer_eval import transfer_evaluate, TransferResults

source_ckpt = "artifacts/qe_sac/opendss_qe-sac_opendss_seed0.pt"
results = transfer_evaluate(
    source_ckpt_path = source_ckpt,
    source_obs_dim   = 42,       # 13-bus DistFlow
    target_env       = VVCEnv123Bus(),
    freeze_vqc       = True,
    n_adapt_steps    = 500,
    n_eval_episodes  = 10,
)
print(results)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class TransferResults:
    """Results from one transfer evaluation run."""
    condition:        str        # "zero_shot" | "gnn_adapt" | "full_retrain"
    source_feeder:    str        # e.g. "13-bus"
    target_feeder:    str        # e.g. "123-bus"
    n_adapt_steps:    int        # 0 for zero-shot
    mean_reward:      float
    std_reward:       float
    mean_vviol:       float
    reward_vs_scratch: float     # fractional reward vs training from scratch
    n_eval_episodes:  int

    def to_dict(self) -> dict:
        return asdict(self)


def _freeze_module(module: nn.Module) -> None:
    """Set all parameters to requires_grad=False."""
    for p in module.parameters():
        p.requires_grad_(False)


def _unfreeze_module(module: nn.Module) -> None:
    """Set all parameters to requires_grad=True."""
    for p in module.parameters():
        p.requires_grad_(True)


def evaluate_agent_episodes(
    agent,
    env,
    n_episodes: int = 10,
    device: str = "cpu",
) -> tuple[float, float, float]:
    """
    Evaluate agent for n_episodes.

    Returns (mean_reward, std_reward, mean_vviol).
    """
    rewards, vviols = [], []
    nvec = env.action_space.nvec

    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = ep_vviol = 0.0
        done = False
        while not done:
            with torch.no_grad():
                t_obs = torch.tensor(obs, dtype=torch.float32, device=device)
                scalar = int(agent.select_action(t_obs, deterministic=True))

            env_action = np.zeros(len(nvec), dtype=np.int64)
            remainder  = scalar % int(nvec.prod())
            for i in range(len(nvec) - 1, -1, -1):
                env_action[i] = remainder % nvec[i]
                remainder //= nvec[i]

            obs, rew, term, trunc, info = env.step(env_action)
            ep_reward += rew
            ep_vviol  += info.get("v_viol", 0)
            done = term or trunc

        rewards.append(ep_reward)
        vviols.append(ep_vviol)

    return float(np.mean(rewards)), float(np.std(rewards)), float(np.mean(vviols))


def adapt_gnn_encoder(
    agent,
    target_env,
    n_adapt_steps: int = 500,
    batch_size:    int = 32,
    lr:            float = 1e-3,
    device:        str = "cpu",
) -> None:
    """
    Re-adapt the GNN encoder (or CAE) to target feeder by running
    random exploration and updating the encoder on collected observations.

    VQC weights are frozen during adaptation — only encoder updates.
    """
    # Freeze VQC (and head) — only encoder trains
    if hasattr(agent.actor, 'gnn'):
        encoder = agent.actor.gnn
    elif hasattr(agent.actor, 'cae'):
        encoder = agent.actor.cae
    else:
        raise AttributeError("Agent actor must have .gnn or .cae encoder.")

    _freeze_module(agent.actor.vqc)
    _freeze_module(agent.actor.head)
    _unfreeze_module(encoder)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)

    # Collect random observations from target feeder
    obs_buffer = []
    obs, _ = target_env.reset()
    for step in range(n_adapt_steps * batch_size):
        action = target_env.action_space.sample()
        next_obs, _, term, trunc, _ = target_env.step(action)
        obs_buffer.append(obs.copy())
        obs = next_obs
        if term or trunc:
            obs, _ = target_env.reset()

    obs_data = np.array(obs_buffer, dtype=np.float32)

    # Train encoder on collected observations
    if hasattr(encoder, 'encode'):   # GNN or CAE
        from src.qe_sac.gnn_encoder import train_gnn_encoder
        from src.qe_sac.autoencoder import train_cae
        if hasattr(encoder, 'conv1'):   # GNN
            train_gnn_encoder(encoder, obs_data, n_steps=n_adapt_steps, lr=lr, device=device)
        else:                           # CAE
            train_cae(encoder, obs_data, n_steps=n_adapt_steps, lr=lr, device=device)

    # Unfreeze everything after adaptation
    _unfreeze_module(agent.actor.vqc)
    _unfreeze_module(agent.actor.head)


def transfer_evaluate(
    agent,
    source_feeder:   str,
    target_env,
    target_feeder:   str,
    freeze_vqc:      bool = True,
    n_adapt_steps:   int  = 0,
    n_eval_episodes: int  = 10,
    device:          str  = "cpu",
) -> TransferResults:
    """
    Evaluate transfer from source to target feeder.

    Parameters
    ----------
    agent          : trained QESACAgent or GNNQESACAgent from source feeder
    source_feeder  : label e.g. "13-bus"
    target_env     : gymnasium env for the target feeder
    target_feeder  : label e.g. "123-bus"
    freeze_vqc     : whether to freeze VQC during adaptation
    n_adapt_steps  : 0 = zero-shot, >0 = GNN adaptation steps
    n_eval_episodes: evaluation episodes

    Returns
    -------
    TransferResults
    """
    condition = "zero_shot" if n_adapt_steps == 0 else "gnn_adapt"

    if n_adapt_steps > 0 and freeze_vqc:
        adapt_gnn_encoder(agent, target_env, n_adapt_steps=n_adapt_steps, device=device)

    mean_r, std_r, mean_v = evaluate_agent_episodes(
        agent, target_env, n_episodes=n_eval_episodes, device=device
    )

    return TransferResults(
        condition=condition,
        source_feeder=source_feeder,
        target_feeder=target_feeder,
        n_adapt_steps=n_adapt_steps,
        mean_reward=mean_r,
        std_reward=std_r,
        mean_vviol=mean_v,
        reward_vs_scratch=float("nan"),   # filled in by caller after scratch baseline
        n_eval_episodes=n_eval_episodes,
    )
