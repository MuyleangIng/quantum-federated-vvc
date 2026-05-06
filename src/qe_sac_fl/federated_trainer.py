"""
Federated trainer for QE-SAC-FL.

Implements FedAvg over VQC weights only.
Each client trains locally; only VQC weights are sent to the server.
All other weights (CAE/GNN encoder, critics, λ) stay local and private.

Usage:
    from src.qe_sac_fl.federated_trainer import FederatedTrainer, FedResults
    from src.qe_sac_fl.fed_config import FedConfig

    cfg     = FedConfig()
    trainer = FederatedTrainer(cfg, device='cuda')
    results = trainer.run()
    results.summary()
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.qe_sac_fl.fed_config import FedConfig, ClientConfig
from src.qe_sac_fl.env_34bus import VVCEnv34Bus, VVCEnv34BusFL, VVCEnv123BusFL
from src.qe_sac_fl.aligned_encoder import fedavg_shared_head, bytes_per_aligned_update
from src.qe_sac_fl.aligned_agent import AlignedQESACAgent
from src.qe_sac.env_utils import VVCEnv13Bus, VVCEnv123Bus
from src.qe_sac.qe_sac_policy import QESACAgent
from src.qe_sac.trainer import QESACTrainer


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class ClientRoundLog:
    """Metrics logged by one client after one local training round."""
    client_name: str
    round_idx:   int
    mean_reward: float
    total_vviol: int
    steps_done:  int
    vqc_grad_norm: float  # check for barren plateau


@dataclass
class FedResults:
    """Accumulated results across all rounds and clients."""
    config: FedConfig
    condition: str                              # "QE-SAC-FL", "local_only", etc.
    logs: List[ClientRoundLog] = field(default_factory=list)
    bytes_communicated: int = 0                 # for H3
    wall_time_seconds: float = 0.0

    # --- H1: final reward per client ---
    def final_rewards(self) -> Dict[str, float]:
        out = {}
        for client_cfg in self.config.clients:
            client_logs = [l for l in self.logs if l.client_name == client_cfg.name]
            if client_logs:
                out[client_cfg.name] = client_logs[-1].mean_reward
        return out

    # --- H2: steps to convergence per client ---
    def steps_to_convergence(self, threshold: float) -> Dict[str, Optional[int]]:
        out = {}
        for client_cfg in self.config.clients:
            client_logs = [l for l in self.logs if l.client_name == client_cfg.name]
            converged_at = None
            cumulative = 0
            for log in client_logs:
                cumulative += log.steps_done
                if log.mean_reward >= threshold and converged_at is None:
                    converged_at = cumulative
            out[client_cfg.name] = converged_at
        return out

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"  Condition : {self.condition}",
            f"  Rounds    : {self.config.n_rounds}",
            f"  Bytes TX  : {self.bytes_communicated:,}  (H3 metric)",
            f"  Wall time : {self.wall_time_seconds:.1f}s",
            f"{'='*60}",
            f"  Final rewards per client (H1):",
        ]
        for name, r in self.final_rewards().items():
            lines.append(f"    {name:<30} {r:+.3f}")
        thresh = self.config.reward_convergence_threshold
        lines.append(f"\n  Steps to convergence (reward > {thresh}) (H2):")
        for name, s in self.steps_to_convergence(thresh).items():
            lines.append(f"    {name:<30} {s if s else 'not reached'}")
        lines.append("="*60)
        return "\n".join(lines)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "condition": self.condition,
            "n_rounds":  self.config.n_rounds,
            "bytes_communicated": self.bytes_communicated,
            "wall_time_seconds":  self.wall_time_seconds,
            "logs": [
                {
                    "client": l.client_name,
                    "round":  l.round_idx,
                    "reward": l.mean_reward,
                    "vviol":  l.total_vviol,
                    "steps":  l.steps_done,
                    "vqc_grad_norm": l.vqc_grad_norm,
                }
                for l in self.logs
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Results saved → {path}")


# ---------------------------------------------------------------------------
# Reward scaling wrapper
# ---------------------------------------------------------------------------

class RewardScaledEnv:
    """
    Thin env wrapper that divides rewards by a fixed scale before returning them.

    Keeps logged episode returns (which QESACTrainer accumulates from raw rewards)
    in a comparable range across heterogeneous feeders:
      A_13bus  raw ~-333 / 50  → ~-6.7
      B_34bus  raw  ~-74 / 10  → ~-7.4
      C_123bus raw ~-5359 / 750 → ~-7.1

    This normalises critic targets and VQC gradient magnitudes so FedAvg
    gives equal effective influence to every client regardless of feeder size.
    The scale is applied consistently to local_only, naive_fl, and aligned_fl
    so cross-condition comparisons remain valid (all in normalised units).
    """

    def __init__(self, env, scale: float):
        self.env = env
        self.scale = float(scale)
        self.observation_space = env.observation_space
        self.action_space      = env.action_space

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward / self.scale, terminated, truncated, info

    def __getattr__(self, name):
        return getattr(self.env, name)


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def _make_env(env_id: str, seed: int, reward_scale: float = 1.0):
    if env_id == "13bus" or env_id == "13bus_fl":
        env = VVCEnv13Bus(seed=seed)
    elif env_id == "34bus":
        env = VVCEnv34Bus(seed=seed)
    elif env_id == "34bus_fl":
        env = VVCEnv34BusFL(seed=seed)
    elif env_id == "123bus":
        env = VVCEnv123Bus(seed=seed)
    elif env_id == "123bus_fl":
        env = VVCEnv123BusFL(seed=seed)
    else:
        raise ValueError(f"Unknown env_id: {env_id!r}")
    if reward_scale != 1.0:
        env = RewardScaledEnv(env, reward_scale)
    return env


# ---------------------------------------------------------------------------
# VQC weight extraction and loading
# ---------------------------------------------------------------------------

def _get_vqc_weights(agent: QESACAgent) -> torch.Tensor:
    """Extract VQC parameters as a flat tensor (16 values)."""
    return agent.actor.vqc.weights.data.clone()


def _set_vqc_weights(agent: QESACAgent, weights: torch.Tensor) -> None:
    """Load VQC parameters back into agent (in-place)."""
    with torch.no_grad():
        agent.actor.vqc.weights.copy_(weights)


def _vqc_grad_norm(agent: QESACAgent) -> float:
    """Return L2 norm of VQC weight gradients (barren plateau diagnostic)."""
    g = agent.actor.vqc.weights.grad
    if g is None:
        return 0.0
    return float(g.norm().item())


def _bytes_per_vqc_update(n_clients: int) -> int:
    """
    Bytes communicated per FL round (upload + download).
    VQC has 16 float32 parameters = 64 bytes per client per direction.
    """
    bytes_per_param = 4        # float32
    n_params        = 16       # VQC always has exactly 16 params
    upload          = n_clients * n_params * bytes_per_param
    download        = n_clients * n_params * bytes_per_param
    return upload + download


# ---------------------------------------------------------------------------
# FedAvg aggregation
# ---------------------------------------------------------------------------

def _fedavg(
    weight_list: List[torch.Tensor],
    aggregation: str = "uniform",
    rewards: Optional[List[float]] = None,
) -> torch.Tensor:
    """
    Average a list of VQC weight tensors.

    aggregation = "uniform"       → standard FedAvg (equal weights).
    aggregation = "magnitude_inv" → weight by 1/|reward|; clients with
                                     smaller |reward| (closer to 0 = better
                                     performance) contribute MORE to the global
                                     model. Prevents large-scale feeders from
                                     dominating the aggregate.
    """
    stacked = torch.stack(weight_list, dim=0)  # (n_clients, n_params)
    if aggregation == "magnitude_inv" and rewards is not None:
        magnitudes = np.array([max(abs(r), 1e-6) for r in rewards], dtype=np.float64)
        w = 1.0 / magnitudes
        w = w / w.sum()
        wt = torch.tensor(w, dtype=torch.float32)
        # reshape to (n_clients, 1, 1, ...) to broadcast over any weight shape
        wt = wt.view(-1, *([1] * (stacked.dim() - 1)))
        return (stacked * wt).sum(dim=0)
    return stacked.mean(dim=0)


# ---------------------------------------------------------------------------
# Main federated trainer
# ---------------------------------------------------------------------------

class FederatedTrainer:
    """
    Runs the QE-SAC-FL federated training loop.

    Each round:
      1. Server broadcasts current global VQC weights to all clients.
      2. Each client loads VQC weights, runs local_steps of QE-SAC training.
      3. Each client sends updated VQC weights back.
      4. Server runs FedAvg → new global VQC weights.
      5. Log metrics.
    """

    def __init__(self, config: FedConfig):
        self.cfg = config

    def _build_client(self, client_cfg: ClientConfig) -> tuple:
        """Build (env, agent, trainer) for one client on its assigned device."""
        env         = _make_env(client_cfg.env_id, client_cfg.seed, client_cfg.reward_scale)
        device_dims = list(map(int, env.action_space.nvec))
        agent = QESACAgent(
            obs_dim     = env.observation_space.shape[0],
            device_dims = device_dims,
            lr          = self.cfg.lr,
            gamma       = self.cfg.gamma,
            tau         = self.cfg.tau,
            alpha       = self.cfg.alpha,
            buffer_size = self.cfg.buffer_size,
            device      = client_cfg.device,
        )
        trainer = QESACTrainer(
            agent,
            env,
            batch_size   = self.cfg.batch_size,
            warmup_steps = self.cfg.warmup_steps,
            log_interval = 9999,
            device       = client_cfg.device,
        )
        return env, agent, trainer

    def _train_one_client(
        self,
        ccfg: ClientConfig,
        agent: QESACAgent,
        trainer: QESACTrainer,
        global_vqc: torch.Tensor,
        round_idx: int,
        condition: str,
    ) -> Tuple[str, torch.Tensor, ClientRoundLog]:
        """Train one client for local_steps. Returns updated VQC + log."""
        # Load global VQC onto this client's device
        if condition == "QE-SAC-FL":
            _set_vqc_weights(agent, global_vqc.to(ccfg.device))

        metrics = trainer.train(n_steps=self.cfg.local_steps)

        log = ClientRoundLog(
            client_name   = ccfg.name,
            round_idx     = round_idx,
            mean_reward   = metrics.mean_reward(last_n=10),
            total_vviol   = metrics.total_v_viols(),
            steps_done    = self.cfg.local_steps,
            vqc_grad_norm = _vqc_grad_norm(agent),
        )
        # Move weights to CPU for aggregation
        return ccfg.name, _get_vqc_weights(agent).cpu(), log

    def run(self, condition: str = "QE-SAC-FL") -> FedResults:
        """
        Run full federated training.
        If cfg.parallel_clients=True and multiple GPUs available,
        each client trains simultaneously on its own GPU.
        """
        results = FedResults(config=self.cfg, condition=condition)
        t_start = time.time()

        n_clients = len(self.cfg.clients)
        print(f"\n{'='*60}")
        print(f"  {condition}  |  {n_clients} clients  |  "
              f"{self.cfg.n_rounds} rounds  |  {self.cfg.local_steps} steps/round")
        if self.cfg.parallel_clients:
            print(f"  Mode: PARALLEL — each client on its own GPU")
        else:
            print(f"  Mode: sequential")
        print(f"{'='*60}")

        # Build all clients
        clients = []
        for ccfg in self.cfg.clients:
            env, agent, trainer = self._build_client(ccfg)
            clients.append((ccfg, agent, trainer))
            print(f"  {ccfg.name:<25}  obs={ccfg.obs_dim}  "
                  f"device={ccfg.device}")

        # Global VQC starts as client 0's random init (on CPU)
        global_vqc = _get_vqc_weights(clients[0][1]).cpu()

        for round_idx in range(self.cfg.n_rounds):
            round_logs: Dict[str, ClientRoundLog] = {}
            updated_weights: List[torch.Tensor] = []

            if self.cfg.parallel_clients:
                # Each client trains on its own GPU in a separate thread
                with ThreadPoolExecutor(max_workers=n_clients) as executor:
                    futures = {
                        executor.submit(
                            self._train_one_client,
                            ccfg, agent, trainer,
                            global_vqc, round_idx, condition
                        ): ccfg.name
                        for ccfg, agent, trainer in clients
                    }
                    for future in as_completed(futures):
                        name, w, log = future.result()
                        updated_weights.append(w)
                        round_logs[name] = log
                        results.logs.append(log)
            else:
                for ccfg, agent, trainer in clients:
                    name, w, log = self._train_one_client(
                        ccfg, agent, trainer,
                        global_vqc, round_idx, condition
                    )
                    updated_weights.append(w)
                    round_logs[name] = log
                    results.logs.append(log)

            # FedAvg on CPU (reward-weighted if configured)
            if condition == "QE-SAC-FL" and self.cfg.federate_vqc_only:
                round_rewards = [round_logs[ccfg.name].mean_reward for ccfg, _, _ in clients]
                new_vqc = _fedavg(updated_weights, self.cfg.aggregation, round_rewards)
                # Server momentum: EMA prevents aggressive weight replacement
                if self.cfg.server_momentum > 0:
                    m = self.cfg.server_momentum
                    global_vqc = m * global_vqc + (1 - m) * new_vqc
                else:
                    global_vqc = new_vqc
                results.bytes_communicated += _bytes_per_vqc_update(n_clients)

            if (round_idx + 1) % self.cfg.log_interval == 0:
                reward_str = "  ".join(
                    f"{n.split('_')[1]}: {round_logs[n].mean_reward:+.2f}"
                    for ccfg, _, _ in clients
                    for n in [ccfg.name]
                )
                print(f"  Round {round_idx+1:>3}/{self.cfg.n_rounds}  |  {reward_str}")

        results.wall_time_seconds = time.time() - t_start
        return results

    def run_local_only(self) -> FedResults:
        """Condition: each client trains alone, no VQC sharing."""
        return self.run(condition="local_only")

    # -----------------------------------------------------------------------
    # Aligned federation (Solution 2 — SharedEncoderHead + VQC federated)
    # -----------------------------------------------------------------------

    def _build_aligned_client(self, client_cfg: ClientConfig) -> tuple:
        """Build (env, AlignedQESACAgent, QESACTrainer) for one client."""
        env         = _make_env(client_cfg.env_id, client_cfg.seed, client_cfg.reward_scale)
        obs_dim     = env.observation_space.shape[0]
        device_dims = list(map(int, env.action_space.nvec))
        agent = AlignedQESACAgent(
            obs_dim     = obs_dim,
            device_dims = device_dims,
            lr          = self.cfg.lr,
            gamma       = self.cfg.gamma,
            tau         = self.cfg.tau,
            alpha       = self.cfg.alpha,
            buffer_size = self.cfg.buffer_size,
            hidden_dim  = self.cfg.hidden_dim,
            device      = client_cfg.device,
        )
        trainer = QESACTrainer(
            agent, env,
            batch_size   = self.cfg.batch_size,
            warmup_steps = self.cfg.warmup_steps,
            log_interval = 9999,
            device       = client_cfg.device,
        )
        return env, agent, trainer

    def run_aligned(self) -> FedResults:
        """
        Aligned federation: federate SharedEncoderHead + VQC each round.
        Fixes heterogeneous latent space mismatch by aligning all clients
        into the same 8-dim latent space before the VQC sees it.
        """
        results = FedResults(config=self.cfg, condition="QE-SAC-FL-Aligned")
        t_start = time.time()
        n_clients = len(self.cfg.clients)

        print(f"\n{'='*60}")
        print(f"  QE-SAC-FL-Aligned  |  {n_clients} clients  |  "
              f"{self.cfg.n_rounds} rounds  |  {self.cfg.local_steps} steps/round")
        print(f"  Federating: SharedEncoderHead (272 params) + VQC (16 params)")
        print(f"{'='*60}")

        clients = []
        for ccfg in self.cfg.clients:
            env, agent, trainer = self._build_aligned_client(ccfg)
            clients.append((ccfg, agent, trainer))
            print(f"  {ccfg.name:<25}  obs={ccfg.obs_dim}  device={ccfg.device}")

        # Global shared weights start from client 0's random init
        global_shared = clients[0][1].get_shared_weights()
        # Move to CPU for aggregation
        global_shared = {
            "shared_head": {k: v.cpu() for k, v in global_shared["shared_head"].items()},
            "vqc":         global_shared["vqc"].cpu(),
        }

        for round_idx in range(self.cfg.n_rounds):
            updated_shared: List[dict] = []
            round_logs: Dict[str, ClientRoundLog] = {}

            def _train_aligned(ccfg, agent, trainer):
                # Load global shared weights
                agent.set_shared_weights({
                    "shared_head": {k: v.to(ccfg.device)
                                    for k, v in global_shared["shared_head"].items()},
                    "vqc": global_shared["vqc"].to(ccfg.device),
                })
                metrics = trainer.train(n_steps=self.cfg.local_steps)
                sw = agent.get_shared_weights()
                log = ClientRoundLog(
                    client_name   = ccfg.name,
                    round_idx     = round_idx,
                    mean_reward   = metrics.mean_reward(last_n=10),
                    total_vviol   = metrics.total_v_viols(),
                    steps_done    = self.cfg.local_steps,
                    vqc_grad_norm = float(
                        agent.actor.vqc.weights.grad.norm().item()
                        if agent.actor.vqc.weights.grad is not None else 0.0
                    ),
                )
                return ccfg.name, sw, log

            if self.cfg.parallel_clients:
                with ThreadPoolExecutor(max_workers=n_clients) as ex:
                    futures = {
                        ex.submit(_train_aligned, ccfg, agent, trainer): ccfg.name
                        for ccfg, agent, trainer in clients
                    }
                    for future in as_completed(futures):
                        name, sw, log = future.result()
                        updated_shared.append(sw)
                        round_logs[name] = log
                        results.logs.append(log)
            else:
                for ccfg, agent, trainer in clients:
                    name, sw, log = _train_aligned(ccfg, agent, trainer)
                    updated_shared.append(sw)
                    round_logs[name] = log
                    results.logs.append(log)

            # FedAvg: average SharedEncoderHead + VQC separately
            # Use reward-weighted aggregation so larger feeders don't dominate
            round_rewards = [round_logs[ccfg.name].mean_reward for ccfg, _, _ in clients]

            new_vqc = _fedavg(
                [s["vqc"].cpu() for s in updated_shared],
                self.cfg.aggregation,
                round_rewards,
            )
            new_head = fedavg_shared_head(
                [s["shared_head"] for s in updated_shared],
                aggregation=self.cfg.aggregation,
                rewards=round_rewards,
            )

            # Server momentum: EMA on global weights damps aggressive updates
            m = self.cfg.server_momentum
            if m > 0:
                global_shared = {
                    "vqc": m * global_shared["vqc"] + (1 - m) * new_vqc,
                    "shared_head": {
                        k: m * global_shared["shared_head"][k] + (1 - m) * new_head[k]
                        for k in new_head
                    },
                }
            else:
                global_shared = {"vqc": new_vqc, "shared_head": new_head}

            results.bytes_communicated += bytes_per_aligned_update(n_clients)

            if (round_idx + 1) % self.cfg.log_interval == 0:
                reward_str = "  ".join(
                    f"{n.split('_')[1]}: {round_logs[n].mean_reward:+.2f}"
                    for ccfg, _, _ in clients
                    for n in [ccfg.name]
                )
                print(f"  Round {round_idx+1:>3}/{self.cfg.n_rounds}  |  {reward_str}")

        results.wall_time_seconds = time.time() - t_start
        return results

    def run_all_conditions(self) -> Dict[str, FedResults]:
        """Run all active conditions. Returns dict for H1/H2/H3 comparison."""
        all_results = {}

        if self.cfg.run_local_only:
            print("\n[1/3] Running: local_only baseline...")
            all_results["local_only"] = self.run_local_only()

        if self.cfg.run_qe_sac_fl:
            print("\n[2/3] Running: QE-SAC-FL (VQC only, unaligned)...")
            all_results["QE-SAC-FL"] = self.run("QE-SAC-FL")

        print("\n[3/3] Running: QE-SAC-FL-Aligned (SharedHead + VQC)...")
        all_results["QE-SAC-FL-Aligned"] = self.run_aligned()

        return all_results


    # -----------------------------------------------------------------------
    # H6 — Partial participation (client dropout robustness)
    # -----------------------------------------------------------------------

    def run_partial_participation(
        self,
        dropout_rate: float = 0.33,
        seed: int = 42,
    ) -> FedResults:
        """
        H6: Aligned FL but only (1 - dropout_rate) fraction of clients
        participate each round, chosen randomly.

        With 3 clients and dropout_rate=0.33, exactly 2 clients participate
        per round. Tests robustness of quantum FL to client unavailability.
        """
        rng = np.random.default_rng(seed)
        results = FedResults(config=self.cfg, condition="QE-SAC-FL-Partial")
        t_start = time.time()
        n_clients = len(self.cfg.clients)
        n_active = max(2, int(round(n_clients * (1 - dropout_rate))))

        print(f"\n{'='*60}")
        print(f"  QE-SAC-FL-Partial  |  {n_clients} clients  |  "
              f"{self.cfg.n_rounds} rounds  |  "
              f"{n_active}/{n_clients} participate per round")
        print(f"{'='*60}")

        clients = []
        for ccfg in self.cfg.clients:
            env, agent, trainer = self._build_aligned_client(ccfg)
            clients.append((ccfg, agent, trainer))
            print(f"  {ccfg.name:<25}  obs={ccfg.obs_dim}  device={ccfg.device}")

        global_shared = clients[0][1].get_shared_weights()
        global_shared = {
            "shared_head": {k: v.cpu() for k, v in global_shared["shared_head"].items()},
            "vqc":         global_shared["vqc"].cpu(),
        }

        for round_idx in range(self.cfg.n_rounds):
            # Randomly select which clients participate this round
            active_idx = rng.choice(n_clients, size=n_active, replace=False)
            active_clients = [clients[i] for i in active_idx]

            updated_shared: List[dict] = []
            round_logs: Dict[str, ClientRoundLog] = {}

            def _train_partial(ccfg, agent, trainer):
                agent.set_shared_weights({
                    "shared_head": {k: v.to(ccfg.device)
                                    for k, v in global_shared["shared_head"].items()},
                    "vqc": global_shared["vqc"].to(ccfg.device),
                })
                metrics = trainer.train(n_steps=self.cfg.local_steps)
                sw = agent.get_shared_weights()
                log = ClientRoundLog(
                    client_name   = ccfg.name,
                    round_idx     = round_idx,
                    mean_reward   = metrics.mean_reward(last_n=10),
                    total_vviol   = metrics.total_v_viols(),
                    steps_done    = self.cfg.local_steps,
                    vqc_grad_norm = float(
                        agent.actor.vqc.weights.grad.norm().item()
                        if agent.actor.vqc.weights.grad is not None else 0.0
                    ),
                )
                return ccfg.name, sw, log

            if self.cfg.parallel_clients:
                with ThreadPoolExecutor(max_workers=n_active) as ex:
                    futures = {
                        ex.submit(_train_partial, ccfg, agent, trainer): ccfg.name
                        for ccfg, agent, trainer in active_clients
                    }
                    for future in as_completed(futures):
                        name, sw, log = future.result()
                        updated_shared.append(sw)
                        round_logs[name] = log
                        results.logs.append(log)
            else:
                for ccfg, agent, trainer in active_clients:
                    name, sw, log = _train_partial(ccfg, agent, trainer)
                    updated_shared.append(sw)
                    round_logs[name] = log
                    results.logs.append(log)

            # FedAvg over participating clients only (reward-weighted + momentum)
            active_rewards = [round_logs[ccfg.name].mean_reward
                              for ccfg, _, _ in active_clients]
            new_vqc = _fedavg(
                [s["vqc"].cpu() for s in updated_shared],
                self.cfg.aggregation, active_rewards,
            )
            new_head = fedavg_shared_head(
                [s["shared_head"] for s in updated_shared],
                aggregation=self.cfg.aggregation, rewards=active_rewards,
            )
            m = self.cfg.server_momentum
            if m > 0:
                global_shared = {
                    "vqc": m * global_shared["vqc"] + (1 - m) * new_vqc,
                    "shared_head": {
                        k: m * global_shared["shared_head"][k] + (1 - m) * new_head[k]
                        for k in new_head
                    },
                }
            else:
                global_shared = {"vqc": new_vqc, "shared_head": new_head}
            results.bytes_communicated += bytes_per_aligned_update(n_active)

            if (round_idx + 1) % self.cfg.log_interval == 0:
                active_names = [ccfg.name for ccfg, _, _ in active_clients]
                reward_str = "  ".join(
                    f"{n.split('_')[1]}: {round_logs[n].mean_reward:+.2f}"
                    for n in active_names
                )
                print(f"  Round {round_idx+1:>3}/{self.cfg.n_rounds}  "
                      f"[active: {[n.split('_')[1] for n in active_names]}]  "
                      f"|  {reward_str}")

        results.wall_time_seconds = time.time() - t_start
        return results

    # -----------------------------------------------------------------------
    # H5 — Personalised FL: aligned FL warm-start + local fine-tuning
    # -----------------------------------------------------------------------

    def run_personalized(
        self,
        n_fl_rounds: int = 50,
        n_finetune_steps: int = 5000,
    ) -> FedResults:
        """
        H5: Run aligned FL for n_fl_rounds, then fine-tune each client
        locally for n_finetune_steps WITHOUT federation.

        Returns FedResults for the fine-tuning phase.
        The FL phase results can be obtained via run_aligned() separately.

        Architecture: all actor params are unfrozen during fine-tuning
        (LocalEncoder + SharedHead + VQC all adapt to the local feeder).
        """
        results = FedResults(config=self.cfg, condition="QE-SAC-FL-Personalized")
        t_start = time.time()
        n_clients = len(self.cfg.clients)

        print(f"\n{'='*60}")
        print(f"  QE-SAC-FL-Personalized")
        print(f"  Phase 1: Aligned FL  {n_fl_rounds} rounds")
        print(f"  Phase 2: Local fine-tune  {n_finetune_steps:,} steps per client")
        print(f"{'='*60}")

        # Phase 1: aligned FL (same as run_aligned but returns agents)
        clients = []
        for ccfg in self.cfg.clients:
            env, agent, trainer = self._build_aligned_client(ccfg)
            clients.append((ccfg, agent, trainer))

        global_shared = clients[0][1].get_shared_weights()
        global_shared = {
            "shared_head": {k: v.cpu() for k, v in global_shared["shared_head"].items()},
            "vqc":         global_shared["vqc"].cpu(),
        }

        print(f"\nPhase 1: Aligned FL ({n_fl_rounds} rounds)...")
        for round_idx in range(n_fl_rounds):
            updated_shared: List[dict] = []

            def _fl_round(ccfg, agent, trainer):
                agent.set_shared_weights({
                    "shared_head": {k: v.to(ccfg.device)
                                    for k, v in global_shared["shared_head"].items()},
                    "vqc": global_shared["vqc"].to(ccfg.device),
                })
                trainer.train(n_steps=self.cfg.local_steps)
                return agent.get_shared_weights()

            if self.cfg.parallel_clients:
                with ThreadPoolExecutor(max_workers=n_clients) as ex:
                    futures = [ex.submit(_fl_round, ccfg, agent, trainer)
                               for ccfg, agent, trainer in clients]
                    for fut in as_completed(futures):
                        updated_shared.append(fut.result())
            else:
                for ccfg, agent, trainer in clients:
                    updated_shared.append(_fl_round(ccfg, agent, trainer))

            global_shared = {
                "shared_head": fedavg_shared_head(
                    [s["shared_head"] for s in updated_shared]
                ),
                "vqc": torch.stack(
                    [s["vqc"].cpu() for s in updated_shared], dim=0
                ).mean(dim=0),
            }
            if (round_idx + 1) % self.cfg.log_interval == 0:
                print(f"  FL round {round_idx+1}/{n_fl_rounds} done")

        # Phase 2: local fine-tuning — no federation
        print(f"\nPhase 2: Local fine-tuning ({n_finetune_steps:,} steps/client)...")
        # Broadcast final global weights before fine-tuning starts
        for ccfg, agent, _ in clients:
            agent.set_shared_weights({
                "shared_head": {k: v.to(ccfg.device)
                                for k, v in global_shared["shared_head"].items()},
                "vqc": global_shared["vqc"].to(ccfg.device),
            })

        def _finetune(ccfg, agent, trainer, round_idx=0):
            metrics = trainer.train(n_steps=n_finetune_steps)
            log = ClientRoundLog(
                client_name   = ccfg.name,
                round_idx     = round_idx,
                mean_reward   = metrics.mean_reward(last_n=10),
                total_vviol   = metrics.total_v_viols(),
                steps_done    = n_finetune_steps,
                vqc_grad_norm = float(
                    agent.actor.vqc.weights.grad.norm().item()
                    if agent.actor.vqc.weights.grad is not None else 0.0
                ),
            )
            return ccfg.name, log

        if self.cfg.parallel_clients:
            with ThreadPoolExecutor(max_workers=n_clients) as ex:
                futures = {
                    ex.submit(_finetune, ccfg, agent, trainer): ccfg.name
                    for ccfg, agent, trainer in clients
                }
                for future in as_completed(futures):
                    name, log = future.result()
                    results.logs.append(log)
                    print(f"  {name:<25}  reward={log.mean_reward:+.3f}")
        else:
            for ccfg, agent, trainer in clients:
                name, log = _finetune(ccfg, agent, trainer)
                results.logs.append(log)
                print(f"  {name:<25}  reward={log.mean_reward:+.3f}")

        results.bytes_communicated = n_fl_rounds * bytes_per_aligned_update(n_clients)
        results.wall_time_seconds = time.time() - t_start
        return results


# ---------------------------------------------------------------------------
# Communication cost calculator (for H3 table)
# ---------------------------------------------------------------------------

def communication_cost_table(n_rounds: int, n_clients: int) -> None:
    """H3: total bytes for all three conditions vs Federated Classical SAC."""
    from src.qe_sac_fl.aligned_encoder import shared_head_param_count

    vqc_params       = 16
    head_params      = shared_head_param_count()   # 272
    aligned_params   = vqc_params + head_params    # 288
    classical_params = 110_724

    bytes_qe      = n_rounds * n_clients * 2 * vqc_params      * 4
    bytes_aligned = n_rounds * n_clients * 2 * aligned_params   * 4
    bytes_cl      = n_rounds * n_clients * 2 * classical_params * 4

    print(f"\n{'─'*60}")
    print(f"  H3: Communication cost  ({n_rounds} rounds, {n_clients} clients)")
    print(f"{'─'*60}")
    print(f"  QE-SAC-FL (VQC only)        : {bytes_qe:>10,} bytes  "
          f"({bytes_qe/1024:.1f} KB)")
    print(f"  QE-SAC-FL-Aligned (Head+VQC): {bytes_aligned:>10,} bytes  "
          f"({bytes_aligned/1024:.1f} KB)")
    print(f"  Fed Classical SAC           : {bytes_cl:>10,} bytes  "
          f"({bytes_cl/1024/1024:.1f} MB)")
    print(f"  Aligned vs Classical        : {bytes_cl/bytes_aligned:>10.0f}x reduction")
    print(f"{'─'*60}")
