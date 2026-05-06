#!/usr/bin/env python3
"""
VQC Circuit Ablation Study
===========================
Compares 5 VQC circuit architectures on the 13-bus Volt-VAR control task.

Circuits:
  paper   - RY + linear CNOT + RX, 2 layers, 16 params  [paper choice]
  ring    - RY + ring CNOT   + RX, 2 layers, 16 params
  no_ent  - RY + RX, no entanglement,        2 layers, 16 params
  deep    - RY + linear CNOT + RX, 4 layers, 32 params
  sel     - Strongly Entangling Layers,       2 layers, 48 params

Key metrics:
  - VQC gradient norm per round  (barren plateau detection)
  - Episode reward per round     (performance)
  - Total trainable params       (communication cost)

References:
  Sim et al. 2019  (arXiv:1905.10876) — expressibility metric
  McClean et al. 2018 (Nature Comm)  — barren plateau theorem
  Cerezo et al. 2021 (Nature Comm)   — cost-function barren plateaus

Output: artifacts/qe_sac_fl/circuit_ablation/
"""

from __future__ import annotations

import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.qe_sac.env_utils import VVCEnv13Bus
from src.qe_sac_fl.aligned_agent import AlignedQESACAgent
from src.qe_sac.vqc import (
    _ry_matrix, _rx_matrix,
    _apply_single_qubit_gate, _apply_cnot, _pauli_z_expectation,
    N_QUBITS, N_LAYERS,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_ROUNDS    = 30          # rounds per circuit (30K steps total — fast enough)
LOCAL_STEPS = 1_000
WARMUP      = 1_000
BATCH_SIZE  = 256
LR          = 3e-4
BUFFER_SIZE = 50_000
REWARD_SCALE = 50.0       # Client A normalisation
SEEDS       = [0, 1]      # 2 seeds — enough to show the pattern
SAVE_DIR    = "artifacts/qe_sac_fl/circuit_ablation"
DEVICE      = "cuda:0" if torch.cuda.is_available() else "cpu"

# 13-bus Client A dims (from FedConfig)
OBS_DIM     = 43
DEVICE_DIMS = [2, 2, 33, 33]   # MultiDiscrete action dims (2 caps + 1 reg + 1 battery)

DIM = 2 ** N_QUBITS   # 256


# ---------------------------------------------------------------------------
# Circuit implementations
# ---------------------------------------------------------------------------

def _forward_paper(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Paper circuit (Lin et al. 2025, Fig.2):
      RY(z_i) encoding → linear CNOT chain → RX(θ) rotations  × 2 layers
    Params: N_LAYERS × N_QUBITS = 16
    Entanglement: linear (0→1→2→...→6→7)
    """
    B   = inputs.shape[0]
    dev = inputs.device
    psi = torch.zeros(B, DIM, dtype=torch.complex64, device=dev)
    psi[:, 0] = 1.0

    ry = _ry_matrix(inputs).to(torch.complex64)
    for i in range(N_QUBITS):
        psi = _apply_single_qubit_gate(psi, ry[:, i], i)

    rx = _rx_matrix(weights)
    for layer in range(N_LAYERS):
        for i in range(N_QUBITS - 1):           # linear chain
            psi = _apply_cnot(psi, i, i + 1)
        for i in range(N_QUBITS):
            psi = _apply_single_qubit_gate(psi, rx[layer, i], i)

    return torch.stack([_pauli_z_expectation(psi, i) for i in range(N_QUBITS)], dim=1)


def _forward_ring(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Ring entanglement:
      RY(z_i) → ring CNOT (0→1→...→6→7→0) → RX(θ)  × 2 layers
    Params: 16  (same count as paper)
    Difference: adds CNOT(7, 0) to close the ring.
    Sim et al. 2019: ring slightly higher entanglement than linear at same depth.
    """
    B   = inputs.shape[0]
    dev = inputs.device
    psi = torch.zeros(B, DIM, dtype=torch.complex64, device=dev)
    psi[:, 0] = 1.0

    ry = _ry_matrix(inputs).to(torch.complex64)
    for i in range(N_QUBITS):
        psi = _apply_single_qubit_gate(psi, ry[:, i], i)

    rx = _rx_matrix(weights)
    for layer in range(N_LAYERS):
        for i in range(N_QUBITS - 1):           # linear part
            psi = _apply_cnot(psi, i, i + 1)
        psi = _apply_cnot(psi, N_QUBITS - 1, 0) # close ring
        for i in range(N_QUBITS):
            psi = _apply_single_qubit_gate(psi, rx[layer, i], i)

    return torch.stack([_pauli_z_expectation(psi, i) for i in range(N_QUBITS)], dim=1)


def _forward_no_ent(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    No entanglement (product state):
      RY(z_i) → RX(θ_i) per qubit  × 2 layers, no CNOT at all.
    Params: 16  (same count as paper)
    Expected failure: qubits are independent, cannot learn correlations between buses.
    McClean et al. 2018: product-state circuits cannot represent entangled policies.
    """
    B   = inputs.shape[0]
    dev = inputs.device
    psi = torch.zeros(B, DIM, dtype=torch.complex64, device=dev)
    psi[:, 0] = 1.0

    ry = _ry_matrix(inputs).to(torch.complex64)
    for i in range(N_QUBITS):
        psi = _apply_single_qubit_gate(psi, ry[:, i], i)

    rx = _rx_matrix(weights)
    for layer in range(N_LAYERS):
        # NO CNOT — qubits remain unentangled
        for i in range(N_QUBITS):
            psi = _apply_single_qubit_gate(psi, rx[layer, i], i)

    return torch.stack([_pauli_z_expectation(psi, i) for i in range(N_QUBITS)], dim=1)


def _forward_deep(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Deep circuit: same as paper but 4 layers instead of 2.
    Params: 4 × 8 = 32
    Expected failure: deeper circuits have exponentially smaller gradient variance
    (McClean et al. 2018: Var[∂L/∂θ] ~ O(2^{-n×L}) for random init).
    """
    n_layers = weights.shape[0]   # will be 4
    B   = inputs.shape[0]
    dev = inputs.device
    psi = torch.zeros(B, DIM, dtype=torch.complex64, device=dev)
    psi[:, 0] = 1.0

    ry = _ry_matrix(inputs).to(torch.complex64)
    for i in range(N_QUBITS):
        psi = _apply_single_qubit_gate(psi, ry[:, i], i)

    rx = _rx_matrix(weights)
    for layer in range(n_layers):
        for i in range(N_QUBITS - 1):
            psi = _apply_cnot(psi, i, i + 1)
        for i in range(N_QUBITS):
            psi = _apply_single_qubit_gate(psi, rx[layer, i], i)

    return torch.stack([_pauli_z_expectation(psi, i) for i in range(N_QUBITS)], dim=1)


def _rz_matrix(theta: torch.Tensor) -> torch.Tensor:
    """RZ(θ) = diag(e^{-iθ/2}, e^{+iθ/2}) as complex64 matrix."""
    half = (theta / 2).to(torch.complex64)
    z    = torch.zeros_like(half)
    return torch.stack([
        torch.stack([torch.exp(-1j * half), z], dim=-1),
        torch.stack([z, torch.exp( 1j * half)], dim=-1),
    ], dim=-2)


def _forward_sel(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Strongly Entangling Layers (SEL) — Sim et al. 2019, Fig.1 Circuit 14.
    Each layer:
      RZ(φ_i) RY(θ_i) RZ(ω_i) on each qubit   [3 params/qubit]
      CNOT(i, (i + step) mod N_QUBITS)           step = layer+1
    Params: 3 × N_QUBITS × N_LAYERS = 3 × 8 × 2 = 48

    Highest expressibility of all 5 circuits, but:
    - More params → federation cost 3× higher
    - Barren plateau risk increases with expressibility (Cerezo et al. 2021)
    """
    n_layers = weights.shape[0]   # 2
    B   = inputs.shape[0]
    dev = inputs.device
    psi = torch.zeros(B, DIM, dtype=torch.complex64, device=dev)
    psi[:, 0] = 1.0

    ry = _ry_matrix(inputs).to(torch.complex64)
    for i in range(N_QUBITS):
        psi = _apply_single_qubit_gate(psi, ry[:, i], i)

    # weights shape: (n_layers, N_QUBITS, 3)  — [φ, θ, ω] per qubit per layer
    for layer in range(n_layers):
        phi   = weights[layer, :, 0]
        theta = weights[layer, :, 1]
        omega = weights[layer, :, 2]

        rz_phi   = _rz_matrix(phi)
        ry_theta = _ry_matrix(theta).to(torch.complex64)
        rz_omega = _rz_matrix(omega)

        for i in range(N_QUBITS):
            psi = _apply_single_qubit_gate(psi, rz_phi[i],   i)
            psi = _apply_single_qubit_gate(psi, ry_theta[i], i)
            psi = _apply_single_qubit_gate(psi, rz_omega[i], i)

        step = (layer % (N_QUBITS - 1)) + 1
        for i in range(N_QUBITS):
            psi = _apply_cnot(psi, i, (i + step) % N_QUBITS)

    return torch.stack([_pauli_z_expectation(psi, i) for i in range(N_QUBITS)], dim=1)


# ---------------------------------------------------------------------------
# FlexVQCLayer — drop-in replacement for VQCLayer with configurable circuit
# ---------------------------------------------------------------------------

CIRCUIT_REGISTRY = {
    "paper":  (_forward_paper,  (N_LAYERS,  N_QUBITS),      16),
    "ring":   (_forward_ring,   (N_LAYERS,  N_QUBITS),      16),
    "no_ent": (_forward_no_ent, (N_LAYERS,  N_QUBITS),      16),
    "deep":   (_forward_deep,   (4,         N_QUBITS),       32),
    "sel":    (_forward_sel,    (N_LAYERS,  N_QUBITS, 3),   48),
}


class FlexVQCLayer(nn.Module):
    """VQC with swappable circuit type for ablation study."""

    def __init__(self, circuit_type: str = "paper"):
        super().__init__()
        assert circuit_type in CIRCUIT_REGISTRY, \
            f"Unknown circuit: {circuit_type}. Choose from {list(CIRCUIT_REGISTRY)}"
        self._type     = circuit_type
        self._fwd, w_shape, self._n_params = CIRCUIT_REGISTRY[circuit_type]
        self.weights = nn.Parameter(torch.randn(*w_shape) * 0.1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() == 1:
            return self._fwd(inputs.unsqueeze(0), self.weights).squeeze(0)
        return self._fwd(inputs, self.weights)

    @property
    def n_params(self) -> int:
        return self._n_params


# ---------------------------------------------------------------------------
# Build agent with chosen circuit
# ---------------------------------------------------------------------------

def build_agent(circuit_type: str, seed: int) -> AlignedQESACAgent:
    torch.manual_seed(seed)
    np.random.seed(seed)

    agent = AlignedQESACAgent(
        obs_dim      = OBS_DIM,
        device_dims  = DEVICE_DIMS,
        lr           = LR,
        buffer_size  = BUFFER_SIZE,
        device       = DEVICE,
        hidden_dim   = 32,
    )
    # Swap VQC with chosen circuit type
    agent.actor.vqc = FlexVQCLayer(circuit_type).to(DEVICE)
    # Re-build actor optimiser to include new VQC params
    agent.actor_opt = torch.optim.Adam(agent.actor.parameters(), lr=LR)

    return agent


# ---------------------------------------------------------------------------
# Gradient norm helper
# ---------------------------------------------------------------------------

def vqc_grad_norm(agent: AlignedQESACAgent) -> float:
    """L2 norm of VQC weight gradients. Returns 0 if no grad yet."""
    g = agent.actor.vqc.weights.grad
    if g is None:
        return 0.0
    return g.norm(2).item()


# ---------------------------------------------------------------------------
# Training loop — local only (no federation needed for circuit comparison)
# ---------------------------------------------------------------------------

def run_circuit(circuit_type: str, seed: int) -> dict:
    print(f"\n{'='*60}")
    print(f"  Circuit: {circuit_type:<10}  |  Seed: {seed}")
    print(f"{'='*60}")

    env   = VVCEnv13Bus(seed=seed)
    agent = build_agent(circuit_type, seed)

    n_params    = agent.actor.vqc.n_params
    grad_norms  = []
    rewards     = []
    total_steps = 0

    obs, _ = env.reset()

    # Warm-up: fill buffer before any gradient updates
    print(f"  Warm-up ({WARMUP} steps)...")
    for _ in range(WARMUP):
        obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        action = agent.actor.select_action(obs_t)
        next_obs, reward, done, trunc, _ = env.step(action)
        agent.store(obs, action, reward / REWARD_SCALE, next_obs, done or trunc)
        obs = next_obs
        if done or trunc:
            obs, _ = env.reset()
        total_steps += 1

    # Training rounds
    for rnd in range(N_ROUNDS):
        ep_rewards = []
        ep_rew     = 0.0
        obs, _     = env.reset()

        for step in range(LOCAL_STEPS):
            obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            action = agent.actor.select_action(obs_t)
            next_obs, reward, done, trunc, _ = env.step(action)
            scaled = reward / REWARD_SCALE
            agent.store(obs, action, scaled, next_obs, done or trunc)
            ep_rew += scaled
            obs     = next_obs
            total_steps += 1

            if done or trunc:
                ep_rewards.append(ep_rew)
                ep_rew = 0.0
                obs, _ = env.reset()

            # SAC update
            agent.update(batch_size=BATCH_SIZE)

        # Capture gradient norm after last update in round
        gn = vqc_grad_norm(agent)
        grad_norms.append(gn)

        mean_rew = float(np.mean(ep_rewards)) if ep_rewards else float("nan")
        rewards.append(mean_rew)

        print(f"  Round {rnd+1:>3}/{N_ROUNDS} | "
              f"reward={mean_rew:+.3f} | grad_norm={gn:.2e}")

    env.close()

    return {
        "circuit":    circuit_type,
        "seed":       seed,
        "n_params":   n_params,
        "grad_norms": grad_norms,
        "rewards":    rewards,
        "final_reward": float(np.nanmean(rewards[-5:])),  # last 5 rounds
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = {
    "paper":  "#2196F3",   # blue  — our choice
    "ring":   "#4CAF50",   # green
    "no_ent": "#FF5722",   # red-orange
    "deep":   "#9C27B0",   # purple
    "sel":    "#FF9800",   # orange
}

LABELS = {
    "paper":  "Paper (RY+linCNOT+RX, 2L, 16p)",
    "ring":   "Ring CNOT (2L, 16p)",
    "no_ent": "No Entanglement (2L, 16p)",
    "deep":   "Deep (RY+linCNOT+RX, 4L, 32p)",
    "sel":    "SEL (RZ+RY+RZ+multiCNOT, 2L, 48p)",
}


def plot_results(all_results: dict, save_dir: str) -> None:
    circuits = [c for c in CIRCUIT_REGISTRY.keys() if all_results.get(c)]
    rounds   = np.arange(1, N_ROUNDS + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("VQC Circuit Ablation — 13-bus Volt-VAR Control\n"
                 "(Local training, 30K steps, 2 seeds ± 1 std)",
                 fontsize=13, fontweight="bold")

    # ── Panel 1: gradient norms (log scale) ─────────────────────────────────
    ax = axes[0]
    ax.set_title("VQC Gradient Norm ‖∇θ‖₂\n(collapse = barren plateau)", fontsize=11)
    ax.set_xlabel("Training Round")
    ax.set_ylabel("Gradient Norm (log scale)")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1e0)

    for circ in circuits:
        gns = [r["grad_norms"] for r in all_results[circ]]
        if not gns:
            continue
        mn  = np.nanmean(gns, axis=0)
        std = np.nanstd(gns, axis=0)
        ax.plot(rounds, mn, color=COLORS[circ], label=LABELS[circ], linewidth=2)
        ax.fill_between(rounds, np.clip(mn - std, 1e-6, None), mn + std,
                        alpha=0.15, color=COLORS[circ])

    ax.axhline(1e-3, color="red", linestyle="--", linewidth=1, alpha=0.7,
               label="Barren plateau threshold (10⁻³)")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, which="both", alpha=0.3)

    # ── Panel 2: episode reward ──────────────────────────────────────────────
    ax = axes[1]
    ax.set_title("Episode Reward (normalised)\n(higher = better)", fontsize=11)
    ax.set_xlabel("Training Round")
    ax.set_ylabel("Mean Episode Reward")

    for circ in circuits:
        rws = [r["rewards"] for r in all_results[circ]]
        if not rws:
            continue
        mn  = np.nanmean(rws, axis=0)
        std = np.nanstd(rws, axis=0)
        ax.plot(rounds, mn, color=COLORS[circ], label=LABELS[circ], linewidth=2)
        ax.fill_between(rounds, mn - std, mn + std, alpha=0.15, color=COLORS[circ])

    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: final reward vs param count ─────────────────────────────────
    ax = axes[2]
    ax.set_title("Final Reward vs. Param Count\n(top-left = best)", fontsize=11)
    ax.set_xlabel("Federated Params (VQC only)")
    ax.set_ylabel("Final Reward (last 5 rounds)")

    for circ in circuits:
        res = all_results[circ]
        if not res:
            continue
        n_params = res[0]["n_params"]
        final    = [r["final_reward"] for r in res]
        mn, std  = np.nanmean(final), np.nanstd(final)
        ax.scatter(n_params, mn, color=COLORS[circ], s=120, zorder=3)
        ax.errorbar(n_params, mn, yerr=std, color=COLORS[circ],
                    capsize=5, linewidth=2)
        ax.annotate(circ, (n_params, mn),
                    textcoords="offset points", xytext=(6, 4), fontsize=8,
                    color=COLORS[circ], fontweight="bold")

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(save_dir, "vqc_circuit_ablation.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--circuits", nargs="+",
                        default=list(CIRCUIT_REGISTRY.keys()),
                        choices=list(CIRCUIT_REGISTRY.keys()),
                        help="Which circuits to test (default: all 5)")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)

    all_results: dict[str, list] = {c: [] for c in args.circuits}
    t0 = time.time()

    for circuit_type in args.circuits:
        for seed in args.seeds:
            result = run_circuit(circuit_type, seed)
            all_results[circuit_type].append(result)

            # Save per-run JSON
            fname = f"{SAVE_DIR}/{circuit_type}_seed{seed}.json"
            with open(fname, "w") as f:
                json.dump(result, f, indent=2)

    # Summary table
    print(f"\n{'='*70}")
    print(f"  CIRCUIT ABLATION SUMMARY  ({(time.time()-t0)/60:.1f} min total)")
    print(f"{'='*70}")
    print(f"  {'Circuit':<10} {'Params':>7}  {'Grad Norm (mean)':>18}  "
          f"{'Final Reward':>14}  {'Barren?':>8}")
    print(f"  {'-'*66}")

    for circ in args.circuits:
        res = all_results[circ]
        if not res:
            continue
        n_params   = res[0]["n_params"]
        gn_means   = [np.mean(r["grad_norms"][-10:]) for r in res]  # last 10 rounds
        rew_finals = [r["final_reward"] for r in res]
        gn_mean    = np.mean(gn_means)
        rew_mean   = np.mean(rew_finals)
        barren     = "YES" if gn_mean < 1e-3 else "no"
        print(f"  {circ:<10} {n_params:>7}  {gn_mean:>18.2e}  "
              f"{rew_mean:>14.3f}  {barren:>8}")

    # Save summary JSON
    summary = {}
    for circ, res in all_results.items():
        if not res:
            continue
        summary[circ] = {
            "n_params":     res[0]["n_params"],
            "mean_final_reward": float(np.mean([r["final_reward"] for r in res])),
            "std_final_reward":  float(np.std([r["final_reward"] for r in res])),
            "mean_final_grad_norm": float(np.mean([
                np.mean(r["grad_norms"][-10:]) for r in res
            ])),
            "barren_plateau": float(np.mean([
                np.mean(r["grad_norms"][-10:]) for r in res
            ])) < 1e-3,
        }

    with open(f"{SAVE_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary → {SAVE_DIR}/summary.json")

    # Plot
    plot_results(all_results, SAVE_DIR)
    print(f"\n  Done. Total time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
