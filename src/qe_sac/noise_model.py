"""
Noise model utilities for QE-SAC robustness evaluation.

Tests the VQCLayer under depolarising noise at λ = 0.1%, 0.5%, 1.0%
to replicate the paper's noise robustness experiments.
"""

from __future__ import annotations

import torch
import numpy as np
from .vqc import VQCLayer, N_QUBITS


NOISE_LEVELS = [0.001, 0.005, 0.010]   # λ = 0.1%, 0.5%, 1.0%


def make_noisy_vqc(noise_lambda: float) -> VQCLayer:
    """Return a VQCLayer configured with depolarising noise λ."""
    return VQCLayer(noise_lambda=noise_lambda)


def evaluate_noise_robustness(
    clean_vqc: VQCLayer,
    n_samples: int = 100,
    noise_levels: list[float] | None = None,
    device: str = "cpu",
) -> dict[float, dict[str, float]]:
    """
    Compare clean VQC output against noisy versions on random inputs.

    Parameters
    ----------
    clean_vqc   : trained noiseless VQCLayer (weights will be copied)
    n_samples   : number of random input vectors to test
    noise_levels: list of λ values to test (default: NOISE_LEVELS)
    device      : torch device

    Returns
    -------
    results : dict mapping λ → {"mean_output_diff": ..., "std_output_diff": ...}
    """
    if noise_levels is None:
        noise_levels = NOISE_LEVELS

    clean_vqc.to(device)
    clean_vqc.eval()

    rng = torch.Generator()
    rng.manual_seed(0)
    inputs = torch.rand(n_samples, N_QUBITS, generator=rng, device=device) * 2 * torch.pi - torch.pi

    with torch.no_grad():
        clean_out = clean_vqc(inputs)   # (n_samples, N_QUBITS)

    results: dict[float, dict[str, float]] = {}
    for lam in noise_levels:
        noisy_vqc = make_noisy_vqc(lam)
        # Copy weights from clean VQC
        noisy_vqc.weights = torch.nn.Parameter(clean_vqc.weights.detach().clone())
        noisy_vqc.to(device)
        noisy_vqc.eval()

        with torch.no_grad():
            noisy_out = torch.stack([
                torch.stack(noisy_vqc._get_pl_circuit()(x, noisy_vqc.weights))
                for x in inputs
            ])

        diff = (clean_out - noisy_out).abs()
        results[lam] = {
            "mean_output_diff": float(diff.mean()),
            "std_output_diff":  float(diff.std()),
        }

    return results
