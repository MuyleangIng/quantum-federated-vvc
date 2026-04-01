"""
Co-Adaptive Autoencoder (CAE) for QE-SAC.

Compresses the high-dimensional grid state s → 8-dim latent s'.
The encoder output is scaled to [-π, π] for VQC angle encoding.

Architecture:
    Encoder: input_dim → 64 → 32 → 8  (tanh → scaled to [-π, π])
    Decoder: 8 → 32 → 64 → input_dim  (for reconstruction loss)

Co-adaptive retraining: called every C=500 gradient steps during RL training,
training on a recent batch from the replay buffer so the latent space stays
aligned with the current data distribution.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class CAE(nn.Module):
    """
    Classical Autoencoder with configurable latent dimension.

    Default latent_dim=8 matches the paper (8-qubit VQC input).
    Use latent_dim=n_qubits for VQC ablation with different qubit counts.
    """

    LATENT_DIM = 8   # kept for backward compatibility

    def __init__(
        self,
        input_dim:   int,
        hidden_dims: tuple[int, int] = (64, 32),
        latent_dim:  int = 8,
    ):
        super().__init__()
        h1, h2 = hidden_dims
        self._latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, latent_dim),
            nn.Tanh(),  # output in (-1, 1); scaled to (-π, π) in encode()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2),
            nn.ReLU(),
            nn.Linear(h2, h1),
            nn.ReLU(),
            nn.Linear(h1, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent vector scaled to [-π, π] for angle encoding."""
        return self.encoder(x) * torch.pi

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z / torch.pi)   # undo π scaling before decode

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


def train_cae(
    cae: CAE,
    observations: np.ndarray | torch.Tensor,
    n_steps: int = 200,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: str = "cpu",
    latent_dim: int = 8,   # unused here; kept for ablation script compatibility
) -> float:
    """
    Train the CAE for *n_steps* gradient steps on *observations*.

    Parameters
    ----------
    cae          : CAE model (updated in-place)
    observations : array of shape (N, obs_dim) — recent replay buffer samples
    n_steps      : number of gradient steps
    lr           : learning rate
    batch_size   : mini-batch size
    device       : torch device string

    Returns
    -------
    final_loss : float — mean reconstruction loss of the last batch
    """
    cae.to(device)
    cae.train()

    if not isinstance(observations, torch.Tensor):
        observations = torch.tensor(observations, dtype=torch.float32)
    observations = observations.to(device)

    optimizer = optim.Adam(cae.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    N = len(observations)
    final_loss = 0.0

    for _ in range(n_steps):
        idx = torch.randint(0, N, (min(batch_size, N),))
        batch = observations[idx]
        x_hat, _ = cae(batch)
        loss = loss_fn(x_hat, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    return final_loss
