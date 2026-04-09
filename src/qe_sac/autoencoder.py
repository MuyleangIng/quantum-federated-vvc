"""
Co-Adaptive Autoencoder (CAE) for QE-SAC.

Compresses the high-dimensional grid state s → 8-dim latent s'.
The encoder output is scaled to [-π, π] for VQC angle encoding.

Architecture:
    Encoder: input_dim → 64 → 8  (tanh → scaled to [-π, π])
    Decoder: 8 → 64 → input_dim  (for reconstruction loss)

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
        input_dim:  int,
        hidden_dim: int = 64,
        latent_dim: int = 8,
        # legacy two-tuple argument kept for backward compat
        hidden_dims: tuple | None = None,
    ):
        super().__init__()
        if hidden_dims is not None:
            hidden_dim = hidden_dims[0]
        self._latent_dim = latent_dim

        # Encoder: input → hidden_dim → latent_dim
        # Paper specifies FC layers with latent_dim=8; hidden_dim=64 is an
        # implementation choice consistent with the paper's parameter count.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh(),   # output in (-1, 1); scaled to (-π, π) in encode()
        )

        # Decoder mirrors encoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
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


def collect_random_observations(env, n_steps: int = 5000) -> np.ndarray:
    """
    Collect observations from a random policy (paper Section III-C).

    Run the environment for *n_steps* steps using a purely random policy
    and return the collected observations as an array.  Used to pre-train
    the CAE offline before RL training begins.

    Parameters
    ----------
    env     : Gymnasium VVC environment (reset() / step() interface)
    n_steps : number of random environment steps to collect

    Returns
    -------
    observations : np.ndarray of shape (n_steps, obs_dim)
    """
    obs_list = []
    obs, _ = env.reset()
    for _ in range(n_steps):
        obs_list.append(obs)
        action = env.action_space.sample()
        next_obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs
    return np.array(obs_list, dtype=np.float32)


def train_cae(
    cae: CAE,
    observations: np.ndarray | torch.Tensor,
    n_steps: int = 200,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: str = "cpu",
    latent_dim: int = 8,       # unused; kept for ablation script compatibility
    optimizer: optim.Optimizer | None = None,  # pass persistent optimizer if available
) -> float:
    """
    Train the CAE for *n_steps* gradient steps on *observations*.

    Pass a persistent *optimizer* (owned by the agent) to avoid recreating
    Adam on every co-adaptive retraining call.  If None, a fresh Adam is
    created (used for stand-alone / offline pretraining calls).

    Parameters
    ----------
    cae          : CAE model (updated in-place)
    observations : array of shape (N, obs_dim)
    n_steps      : number of gradient steps
    lr           : learning rate (ignored when optimizer is provided)
    batch_size   : mini-batch size
    device       : torch device string
    optimizer    : persistent optimizer; if None a temporary Adam is created

    Returns
    -------
    final_loss : float — reconstruction loss of the last batch
    """
    cae.to(device)
    cae.train()

    if not isinstance(observations, torch.Tensor):
        observations = torch.tensor(observations, dtype=torch.float32)
    observations = observations.to(device)

    _opt    = optimizer if optimizer is not None else optim.Adam(cae.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    N       = len(observations)
    final_loss = 0.0

    for _ in range(n_steps):
        idx   = torch.randint(0, N, (min(batch_size, N),))
        batch = observations[idx]
        x_hat, _ = cae(batch)
        loss = loss_fn(x_hat, batch)
        _opt.zero_grad()
        loss.backward()
        _opt.step()
        final_loss = loss.item()

    return final_loss
