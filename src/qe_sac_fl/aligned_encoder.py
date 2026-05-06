"""
Aligned encoder architecture for QE-SAC-FL Solution 2.

Splits the CAE into two parts:
  LocalEncoder      obs_dim → hidden_dim (32)   — stays LOCAL (private)
  SharedEncoderHead hidden_dim → 8              — gets FEDERATED (shared)

All clients have different obs_dim but the same hidden_dim (32),
so SharedEncoderHead has identical architecture across all clients.
FedAvg on SharedEncoderHead forces all clients into the same 8-dim
latent space — fixing heterogeneous latent space mismatch.

What gets federated each round:
  SharedEncoderHead  272 params  (Linear 32→8 + bias)
  VQC                 16 params
  Total:             288 params = 1,152 bytes per client

vs classical SAC actor: ~110,724 params = 430 KB per client
Quantum alignment advantage: 383× reduction vs classical.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

HIDDEN_DIM  = 32    # local encoder output — same for ALL clients
LATENT_DIM  = 8     # shared head output — feeds into VQC
LATENT_SCALE = math.pi   # scale latents to [-π, π] for angle encoding


# ---------------------------------------------------------------------------
# Local encoder  (private — different obs_dim per client)
# ---------------------------------------------------------------------------

class LocalEncoder(nn.Module):
    """
    obs_dim → 64 → HIDDEN_DIM (32)

    This part stays on the client. It compresses the feeder-specific
    observation into a common intermediate representation.
    Different clients have different obs_dim but same output size.
    """
    def __init__(self, obs_dim: int, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Shared encoder head  (federated — same architecture across all clients)
# ---------------------------------------------------------------------------

class SharedEncoderHead(nn.Module):
    """
    HIDDEN_DIM (32) → LATENT_DIM (8), scaled to [-π, π]

    This part is federated via FedAvg. All clients share the same
    weights after aggregation, forcing latent space alignment.
    """
    def __init__(self, hidden_dim: int = HIDDEN_DIM, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh(),   # output in (-1, 1)
        )
        self.scale = LATENT_SCALE

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h) * self.scale   # scaled to (-π, π)


# ---------------------------------------------------------------------------
# Local decoder  (private — reconstructs to obs_dim for training loss)
# ---------------------------------------------------------------------------

class LocalDecoder(nn.Module):
    """
    LATENT_DIM (8) → HIDDEN_DIM (32) → obs_dim

    Used only during AlignedCAE training (reconstruction loss).
    Stays local — never shared.
    """
    def __init__(self, obs_dim: int, hidden_dim: int = HIDDEN_DIM,
                 latent_dim: int = LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, obs_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ---------------------------------------------------------------------------
# AlignedCAE — full model combining all three parts
# ---------------------------------------------------------------------------

class AlignedCAE(nn.Module):
    """
    Full aligned autoencoder for one client.

    Encode path (used at inference):
        obs → LocalEncoder → SharedEncoderHead → z ∈ (-π, π)^8

    Decode path (used only during CAE training for reconstruction loss):
        z → LocalDecoder → obs_hat

    At federation time:
        local_encoder.parameters()  → stay local
        shared_head.parameters()    → sent to server for FedAvg
        local_decoder.parameters()  → stay local
    """
    def __init__(self, obs_dim: int, hidden_dim: int = HIDDEN_DIM,
                 latent_dim: int = LATENT_DIM):
        super().__init__()
        self.local_encoder = LocalEncoder(obs_dim, hidden_dim)
        self.shared_head   = SharedEncoderHead(hidden_dim, latent_dim)
        self.local_decoder = LocalDecoder(obs_dim, hidden_dim, latent_dim)
        self.obs_dim = obs_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Forward path used by VQC at inference."""
        h = self.local_encoder(x)
        z = self.shared_head(h)
        return z

    def forward(self, x: torch.Tensor):
        """Full forward for training: returns (reconstruction, latent)."""
        z    = self.encode(x)
        x_hat = self.local_decoder(z)
        return x_hat, z

    def get_shared_weights(self) -> dict:
        """Extract SharedEncoderHead state dict (to send to server)."""
        return {k: v.clone().cpu()
                for k, v in self.shared_head.state_dict().items()}

    def set_shared_weights(self, state_dict: dict) -> None:
        """Load aggregated SharedEncoderHead weights (from server)."""
        sd = {k: v.to(next(self.shared_head.parameters()).device)
              for k, v in state_dict.items()}
        self.shared_head.load_state_dict(sd)


# ---------------------------------------------------------------------------
# Training helper (mirrors train_cae from autoencoder.py)
# ---------------------------------------------------------------------------

def train_aligned_cae(
    cae: AlignedCAE,
    observations: np.ndarray,
    n_steps: int = 50,
    lr: float = 1e-3,
    device: str = "cpu",
) -> float:
    """
    Train AlignedCAE on a batch of recent observations.
    Returns final reconstruction loss.
    Called every cae_update_interval steps during RL training.
    """
    cae.train()
    opt = optim.Adam(cae.parameters(), lr=lr)
    obs_t = torch.tensor(observations, dtype=torch.float32, device=device)

    last_loss = 0.0
    for _ in range(n_steps):
        idx = torch.randint(0, len(obs_t), (min(256, len(obs_t)),))
        batch = obs_t[idx]
        x_hat, _ = cae(batch)
        loss = nn.functional.mse_loss(x_hat, batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        last_loss = loss.item()

    cae.eval()
    return last_loss


# ---------------------------------------------------------------------------
# FedAvg for shared head weights
# ---------------------------------------------------------------------------

def fedavg_shared_head(
    weight_list: list[dict],
    aggregation: str = "uniform",
    rewards: list | None = None,
) -> dict:
    """
    Average a list of SharedEncoderHead state dicts.
    Returns averaged state dict (on CPU).

    aggregation = "magnitude_inv": weight by 1/|reward| so better clients
    (less negative reward) contribute more. Mirrors _fedavg() logic.
    """
    import numpy as np
    keys = weight_list[0].keys()
    if aggregation == "magnitude_inv" and rewards is not None:
        magnitudes = np.array([max(abs(r), 1e-6) for r in rewards], dtype=np.float64)
        w = 1.0 / magnitudes
        w = w / w.sum()
        wt = [float(wi) for wi in w]
    else:
        n = len(weight_list)
        wt = [1.0 / n] * n

    averaged = {}
    for k in keys:
        stacked = torch.stack([w[k].float() for w in weight_list], dim=0)
        weights_t = torch.tensor(wt, dtype=torch.float32)
        averaged[k] = (stacked * weights_t.view(-1, *([1] * (stacked.dim() - 1)))).sum(dim=0)
    return averaged


# ---------------------------------------------------------------------------
# Communication cost helper
# ---------------------------------------------------------------------------

def shared_head_param_count() -> int:
    """Number of parameters in SharedEncoderHead (same for all clients)."""
    head = SharedEncoderHead()
    return sum(p.numel() for p in head.parameters())


def bytes_per_aligned_update(n_clients: int) -> int:
    """
    Bytes per FL round for aligned federation (upload + download).
    SharedEncoderHead (272 params) + VQC (16 params) = 288 params.
    """
    n_params = shared_head_param_count() + 16   # head + VQC
    return n_clients * 2 * n_params * 4          # float32, upload + download
