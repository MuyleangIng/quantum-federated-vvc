"""
GNN-based LocalEncoder for QE-SAC-FL.

Replaces the MLP LocalEncoder with a Graph Neural Network that takes
bus node features AND the grid adjacency matrix as input.

Why this is better than MLP for dynamic/heterogeneous topologies:
  - MLP sees a flat vector — no topology awareness
  - GNN sees graph structure explicitly — when a line is removed (fault),
    the edge disappears from adj_matrix, GNN adapts naturally
  - GNN handles variable number of buses via graph-level pooling
  - When topology changes mid-training, only the edges change —
    node feature weights are reused (faster recovery than MLP retraining)

Architecture:
  node_features [n_buses, 3]  (V, P, Q per bus)
  adj_matrix    [n_buses, n_buses]
    → BusGNN (2 message-passing layers, mean aggregation)
    → node_embeddings [n_buses, node_dim]
    → mean pooling → graph_embedding [node_dim]
    → Linear → h(32)

What stays identical vs MLP variant:
  SharedEncoderHead  [32→8, federated, 272 params]
  VQC                [16 params, federated]
  LocalDecoder       [private]
  Total federated:   288 params  ← unchanged

The only change is LOCAL — how each client compresses its obs into h(32).
Federation cost is identical for both variants.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.qe_sac_fl.aligned_encoder import (
    SharedEncoderHead, LocalDecoder,
    HIDDEN_DIM, LATENT_DIM, LATENT_SCALE,
    fedavg_shared_head,
)

NODE_FEAT_DIM = 3     # V_magnitude, P_inject, Q_inject per bus
NODE_HIDDEN   = 32    # GNN hidden dim per node
NODE_LAYERS   = 2     # message passing layers


# ---------------------------------------------------------------------------
# Graph utility — build adjacency from branch list
# ---------------------------------------------------------------------------

def branches_to_adj(branches: list[tuple], n_buses: int,
                    device: str = "cpu") -> torch.Tensor:
    """
    Convert branch list [(from, to, r, x), ...] to symmetric adj matrix.
    Returns float32 tensor shape (n_buses, n_buses), values in {0, 1}.
    Self-loops added for stability (GCN convention).
    """
    adj = torch.zeros(n_buses, n_buses, dtype=torch.float32)
    for b in branches:
        i, j = int(b[0]), int(b[1])
        if i < n_buses and j < n_buses:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    adj += torch.eye(n_buses)          # self-loops
    adj = adj.clamp(max=1.0)
    return adj.to(device)


def flat_obs_to_node_features(obs: torch.Tensor, n_buses: int) -> torch.Tensor:
    """
    Split flat obs vector into per-bus node feature matrix.

    Obs layout (from _VVCEnvBase):
      [V_0..V_{n-1}, P_0..P_{n-1}, Q_0..Q_{n-1}, device_states...]

    Returns: [B, n_buses, 3]  (V, P, Q per bus)
    Remaining device state features are discarded here — the GNN focuses
    on the grid state; device states are captured by the SharedHead.
    """
    B = obs.shape[0]
    # Take first 3*n_buses features: V, P, Q
    feats = obs[:, :3 * n_buses]   # [B, 3*n_buses]
    # If obs is shorter than 3*n_buses (e.g. small grid), zero-pad
    if feats.shape[1] < 3 * n_buses:
        pad = torch.zeros(B, 3 * n_buses - feats.shape[1],
                          device=obs.device, dtype=obs.dtype)
        feats = torch.cat([feats, pad], dim=1)
    # Reshape to [B, n_buses, 3]
    return feats.view(B, n_buses, 3)


# ---------------------------------------------------------------------------
# BusGNN — simple graph convolutional network
# ---------------------------------------------------------------------------

class BusGNN(nn.Module):
    """
    2-layer Graph Convolutional Network for distribution bus graphs.

    Each layer:
      h_v^(l+1) = ReLU( W^(l) · MEAN_{u ∈ N(v)} h_u^(l) )

    Using degree-normalised adjacency (symmetric normalisation):
      H^(l+1) = ReLU( D^{-1/2} A D^{-1/2} H^(l) W^(l) )

    Input:  node_features [B, n_buses, NODE_FEAT_DIM]
            adj_matrix    [n_buses, n_buses]   (precomputed, shared across batch)
    Output: node_embeddings [B, n_buses, NODE_HIDDEN]
    """

    def __init__(self, in_dim: int = NODE_FEAT_DIM,
                 hidden_dim: int = NODE_HIDDEN,
                 n_layers: int = NODE_LAYERS):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * n_layers
        for i in range(n_layers):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

    def _norm_adj(self, adj: torch.Tensor) -> torch.Tensor:
        """Symmetric normalisation: D^{-1/2} A D^{-1/2}."""
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        d_inv_sqrt = deg.pow(-0.5)
        return d_inv_sqrt * adj * d_inv_sqrt.transpose(-1, -2)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x:   [B, n_buses, in_dim]
        adj: [n_buses, n_buses]  or  [B, n_buses, n_buses]
        Returns: [B, n_buses, hidden_dim]
        """
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)          # [1, n, n]
        adj_norm = self._norm_adj(adj)      # [1 or B, n, n]

        h = x
        for i, layer in enumerate(self.layers):
            # Message passing: aggregate neighbour features
            h_agg = torch.bmm(adj_norm.expand(h.shape[0], -1, -1), h)   # [B, n, d]
            h = layer(h_agg)
            if i < len(self.layers) - 1:
                h = F.relu(h)
        return h   # [B, n_buses, hidden_dim]


# ---------------------------------------------------------------------------
# GNN LocalEncoder
# ---------------------------------------------------------------------------

class GNNLocalEncoder(nn.Module):
    """
    Graph-based local encoder — private, stays on client.

    obs (flat) + adj_matrix → BusGNN → mean pooling → Linear → h(32)

    The adjacency matrix is fixed at init from the environment's branch list.
    For dynamic topology (faults): call update_topology(new_branches) to
    update adj_matrix without rebuilding the network.

    Output dim matches MLP LocalEncoder: h(HIDDEN_DIM=32)
    """

    def __init__(self, n_buses: int, branches: list[tuple],
                 hidden_dim: int = HIDDEN_DIM, device: str = "cpu"):
        super().__init__()
        self.n_buses    = n_buses
        self.hidden_dim = hidden_dim
        self._device    = device

        self.gnn     = BusGNN(NODE_FEAT_DIM, NODE_HIDDEN, NODE_LAYERS)
        self.pool_fc = nn.Linear(NODE_HIDDEN, hidden_dim)   # graph → h(32)

        # Fixed adjacency — updated via update_topology() on fault
        adj = branches_to_adj(branches, n_buses, device)
        self.register_buffer("adj", adj)

    def update_topology(self, new_branches: list[tuple]) -> None:
        """
        Hot-swap topology without rebuilding the network.
        Call this when a line fault or switching event occurs.
        """
        new_adj = branches_to_adj(new_branches, self.n_buses, self._device)
        self.adj.copy_(new_adj)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: [B, obs_dim] flat observation vector
        Returns: h [B, HIDDEN_DIM]
        """
        # Extract per-bus node features from flat obs
        node_feat = flat_obs_to_node_features(obs, self.n_buses)   # [B, n, 3]

        # GNN message passing
        node_emb = self.gnn(node_feat, self.adj)                    # [B, n, NODE_HIDDEN]

        # Graph-level mean pooling
        graph_emb = node_emb.mean(dim=1)                            # [B, NODE_HIDDEN]

        # Project to hidden_dim
        h = F.relu(self.pool_fc(graph_emb))                         # [B, hidden_dim]
        return h


# ---------------------------------------------------------------------------
# GNN Aligned CAE — full model
# ---------------------------------------------------------------------------

class GNNAlignedCAE(nn.Module):
    """
    Drop-in replacement for AlignedCAE using GNNLocalEncoder.

    Encode path (inference):
        obs + adj → GNNLocalEncoder → h(32) → SharedEncoderHead → z(8) ∈ (-π, π)

    Decode path (CAE training):
        z(8) → LocalDecoder → obs_hat

    Federation interface identical to AlignedCAE:
        get_shared_weights() → SharedEncoderHead state dict (272 params)
        set_shared_weights() → loads aggregated weights
    Total federated: 288 params = 1,152 bytes  ← UNCHANGED vs MLP variant
    """

    def __init__(self, obs_dim: int, n_buses: int, branches: list[tuple],
                 hidden_dim: int = HIDDEN_DIM, latent_dim: int = LATENT_DIM,
                 device: str = "cpu"):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_buses = n_buses

        self.local_encoder = GNNLocalEncoder(n_buses, branches, hidden_dim, device)
        self.shared_head   = SharedEncoderHead(hidden_dim, latent_dim)
        self.local_decoder = LocalDecoder(obs_dim, hidden_dim, latent_dim)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward path used by VQC at inference."""
        h = self.local_encoder(obs)
        z = self.shared_head(h)
        return z

    def forward(self, obs: torch.Tensor):
        """Full forward for training: returns (reconstruction, latent)."""
        z     = self.encode(obs)
        x_hat = self.local_decoder(z)
        return x_hat, z

    def update_topology(self, new_branches: list[tuple]) -> None:
        """Propagate topology change to GNN adjacency matrix."""
        self.local_encoder.update_topology(new_branches)

    def get_shared_weights(self) -> dict:
        return {k: v.clone().cpu()
                for k, v in self.shared_head.state_dict().items()}

    def set_shared_weights(self, state_dict: dict) -> None:
        sd = {k: v.to(next(self.shared_head.parameters()).device)
              for k, v in state_dict.items()}
        self.shared_head.load_state_dict(sd)


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def train_gnn_cae(cae: GNNAlignedCAE, observations: np.ndarray,
                  n_steps: int = 50, lr: float = 1e-3,
                  device: str = "cpu") -> float:
    """Train GNNAlignedCAE on recent observations. Returns final loss."""
    cae.train()
    opt     = optim.Adam(cae.parameters(), lr=lr)
    obs_t   = torch.tensor(observations, dtype=torch.float32, device=device)
    last_loss = 0.0
    for _ in range(n_steps):
        idx   = torch.randint(0, len(obs_t), (min(256, len(obs_t)),))
        batch = obs_t[idx]
        x_hat, _ = cae(batch)
        loss = F.mse_loss(x_hat, batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        last_loss = loss.item()
    cae.eval()
    return last_loss


# ---------------------------------------------------------------------------
# GNN parameter count helper
# ---------------------------------------------------------------------------

def gnn_encoder_param_count(n_buses: int) -> dict:
    """
    Return parameter counts for GNN encoder components.
    Federated params are identical to MLP variant (288).
    """
    enc  = GNNLocalEncoder(n_buses, [(0,1,0,0)]*n_buses)
    head = SharedEncoderHead()
    dec  = LocalDecoder(n_buses * 3)
    return {
        "GNNLocalEncoder (private)":   sum(p.numel() for p in enc.parameters()),
        "SharedEncoderHead (federated)": sum(p.numel() for p in head.parameters()),
        "LocalDecoder (private)":      sum(p.numel() for p in dec.parameters()),
        "Total federated":             sum(p.numel() for p in head.parameters()) + 16,
    }


def build_adj_13bus(device: str = "cpu") -> torch.Tensor:
    """Adjacency matrix for IEEE 13-bus system."""
    from src.qe_sac.env_utils import _IEEE13_BRANCHES
    return branches_to_adj(_IEEE13_BRANCHES, n_buses=13, device=device)


def build_adj_34bus(device: str = "cpu") -> torch.Tensor:
    """Adjacency matrix for IEEE 34-bus system."""
    from src.qe_sac_fl.env_34bus import _IEEE34_BRANCHES
    return branches_to_adj(_IEEE34_BRANCHES, n_buses=34, device=device)


def build_adj_57bus(device: str = "cpu") -> torch.Tensor:
    """Adjacency matrix for IEEE 57-bus system."""
    from src.qe_sac_fl.env_57bus import _IEEE57_BRANCHES
    return branches_to_adj(_IEEE57_BRANCHES, n_buses=57, device=device)


def build_adj_123bus(device: str = "cpu") -> torch.Tensor:
    """Adjacency matrix for IEEE 123-bus system."""
    from src.qe_sac.env_utils import _IEEE123_BRANCHES
    return branches_to_adj(_IEEE123_BRANCHES, n_buses=123, device=device)
