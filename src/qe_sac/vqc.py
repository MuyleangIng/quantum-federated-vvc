"""
Variational Quantum Circuit (VQC) for QE-SAC.

Architecture (per paper):
    State preparation : RY(s'_i) on qubit i  (angle encoding of 8-dim latent)
    Variational layer : CNOT(i, i+1) nearest-neighbour entanglement
                        + RX(ζ_k) trainable rotation on each qubit
    Repeated for L=2 layers.
    Measurement       : <Z_i> expectation on each qubit → 8-dim output vector

Implementation:
    Pure-PyTorch statevector simulator — identical math to PennyLane but
    processes the entire batch in one tensor operation on GPU.
    Speed: ~5-10ms for 64 samples (vs 342ms sequential PennyLane).

Noise evaluation:
    The PennyLane noisy path (default.mixed + parameter-shift) is kept
    inside VQCLayer and used only by evaluate_noise_robustness().
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pennylane as qml


N_QUBITS = 8
N_LAYERS = 2
DIM = 2 ** N_QUBITS  # 256


# ---------------------------------------------------------------------------
# Gate matrix builders
# ---------------------------------------------------------------------------

def _ry_matrix(theta: torch.Tensor) -> torch.Tensor:
    """RY(θ) matrices.  theta: (...,) → (..., 2, 2) real."""
    c = torch.cos(theta / 2)
    s = torch.sin(theta / 2)
    return torch.stack([
        torch.stack([ c, -s], dim=-1),
        torch.stack([ s,  c], dim=-1),
    ], dim=-2)


def _rx_matrix(theta: torch.Tensor) -> torch.Tensor:
    """RX(θ) matrices.  theta: (...,) → (..., 2, 2) complex64."""
    c = torch.cos(theta / 2).to(torch.complex64)
    s = torch.sin(theta / 2).to(torch.complex64)
    j = torch.tensor(1j, dtype=torch.complex64, device=theta.device)
    return torch.stack([
        torch.stack([ c,     -j * s], dim=-1),
        torch.stack([-j * s,  c    ], dim=-1),
    ], dim=-2)


# ---------------------------------------------------------------------------
# Statevector operations
# ---------------------------------------------------------------------------

def _apply_single_qubit_gate(
    psi: torch.Tensor,
    gate: torch.Tensor,
    qubit: int,
) -> torch.Tensor:
    """
    Apply a single-qubit gate to qubit `qubit` of a batched statevector.

    psi  : (B, 2^N_QUBITS) complex64
    gate : (2, 2) complex64   — same gate for all samples
        or (B, 2, 2) complex64 — per-sample gate
    """
    B = psi.shape[0]
    # Reshape to (B, 2, 2, ..., 2) — one axis per qubit
    psi_t = psi.view(B, *([2] * N_QUBITS))
    # Move the target qubit axis to the last position
    ax = qubit + 1          # +1 because axis 0 is batch
    psi_t = psi_t.moveaxis(ax, -1)    # (..., 2)

    # Flatten all non-batch, non-target dims: (B, S, 2)
    psi_flat = psi_t.reshape(B, -1, 2)
    gate_c = gate.to(torch.complex64)

    # Matrix multiply: result[b,s,i] = Σ_j gate[i,j] * psi[b,s,j]
    # Equivalent to  psi_flat @ gate.T  (for 2-D gate, broadcasts over B,S)
    #                psi_flat @ gate.mT (for 3-D gate, batched over B)
    new_flat = psi_flat @ gate_c.mT   # (B, S, 2)

    # Restore shape and move axis back
    new_t = new_flat.reshape(psi_t.shape)
    new_t = new_t.moveaxis(-1, ax)
    return new_t.reshape(B, DIM)


def _apply_cnot(psi: torch.Tensor, control: int, target: int) -> torch.Tensor:
    """
    Apply CNOT(control, target) to a batch of statevectors.

    psi : (B, 2^N_QUBITS) complex64
    """
    B = psi.shape[0]
    psi_t = psi.view(B, *([2] * N_QUBITS)).clone()

    ctrl_ax   = control + 1
    target_ax = target  + 1

    # When control=1, swap target amplitudes (X gate on target)
    idx_c1_t0 = [slice(None)] * (N_QUBITS + 1)
    idx_c1_t1 = [slice(None)] * (N_QUBITS + 1)
    idx_c1_t0[ctrl_ax] = 1;  idx_c1_t0[target_ax] = 0
    idx_c1_t1[ctrl_ax] = 1;  idx_c1_t1[target_ax] = 1
    idx_c1_t0 = tuple(idx_c1_t0)
    idx_c1_t1 = tuple(idx_c1_t1)

    tmp = psi_t[idx_c1_t0].clone()
    psi_t[idx_c1_t0] = psi_t[idx_c1_t1]
    psi_t[idx_c1_t1] = tmp

    return psi_t.reshape(B, DIM)


def _pauli_z_expectation(psi: torch.Tensor, qubit: int) -> torch.Tensor:
    """
    <Z_qubit> = prob(qubit=0) − prob(qubit=1).

    psi : (B, 2^N_QUBITS) complex64
    Returns : (B,) float32
    """
    B = psi.shape[0]
    probs = psi.abs().pow(2).view(B, *([2] * N_QUBITS))
    ax = qubit + 1
    probs = probs.moveaxis(ax, 1)          # (B, 2, ...)
    probs = probs.reshape(B, 2, -1).sum(-1)  # (B, 2)
    return (probs[:, 0] - probs[:, 1]).float()


# ---------------------------------------------------------------------------
# Batched VQC forward (pure PyTorch)
# ---------------------------------------------------------------------------

def _vqc_forward(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Batched VQC forward pass.

    inputs  : (B, N_QUBITS) float — angle-encoded latent
    weights : (N_LAYERS, N_QUBITS) float — trainable RX angles
    Returns : (B, N_QUBITS) float — PauliZ expectations in [−1, 1]
    """
    B = inputs.shape[0]
    dev = inputs.device

    # |0...0⟩ initial state
    psi = torch.zeros(B, DIM, dtype=torch.complex64, device=dev)
    psi[:, 0] = 1.0

    # State preparation: RY(inputs[i]) on qubit i
    # ry_gates: (B, N_QUBITS, 2, 2) — per-sample, per-qubit gates
    ry_gates = _ry_matrix(inputs).to(torch.complex64)   # (B, N_QUBITS, 2, 2)
    for i in range(N_QUBITS):
        psi = _apply_single_qubit_gate(psi, ry_gates[:, i], i)

    # Variational layers
    rx_gates = _rx_matrix(weights)    # (N_LAYERS, N_QUBITS, 2, 2)
    for layer in range(N_LAYERS):
        for i in range(N_QUBITS - 1):
            psi = _apply_cnot(psi, i, i + 1)
        for i in range(N_QUBITS):
            # rx_gates[layer, i] is (2, 2) — same gate for all batch samples
            psi = _apply_single_qubit_gate(psi, rx_gates[layer, i], i)

    # Measure <Z_i> for each qubit
    return torch.stack([_pauli_z_expectation(psi, i) for i in range(N_QUBITS)], dim=1)


# ---------------------------------------------------------------------------
# VQCLayer — nn.Module
# ---------------------------------------------------------------------------

class VQCLayer(nn.Module):
    """
    PyTorch nn.Module wrapping the VQC.

    Fast path  : PennyLane lightning.gpu (GPU-accelerated statevector, adjoint diff).
    Fallback   : PennyLane lightning.qubit (CPU C++, if GPU unavailable).
    Noisy path : PennyLane default.mixed  (depolarising noise, parameter-shift).

    Parameters
    ----------
    noise_lambda : depolarising noise probability per gate (0 = noiseless)
    """

    def __init__(self, noise_lambda: float = 0.0):
        super().__init__()
        self._noise = noise_lambda

        # Trainable weights: (N_LAYERS, N_QUBITS) = 16 parameters
        self.weights = nn.Parameter(
            torch.randn(N_LAYERS, N_QUBITS) * 0.1
        )

        # PennyLane circuits — built lazily on first forward call
        self._pl_circuit       = None   # noiseless: lightning.gpu / lightning.qubit
        self._pl_noisy_circuit = None   # noisy: default.mixed

    def _get_pl_circuit(self):
        """Build noiseless PennyLane circuit on lightning.gpu (falls back to lightning.qubit)."""
        if self._pl_circuit is not None:
            return self._pl_circuit

        try:
            dev = qml.device("lightning.gpu", wires=N_QUBITS)
            diff_method = "adjoint"
        except Exception:
            dev = qml.device("lightning.qubit", wires=N_QUBITS)
            diff_method = "adjoint"

        @qml.qnode(dev, diff_method=diff_method, interface="torch")
        def circuit(inputs, weights):
            # State preparation: RY angle encoding
            for i in range(N_QUBITS):
                qml.RY(inputs[i], wires=i)
            # Variational layers: CNOT entanglement + RX rotations
            for layer in range(N_LAYERS):
                for i in range(N_QUBITS - 1):
                    qml.CNOT(wires=[i, i + 1])
                for i in range(N_QUBITS):
                    qml.RX(weights[layer, i], wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

        self._pl_circuit = circuit
        return circuit

    def _get_pl_noisy_circuit(self):
        """Build noisy PennyLane circuit on default.mixed (parameter-shift)."""
        if self._pl_noisy_circuit is not None:
            return self._pl_noisy_circuit
        dev = qml.device("default.mixed", wires=N_QUBITS)
        lam = self._noise

        @qml.qnode(dev, diff_method="parameter-shift", interface="torch")
        def circuit(inputs, weights):
            for i in range(N_QUBITS):
                qml.RY(inputs[i], wires=i)
                qml.DepolarizingChannel(lam, wires=i)
            for layer in range(N_LAYERS):
                for i in range(N_QUBITS - 1):
                    qml.CNOT(wires=[i, i + 1])
                    qml.DepolarizingChannel(lam, wires=i)
                    qml.DepolarizingChannel(lam, wires=i + 1)
                for i in range(N_QUBITS):
                    qml.RX(weights[layer, i], wires=i)
                    qml.DepolarizingChannel(lam, wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

        self._pl_noisy_circuit = circuit
        return circuit

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : shape (..., N_QUBITS) — latent vectors from CAE encoder

        Returns
        -------
        out : shape (..., N_QUBITS) — PauliZ expectations in [−1, 1]
        """
        if self._noise > 0.0:
            # Noisy path: default.mixed + parameter-shift (sequential per sample)
            circuit = self._get_pl_noisy_circuit()
            if inputs.dim() == 1:
                return torch.stack(circuit(inputs, self.weights)).float()
            return torch.stack([
                torch.stack(circuit(x, self.weights))
                for x in inputs
            ]).float()

        # Fast path: lightning.gpu (adjoint diff, GPU-accelerated statevector)
        circuit = self._get_pl_circuit()
        if inputs.dim() == 1:
            return torch.stack(circuit(inputs, self.weights)).float()
        return torch.stack([
            torch.stack(circuit(x, self.weights))
            for x in inputs
        ]).float()

    @property
    def n_params(self) -> int:
        return self.weights.numel()


# ---------------------------------------------------------------------------
# Configurable VQC for ablation study (Task 4.6)
# ---------------------------------------------------------------------------

class VQCLayerAblation(nn.Module):
    """
    Configurable VQC for Task 4.6 qubit/layer ablation study.

    Same circuit design as VQCLayer but accepts n_qubits and n_layers
    as constructor arguments. Used ONLY for ablation — main experiments
    always use VQCLayer (8 qubits, 2 layers) for fair paper comparison.

    Parameters
    ----------
    n_qubits  : number of qubits (and encoder output dim must match)
    n_layers  : number of variational layers
    noise_lambda : depolarising noise (0 = noiseless)
    """

    def __init__(
        self,
        n_qubits:     int   = N_QUBITS,
        n_layers:     int   = N_LAYERS,
        noise_lambda: float = 0.0,
    ):
        super().__init__()
        self._nq = n_qubits
        self._nl = n_layers
        self._dim = 2 ** n_qubits
        self._noise = noise_lambda

        self.weights = nn.Parameter(
            torch.randn(n_layers, n_qubits) * 0.1
        )

    def _forward_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        """Pure-PyTorch statevector forward for arbitrary n_qubits/n_layers."""
        B = inputs.shape[0]
        nq = self._nq
        dim = self._dim
        dev = inputs.device

        # |0...0⟩
        psi = torch.zeros(B, dim, dtype=torch.complex64, device=dev)
        psi[:, 0] = 1.0

        def apply_single(psi, gate, qubit):
            psi_t = psi.view(B, *([2] * nq))
            ax = qubit + 1
            psi_t = psi_t.moveaxis(ax, -1)
            psi_flat = psi_t.reshape(B, -1, 2)
            new_flat = psi_flat @ gate.to(torch.complex64).mT
            new_t = new_flat.reshape(psi_t.shape)
            new_t = new_t.moveaxis(-1, ax)
            return new_t.reshape(B, dim)

        def apply_cnot(psi, control, target):
            psi_t = psi.view(B, *([2] * nq)).clone()
            ca, ta = control + 1, target + 1
            idx_c1_t0 = [slice(None)] * (nq + 1)
            idx_c1_t1 = [slice(None)] * (nq + 1)
            idx_c1_t0[ca] = 1; idx_c1_t0[ta] = 0
            idx_c1_t1[ca] = 1; idx_c1_t1[ta] = 1
            tmp = psi_t[tuple(idx_c1_t0)].clone()
            psi_t[tuple(idx_c1_t0)] = psi_t[tuple(idx_c1_t1)]
            psi_t[tuple(idx_c1_t1)] = tmp
            return psi_t.reshape(B, dim)

        def pauli_z_exp(psi, qubit):
            probs = psi.abs().pow(2).view(B, *([2] * nq))
            ax = qubit + 1
            probs = probs.moveaxis(ax, 1).reshape(B, 2, -1).sum(-1)
            return (probs[:, 0] - probs[:, 1]).float()

        # State prep
        ry = _ry_matrix(inputs).to(torch.complex64)   # (B, nq, 2, 2)
        for i in range(nq):
            psi = apply_single(psi, ry[:, i], i)

        # Variational layers
        rx = _rx_matrix(self.weights)   # (nl, nq, 2, 2)
        for layer in range(self._nl):
            for i in range(nq - 1):
                psi = apply_cnot(psi, i, i + 1)
            for i in range(nq):
                psi = apply_single(psi, rx[layer, i], i)

        return torch.stack([pauli_z_exp(psi, i) for i in range(nq)], dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() == 1:
            return self._forward_batch(inputs.unsqueeze(0)).squeeze(0)
        return self._forward_batch(inputs)

    @property
    def n_params(self) -> int:
        return self.weights.numel()
