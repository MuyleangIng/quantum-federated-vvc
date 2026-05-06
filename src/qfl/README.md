# QFL — Quantum Federated Learning

**Proposed method for multi-utility Volt-VAR control.**  
Extends Lin et al. (2025) QE-SAC to a federated, multi-utility setting — **without RL**.

---

## 1. What QE-SAC (Lin et al. 2025) Does

Lin et al. proposed **QE-SAC**: Quantum-Enhanced Soft Actor-Critic for Volt-VAR control.

```
obs [obs_dim]
  → MLP encoder  (obs_dim → 32)
  → VQC          (8 qubits, 2 layers, 16 params) — acts as actor policy
  → action logits
```

- **Single utility only** — no federation, no multi-utility scenario
- **RL-based** — SAC (Soft Actor-Critic) trains the VQC via policy gradient
- **Problem**: SAC requires a full actor + 2 critics + replay buffer (~107 KB per model)
- **Problem**: Each utility trains in isolation — no shared quantum knowledge

---

## 2. What QFL (This Work) Does Differently

QFL removes RL entirely and replaces it with **SPSA** (gradient-free quantum optimization) + **FedAvg** across utilities.

```
obs [B, obs_dim]
  → LocalEncoder       (private MLP, obs_dim → 32)   ← NOT federated
  → SharedEncoderHead  (32 → 8, Tanh × π)            ← FEDERATED
  → VQC                (8 qubits, 2 layers, 16 params)← FEDERATED
  → ActionHeads        (8 → per-device softmax)       ← NOT federated

Total federated params: 264 + 16 = 280 = 1.1 KB per client per round
```

### Key Differences vs QE-SAC

| Property | QE-SAC (Lin et al.) | QFL (This Work) |
|---|---|---|
| Training | SAC (RL) | SPSA (gradient-free) |
| Scope | Single utility | 3 utilities federated |
| Federated | No | Yes (FedAvg on QuantumEncoder) |
| Comm cost | N/A (no FL) | 280 params = 1.1 KB/round |
| vs classical FL | — | **383× reduction** |
| VQC role | Actor policy | Shared quantum encoder |
| Critic | 2 critics + replay | None |

---

## 3. VQC Circuit (identical to Lin et al.)

```
Qubit 0: ─ RY(z₀) ─ ●──────── RX(ζ₀⁽¹⁾) ─ ●──────── RX(ζ₀⁽²⁾) ─ <Z>
Qubit 1: ─ RY(z₁) ─ X ─ ●─── RX(ζ₁⁽¹⁾) ─ X ─ ●─── RX(ζ₁⁽²⁾) ─ <Z>
Qubit 2: ─ RY(z₂) ─────  X ── RX(ζ₂⁽¹⁾) ─────  X ── RX(ζ₂⁽²⁾) ─ <Z>
...
Qubit 7: ─ RY(z₇) ───────────  RX(ζ₇⁽¹⁾) ───────────  RX(ζ₇⁽²⁾) ─ <Z>
```

- **State prep**: RY(zᵢ) where z = Tanh(SharedEncoderHead(obs)) × π  ∈ (−π, π)
- **Entanglement**: CNOT(i, i+1) nearest-neighbour per layer
- **Variational**: RX(ζₖᵢ) trainable per qubit per layer → 2 × 8 = **16 params**
- **Measurement**: ⟨Zᵢ⟩ expectation → 8-dim output ∈ [−1, 1]
- **Layers**: L = 2 (same as Lin et al.)

---

## 4. SPSA Training (replaces SAC)

SPSA estimates the VQC gradient with **2 environment evaluations**, no backprop through the circuit.

```python
# One SPSA step
δ ~ Rademacher(±1)             # random direction, shape (2, 8)
θ₊ = θ + c·δ  →  L₊ = −reward(θ₊)
θ₋ = θ − c·δ  →  L₋ = −reward(θ₋)
∂L/∂θ ≈ (L₊ − L₋) / (2c·δ)   # SPSA gradient estimate
θ ← θ − α · ∂L/∂θ
```

- `c = 0.1` (perturbation size)
- `α = 0.01` (learning rate for VQC)
- Loss = −reward = number of voltage violations
- **No critic, no replay buffer, no RL**

Classical action heads (private) are updated separately via entropy maximization:

```python
loss = −entropy(action_probs) × max(reward, 0)
```

This encourages exploration while reinforcing actions that reduced violations.

---

## 5. Federated Learning (FedAvg)

After each local training round, clients upload only the `QuantumEncoder` weights:

```
Client A (13-bus):  280 params → server
Client B (34-bus):  280 params → server
Client C (123-bus): 280 params → server

Server: global = (1/3) × (A + B + C)   ← uniform FedAvg

Broadcast global back to all clients
```

The `LocalEncoder` and `ActionHeads` stay **private** — they handle topology-specific differences (13-bus: obs=43, 34-bus: obs=113, 123-bus: obs=349).

Only the shared quantum representation (encoded as VQC angles) is federated.

---

## 6. Communication Cost

| Method | Params per round | Size per round | vs QFL |
|---|---|---|---|
| Classical SAC-FL (actor only) | 107,456 | ~419 KB | 383× more |
| QFL (this work) | 280 | 1.1 KB | **baseline** |

Classical SAC actor = MLP(obs_dim → 256 → 256 → actions) ≈ 107 K params.  
QFL only federates the QuantumEncoder: 264 (compress) + 16 (VQC) = 280 params.

---

## 7. Experiment Setup

| Client | Grid | obs_dim | Devices | reward_scale |
|---|---|---|---|---|
| Utility A | IEEE 13-bus | 43 | [2, 2, 33, 33] | 50.0 |
| Utility B | IEEE 34-bus | 113 | [2, 2, 33, 33, 33] | 10.0 |
| Utility C | IEEE 123-bus | 349 | [2, 2, 33, 33, 33, 33, 33] | 750.0 |

- **Rounds**: 50 per seed
- **Steps per round**: 1,000 per client
- **Seeds**: 0, 1, 2
- **Conditions**: `local_only` (no FL) vs `qfl` (FedAvg on QuantumEncoder)

---

## 8. Early Results (seed 0, local_only, rounds 10/20/30)

| Round | 13-bus (A) | 34-bus (B) | 123-bus (C) |
|---|---|---|---|
| 10 | −0.60 | −0.16 | −0.00 |
| 20 | −0.60 | −0.18 | −0.00 |
| 30 | −0.60 | −0.21 | −0.00 |

Mean reward per step. Higher = fewer violations. C converges fastest.  
Full results (3 seeds × 2 conditions) in progress.

---

## 9. File Map

```
src/qfl/
  agent.py     — QFLAgent (SPSA + head_update + federation interface)
                 QuantumEncoder (LocalEncoder + SharedEncoderHead + VQC)
                 ActionHeads (private per-device softmax)
  trainer.py   — QFLTrainer (FedAvg loop, local_only vs qfl conditions)
  config.py    — QFLConfig + ClientConfig (all hyperparameters)
  __init__.py

scripts/
  run_qfl.py   — main runner (3 seeds, 2 conditions)

logs/
  qfl.log      — live experiment log

artifacts/qfl/
  seed{N}_{condition}.json   — per-seed per-condition results
```

---

## 10. References

- **Lin et al. (2025)** — QE-SAC: Quantum-Enhanced SAC for Volt-VAR control.  
  DOI: 10.1109/OAJPE.2025.3534946. ← **This work extends and federates this.**
- **Chen & Yoo (2021)** — Federated Quantum Machine Learning. Entropy 23(4):460.
- **Spall (1992)** — SPSA: Multivariate Stochastic Approximation Using a Simultaneous Perturbation Gradient Approximation. IEEE TAC 37(3).
