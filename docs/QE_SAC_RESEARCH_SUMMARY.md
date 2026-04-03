# QE-SAC Research Summary
## Quantum RL for Volt-VAR Control — Full Picture

---

## 1. Why This Problem Exists

Power distribution grids are becoming harder to control. Solar panels and
batteries are added everywhere, making voltage fluctuate constantly.
Grid operators must control three types of devices in real-time:

- **Capacitor banks** — inject reactive power → raise voltage (ON/OFF)
- **Voltage regulators** — adjust tap position → fine-tune voltage (33 taps)
- **Batteries** — charge/discharge → balance load (33 levels)

**Goal:** Keep every bus voltage inside [0.95, 1.05] pu at all times,
while minimising switching costs and power losses.
This is called **Volt-VAR Control (VVC)**.

**Why it is hard:**
- Decisions must happen in real-time (every few minutes)
- Grid state is high-dimensional and physically non-linear
- Actions are discrete (tap positions, ON/OFF switches)
- Multiple objectives conflict: fixing voltage may increase losses
- Classical optimisation solvers are too slow for real-time use

---

## 2. Why We Use Quantum RL (and Not Something Else)

| Approach | Problem |
|---|---|
| Classical MIP/OPF solvers | Too slow for real-time; need full system model |
| Rule-based controllers | Cannot adapt to new load patterns or faults |
| Classical DRL (SAC, PPO) | Works but needs millions of parameters → slow to deploy |
| Supervised learning | Needs expensive pre-solved labels at every operating point |
| **QE-SAC (this work)** | Fast inference, ~1% of classical parameters, no labels needed |

**Why quantum can work here — 3 reasons:**

1. **The state is structured, not random.**
   Grid voltages follow physical laws (DistFlow equations). Angle encoding
   maps voltage magnitudes directly to qubit rotation angles — natural fit.

2. **The action space is small and discrete.**
   Unlike robotics (continuous, huge), VVC actions are a small set of tap
   positions and ON/OFF switches. A compact quantum policy covers this well.

3. **The CAE solves the qubit bottleneck.**
   Past QRL failed on large states because raw high-dimensional input cannot
   fit into a few qubits. The co-adaptive CAE learns which 8 directions in
   state space matter most for control, and feeds only those to the VQC.
   Retraining every 500 steps keeps this compression aligned with grid changes.

> **One sentence:** QE-SAC works because it separates "what to pay attention to"
> (CAE — classical, adaptive) from "how to decide" (VQC — quantum, compact),
> and VVC is exactly the kind of structured, low-action-dimensionality task
> where a tiny quantum circuit can match a massive classical network.

---

## 3. Architecture — How It Works

```
High-dim grid state  s  (42-dim for 13-bus / 380-dim for 123-bus)
         │
         ▼
┌─────────────────────┐
│  Classical          │   Encoder: s → 64 → 32 → 8
│  Autoencoder (CAE)  │   Output scaled to [-π, π]
│  co-adaptive        │   Retrained every C = 500 gradient steps
└─────────────────────┘
         │  latent s' (8-dim)
         ▼
┌─────────────────────┐
│  8-qubit VQC        │   Angle encoding:  RY(s'_i)|0⟩ on each qubit
│  (PennyLane)        │   Layer × 2:
│                     │     CNOT(i, i+1)  nearest-neighbour entanglement
│                     │     RX(ζ_k)       trainable rotation per qubit
│                     │   Measurement: ⟨Z_i⟩ → output ∈ [-1, 1]^8
│                     │   Gradient: parameter-shift rule
└─────────────────────┘
         │  vqc_out (8-dim)
         ▼
┌─────────────────────┐
│  Linear + Softmax   │   8 → n_actions → action probabilities
└─────────────────────┘
         │
         ▼
  Per-device action  (cap ON/OFF, regulator tap, battery SoC)
```

**Critics (classical):** Twin MLP Q-networks, same as standard SAC.
**Replay buffer:** 1,000,000 transitions.
**SAC hyperparams:** γ=0.99, lr=1e-4, batch=256, ρ=0.005, α=0.2.

---

## 4. What We Compared and Who Won

### Performance Results (paper, avg 10 seeds)

| Algorithm | 13-bus Reward | 13-bus VViol | 123-bus Reward | 123-bus VViol | Params (123-bus) |
|---|---|---|---|---|---|
| **QE-SAC (ours)** | **−5.39** | **0** | **−9.53** | **0** | **42,329** |
| Classical SAC | −5.41 | 0.01 | −9.72 | 0 | 3,213,427 |
| QC-SAC (fixed PCA) | −5.91 | 0.02 | −10.78 | 0.01 | ~42K |
| SAC-AE | failed | failed | failed | failed | ~41K |

**QE-SAC wins on 3 things simultaneously:**
- Best reward (highest cumulative return)
- Zero voltage violations across all episodes
- Smallest parameter count (185× fewer than classical SAC on 13-bus)

**Why QE-SAC beats QC-SAC:**
Fixed PCA compression becomes stale as loads shift through the day.
The co-adaptive CAE continuously realigns the latent space to current
grid conditions — this is the key difference.

**Why QE-SAC beats SAC-AE:**
SAC-AE uses a fixed autoencoder with a classical MLP policy on top.
The autoencoder and policy are not jointly optimised end-to-end.
QE-SAC trains the CAE encoder and VQC together through the SAC objective.

---

## 5. Parameter Count — Why Our Numbers Differ from the Paper

| Component | Paper | Ours | Reason |
|---|---|---|---|
| QE-SAC 13-bus | 4,872 | 10,575 | We count decoder; paper counts inference path only |
| QE-SAC inference-only | 4,872 | 5,445 | CAE hidden dim difference (paper not specified) |
| Classical SAC 13-bus | 899,729 | 86,309 | Paper obs_dim ≈ 3,219 (OpenDSS); ours = 42 (DistFlow) |

**Root cause 1 — Classical SAC too small:**
Back-calculating from 899,729 params with 2×256 MLP reveals the paper's
state vector is ~3,219-dimensional. This comes from the full 3-phase
unbalanced OpenDSS state (per-phase voltages + currents + loads at every bus).
Our DistFlow state is only 42-dim, so the classical MLP is much smaller.

**Root cause 2 — QE-SAC slightly large:**
The CAE decoder (5,130 params) is needed for reconstruction loss during
training but is NOT used at inference. The paper counts inference-only.
Removing the decoder: our 5,445 vs paper's 4,872 — gap of 573 params,
which is a hidden layer size difference the paper does not specify.

**The VQC has exactly 16 parameters in both the paper and our code.**
(2 layers × 8 qubits — this matches perfectly.)

---

## 6. Bottlenecks, Constraints, and Limitations

### Algorithmic bottlenecks

| Issue | Detail |
|---|---|
| **VQC training is slow** | Parameter-shift needs 2×16 = 32 circuit evaluations per gradient step |
| **No batch parallelism** | PennyLane processes each sample in a loop — GPU parallelism is lost |
| **Barren plateaus** | Gradients vanish as qubit count or depth grows beyond 8 qubits / 2 layers |
| **Simulator overhead** | Wall-clock time is longer than classical SAC on 13/34-bus |

### Environment constraints

| Issue | Detail |
|---|---|
| **Simulator only** | PennyLane default.qubit is a classical statevector — no real quantum hardware |
| **No hard safety constraints** | Voltage limits are soft penalties in reward, not hard constraints |
| **Single time period** | No look-ahead; real VVC needs multi-period coordination |
| **No communication delay** | Real grids have measurement latency and sensor failures |
| **No generalization** | Trained and tested on the same feeder; retraining needed for new topology |

### What is missing from the paper's comparison

| Missing Comparison | Why It Matters |
|---|---|
| Rule-based Volt-VAR controller | The actual industry baseline — is any RL even needed? |
| Model Predictive Control (MPC) | Optimal when model is known — shows the RL gap to optimality |
| DDPG / TD3 | SAC family comparison incomplete without continuous-action baselines |
| Train 13-bus → test 34-bus | Generalization across different feeder sizes |
| N-1 contingency test | Performance under line outages — critical for real deployment |

---

## 7. Ideas for Future Work and New Comparisons

### Architecture improvements

| Idea | Expected Benefit |
|---|---|
| **Graph Neural Network (GNN) encoder** instead of CAE | Encodes grid topology directly — physically meaningful latent space |
| **Equivariant quantum circuit** | Exploit grid symmetry → fewer parameters, better generalisation |
| **Amplitude encoding** instead of angle encoding | Encodes exponentially more information per qubit |
| **Multi-agent QE-SAC** | Each zone of the grid gets its own small QE-SAC — scalable |
| **Transformer encoder** instead of CAE | Attention over bus states may compress better than MLP |

### Training improvements

| Idea | Expected Benefit |
|---|---|
| **Constrained RL (Lagrangian SAC)** | Hard voltage constraint — guaranteed zero violations |
| **Curriculum learning** (easy → hard load profiles) | Faster convergence, better worst-case performance |
| **Offline RL pre-training on historical data** | Reduce online interaction needed |
| **Physics-informed reward shaping** | Use DistFlow equations in reward → faster learning |

### Ablation studies the paper skipped

| Ablation | Research Question |
|---|---|
| VQC layers: 1 vs 2 vs 3 vs 4 | Where does the barren plateau start? |
| Qubits: 4 vs 8 vs 12 vs 16 | Optimal qubit count for VVC task size? |
| CAE interval C: 100 vs 500 vs 2000 | How sensitive is performance to update frequency? |
| Entanglement: linear vs circular vs full | Which topology best captures grid structure? |
| With vs without co-adaptation | How much does CAE retraining contribute? |

### The most valuable next experiment

> **Transfer learning test:**
> Train QE-SAC on 13-bus. Without retraining, evaluate on 34-bus.
> If performance holds up, it proves the quantum latent space learns
> general voltage control principles, not just feeder-specific patterns.
> This would be the strongest argument for real-world deployment.

---

## 8. Files in This Implementation

```
src/qe_sac/
├── env_utils.py        — IEEE 13-bus & 123-bus VVC environments (DistFlow)
├── autoencoder.py      — Co-adaptive CAE (input → 64 → 32 → 8 latent)
├── vqc.py              — 8-qubit 2-layer VQC (PennyLane, parameter-shift)
├── noise_model.py      — Depolarising noise robustness (λ = 0.1/0.5/1.0%)
├── sac_baseline.py     — Classical SAC with MLP actor
├── qe_sac_policy.py    — QE-SAC: CAE + VQC + Linear head
├── trainer.py          — Training loop + compare_agents()
└── metrics.py          — TrainingMetrics, evaluate_policy, count_parameters

tests/
└── test_qesac_env.py   — 24 unit tests (all passing)

notebooks/
└── qe_sac_experiment.ipynb  — Full training + comparison + noise test

artifacts/qe_sac/
├── results_13bus.json       — Reward & VViol per seed
└── noise_robustness.json    — VQC output diff under noise
```

---

## 9. Quick Start

```bash
# Run all tests
source .venv/bin/activate
python -m pytest tests/test_qesac_env.py -v

# Check parameter counts
python -c "
from src.qe_sac.env_utils import VVCEnv13Bus
from src.qe_sac.qe_sac_policy import QESACAgent
from src.qe_sac.sac_baseline import ClassicalSACAgent
env = VVCEnv13Bus()
obs_dim   = env.observation_space.shape[0]
n_actions = int(env.action_space.nvec.sum())
qe = QESACAgent(obs_dim, n_actions)
cl = ClassicalSACAgent(obs_dim, n_actions)
print(f'QE-SAC : {qe.param_count():,} params')
print(f'SAC    : {cl.param_count():,} params')
print(f'Ratio  : {cl.param_count() / qe.param_count():.1f}x')
"

# Run full experiment notebook
jupyter notebook notebooks/qe_sac_experiment.ipynb
```
