# QE-SAC-FL — Complete Architecture Flow

**Federated Quantum Reinforcement Learning for Volt-VAR Control**
Researcher: Ing Muyleang — Pukyong National University, QCL

---

## Table of Contents
1. [Level 0 — Real World](#level-0--real-world)
2. [Level 1 — Environment](#level-1--environment-per-client)
3. [Level 2 — Agent Architecture](#level-2--agent-architecture-per-client)
4. [Level 3 — SAC Training Loop](#level-3--sac-training-loop-per-client-per-round)
5. [Level 4 — Federated Loop](#level-4--federated-loop-50-rounds)
6. [Level 5 — Federated vs. Private](#level-5--what-is-federated-vs-private)
7. [Level 6 — Three Conditions Compared](#level-6--3-conditions-compared)
8. [Level 7 — Files Map](#level-7--files-map)
9. [Component Summary](#component-summary)

---

## Level 0 — Real World

```
  Utility A               Utility B               Utility C
  (urban grid)            (rural grid)             (large grid)
  13-bus feeder           34-bus feeder            123-bus feeder
  obs_dim = 42            obs_dim = 105            obs_dim = 372
  GPU: cuda:0             GPU: cuda:1              GPU: cuda:2
```

Each utility **owns** its grid. Data is **private**. They never share raw observations.

---

## Level 1 — Environment (per client)

```
          ┌─────────────────────────────────────────────┐
          │              VVCEnv (DistFlow)               │
          │                                              │
          │  State:  voltage at every bus (p.u.)         │
          │          reactive power at loads             │
          │          tap regulator position              │
          │                                              │
          │  Action: [cap_A, cap_B, tap]                 │
          │          cap ∈ {0,1}  tap ∈ {0..32}         │
          │          → 2 × 2 × 33 = 132 combinations    │
          │                                              │
          │  Reward: − Σ voltage_violation²              │
          │          − λ × reactive_power_usage          │
          └─────────────────────────────────────────────┘
                    ↑ action            ↓ obs, reward
```

**Goal:** Keep voltage at every bus between **0.95–1.05 p.u.**
**Actions:** flip capacitor banks ON/OFF + move regulator tap → 132 joint combinations.
**Reward:** penalty for voltage violations + high reactive power usage.

### Three Grids (Three Clients)

| Client | Grid | Size | obs_dim |
|--------|------|------|---------|
| A | 13-bus (urban) | small | 42 |
| B | 34-bus (rural) | medium | 105 |
| C | 123-bus (large) | large | 372 |

---

## Level 2 — Agent Architecture (per client)

```
  obs (42–372 dim)
        │
        ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  AlignedCAE  (encoder — SPLIT into 2 parts for FL)             │
  │                                                                 │
  │   ┌──────────────────────────┐   ┌──────────────────────────┐  │
  │   │   LocalEncoder (PRIVATE) │   │  SharedEncoderHead (FL)  │  │
  │   │   obs_dim → 32           │──▶│  32 → 8                  │  │
  │   │   MLP, private per client│   │  same architecture for   │  │
  │   │   NEVER shared           │   │  all clients             │  │
  │   │   params ≈ 272           │   │  params = 272            │  │
  │   └──────────────────────────┘   └──────────────────────────┘  │
  │                                            │ FEDERATED ↕        │
  └────────────────────────────────────────────│────────────────────┘
                                               │
                                    latent z (8 dim, tanh×π)
                                    z[i] ∈ [−π, +π]
                                               │
                                               ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  VQC — Variational Quantum Circuit          FEDERATED ↕        │
  │                                                                 │
  │  8 qubits (one per latent dim)                                  │
  │                                                                 │
  │  Layer 1:  RY(z[0]) RY(z[1]) ... RY(z[7])   ← encode latent   │
  │            CNOT ring entanglement                               │
  │            RZ(θ[0]) RZ(θ[1]) ... RZ(θ[7])   ← trainable       │
  │                                                                 │
  │  Layer 2:  RY(z[0]) RY(z[1]) ... RY(z[7])   ← re-encode       │
  │            CNOT ring entanglement                               │
  │            RZ(θ[8]) RZ(θ[9]) ... RZ(θ[15])  ← trainable       │
  │                                                                 │
  │  Measure:  ⟨Z⟩ on all 8 qubits → 8 expectation values         │
  │  Params:   16 total  (vs 110,724 in classical SAC)             │
  └─────────────────────────────────────────────────────────────────┘
                                               │
                                    8 expectation values
                                               │
                                               ▼
                              Linear layer → Softmax
                                               │
                                  π(a|s) over 132 actions
                                               │
                                               ▼
                                       action sampled
```

---

## Level 3 — SAC Training Loop (per client, per round)

```
                      ┌─────────────────────┐
                      │   Replay Buffer      │
                      │   (200K transitions) │
                      └──────┬──────────────┘
                             │  sample batch (512)
             ┌───────────────▼──────────────────────────┐
             │             SAC Update                    │
             │                                           │
             │  Critic 1 & 2  (Q-networks, classical)   │
             │    minimize:  (Q − target_Q)²             │
             │                                           │
             │  Actor (VQC + SharedHead + LocalEncoder)  │
             │    maximize:  Q(s,a) − α × log π(a|s)    │
             │                                           │
             │  Entropy coeff α  (auto-tuned)            │
             └───────────────────────────────────────────┘
                         runs for 1,000 steps per round
```

---

## Level 4 — Federated Loop (50 rounds)

```
  Round 1 → 50:

  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │  Client A    │   │  Client B    │   │  Client C    │
  │  13-bus      │   │  34-bus      │   │  123-bus     │
  │              │   │              │   │              │
  │  SAC update  │   │  SAC update  │   │  SAC update  │
  │  1000 steps  │   │  1000 steps  │   │  1000 steps  │
  │  (parallel)  │   │  (parallel)  │   │  (parallel)  │
  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
         │                  │                  │
         │   SharedHead     │   SharedHead     │   SharedHead
         │   weights (272)  │   weights (272)  │   weights (272)
         │   VQC weights    │   VQC weights    │   VQC weights
         │   (16 params)    │   (16 params)    │   (16 params)
         └──────────────────┴──────────────────┘
                            │
                     ┌──────▼──────┐
                     │   FedAvg    │
                     │             │
                     │  w_global = │
                     │  (w_A +     │
                     │   w_B +     │
                     │   w_C) / 3  │
                     └──────┬──────┘
                            │  broadcast updated
                            │  SharedHead + VQC
                            │  back to all 3
             ┌──────────────┼──────────────┐
         Client A       Client B       Client C
         loads           loads          loads
         w_global        w_global       w_global
                            │
                       Round + 1
```

**Configuration (paper_config):**

| Parameter | Value |
|-----------|-------|
| n_rounds | 50 |
| local_steps | 1,000 |
| total steps per client | 50,000 |
| batch_size | 512 |
| lr | 3e-4 |
| buffer_size | 200,000 |
| seeds | [0, 1, 2, 3, 4] |
| parallel_clients | True (3× RTX 4090) |

---

## Level 5 — What is Federated vs. Private

### Shared (uploaded to server every round)

| Component | Params | Bytes |
|-----------|--------|-------|
| VQC params | 16 floats | 64 bytes |
| SharedEncoderHead | 272 floats | 1,088 bytes |
| **Total per round per client** | **288** | **1,152 bytes** |
| **Total (50 rounds × 3 clients)** | | **168 KB** |
| Classical SAC FL (comparison) | | 126.7 MB |
| **Reduction ratio** | | **6,920×** |

### Private (never leaves client)

- Raw observations (grid voltages, loads)
- Replay buffer (200K transitions)
- LocalEncoder weights (obs_dim → 32)
- Critic Q-networks

---

## Level 6 — 3 Conditions Compared

| | Local Only | Unaligned FL | Aligned FL |
|---|---|---|---|
| **What's shared** | Nothing | VQC only | SharedHead + VQC |
| **CAE** | Full private | Full private | LocalEncoder private |
| **VQC** | Private | Federated | Federated |
| **SharedHead** | — | — | Federated |
| **Result** | Baseline | WORSE than local | BETTER (13-bus) |
| **Problem** | — | QLSI | CSA on 123-bus |

### Known Problems

**Problem 1 — QLSI (Quantum Latent Space Incompatibility)**
Each client's CAE independently learns its own meaning for the 8 latent dimensions.
When the VQC is shared, Client A's `z[0]` means "voltage drop" but Client B's `z[0]`
means "aggregate load" → VQC receives contradictory signals → worse than training alone.

```
Fix: AlignedCAE = LocalEncoder (private, obs→32) + SharedEncoderHead (federated, 32→8)
     All clients share the same 8-dim latent meaning → VQC works correctly
```

**Problem 2 — CSA (Client Size Asymmetry)**
At round 50: 13-bus benefits from federation but 123-bus does not — larger clients
dominate the FedAvg gradient.

```
Fix (Task 2): Gradient-normalised FedAvg
     Weight each client's update by 1/obs_dim → fair contribution from all sizes
```

**Problem 3 — PAD (Partial Alignment Drift)**
When a client skips a round, SharedHead is updated without its contribution →
the returning client's local model has drifted → gradient norms spike → VQC destabilises.

```
Fix (Task 3): FedProx
     Add term  μ × ||w − w_global||²  to the SharedHead loss
     Keeps each client close to the global model even when offline
```

---

## Level 7 — Files Map

```
src/
├── qe_sac/
│   ├── env_utils.py          ← 13-bus + 123-bus DistFlow environments
│   ├── qe_sac_policy.py      ← VQC + Actor (original QE-SAC)
│   ├── sac_baseline.py       ← Classical SAC for comparison
│   ├── constrained_sac.py    ← Lagrangian SAC (H3 task series)
│   └── trainer.py            ← Training loop
│
└── qe_sac_fl/
    ├── fed_config.py         ← All hyperparameters + paper_config()
    ├── federated_trainer.py  ← FedAvg loop, run_all_conditions()
    ├── aligned_encoder.py    ← AlignedCAE (LocalEncoder + SharedHead)
    ├── aligned_agent.py      ← QE-SAC agent with shared head
    └── env_34bus.py          ← 34-bus + 123-bus FL environments

notebooks/
├── qe_sac_experiment.ipynb      ← Original QE-SAC results (H1, H2 confirmed)
├── qe_sac_fl_experiment.ipynb   ← FL experiments (H1–H6)
└── fl_research_notes.ipynb      ← Theory, problems, solutions, references

artifacts/qe_sac_fl/
├── local_only_results.json              ✅ done
├── QE-SAC-FL_results.json              ✅ done (unaligned)
├── QE-SAC-FL-Aligned_results.json      ✅ done (50 rounds)
├── QE-SAC-FL-Aligned-200r_results.json ✅ done
├── QE-SAC-FL-Partial_results.json      ✅ done
└── QE-SAC-FL-Personalized_results.json ✅ done
```

---

## Hypotheses Status

| ID | Claim | Status |
|----|-------|--------|
| H1 | Federated VQC reward > local-only (all 3 clients) | Partial — 13-bus YES, 123-bus NO |
| H2 | FL converges faster than local | Inconclusive — 50K steps not enough |
| H3 | QE-SAC-FL communication cost << Classical FL | **Proved** (math: 6,920× reduction) |
| H4 | VQC avoids barren plateau on small feeders | Confirmed — 123-bus shows near-zero gradient |
| H5 | Personalised FL > aligned FL per-client | Partial — marginal at 50 rounds |
| H6 | FedProx makes FL robust to partial participation | **Pending** (Task 3) |

---

## Component Summary

| Component | What it does |
|-----------|-------------|
| **RL (SAC)** | Agent learns to pick grid actions by trial and error to maximise reward |
| **QE (VQC)** | Replaces classical actor with a 16-parameter quantum circuit → more stable |
| **FL (FedAvg)** | 3 utilities share only VQC + SharedHead weights every round, not raw data |
| **AlignedCAE** | Splits encoder into private part + shared part → fixes QLSI |
| **FedProx** *(Task 3)* | Adds penalty to keep clients close to global model → fixes PAD |
| **Grad-norm FedAvg** *(Task 2)* | Normalises client contributions by obs_dim → fixes CSA |
