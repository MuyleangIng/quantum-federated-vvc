# QE-SAC-FL — Architecture, Comparison & Big Picture
**Ing Muyleang — Pukyong National University, QCL — 2026-04-01**

---

## PART 1 — THE FULL ARCHITECTURE IN CODE

### Layer 1: What One Client Looks Like Inside

```
╔══════════════════════════════════════════════════════════════════════════╗
║              AlignedQESACAgent  (one per utility company)                ║
║                                                                          ║
║  ┌────────────────────────────────────────────────────────────────────┐  ║
║  │  ACTOR  (makes decisions)                                          │  ║
║  │                                                                    │  ║
║  │   obs                                                              │  ║
║  │  (42/105/372)                                                      │  ║
║  │      │                                                             │  ║
║  │      ▼                                                             │  ║
║  │  ┌──────────────────────┐                                          │  ║
║  │  │   LocalEncoder       │  Linear(obs→64) → ReLU → Linear(64→32)  │  ║
║  │  │   PRIVATE ❌         │  Different weights per client            │  ║
║  │  │   stays on device    │  Compresses feeder-specific observations │  ║
║  │  └──────────────────────┘                                          │  ║
║  │           │ hidden (32-dim)                                        │  ║
║  │           ▼                                                        │  ║
║  │  ┌──────────────────────┐                                          │  ║
║  │  │  SharedEncoderHead   │  Linear(32→8) → Tanh → scale to [-π,π]  │  ║
║  │  │  FEDERATED ✅        │  264 params — identical across all       │  ║
║  │  │  goes to server      │  Ensures all clients speak same language │  ║
║  │  └──────────────────────┘                                          │  ║
║  │           │ latent z (8-dim, in [-π, π])                          │  ║
║  │           ▼                                                        │  ║
║  │  ┌──────────────────────┐                                          │  ║
║  │  │   8-qubit VQC        │  RY(z_i) encoding + 2 param layers      │  ║
║  │  │   FEDERATED ✅       │  CNOT entanglement + RY(θ) rotations     │  ║
║  │  │   goes to server     │  16 params — identical across all        │  ║
║  │  │                      │  Output: ⟨Z_i⟩ ∈ [-1, +1] × 8          │  ║
║  │  └──────────────────────┘                                          │  ║
║  │           │ vqc_out (8-dim, in [-1, +1])                          │  ║
║  │           ▼                                                        │  ║
║  │  ┌──────────────────────┐                                          │  ║
║  │  │  Linear Head         │  Linear(8→132) → Softmax                │  ║
║  │  │  LOCAL ❌            │  Action probabilities over 132 choices   │  ║
║  │  └──────────────────────┘                                          │  ║
║  │           │                                                        │  ║
║  │           ▼  action (0..131)                                       │  ║
║  └────────────────────────────────────────────────────────────────────┘  ║
║                                                                          ║
║  ┌────────────────────────────────────────────────────────────────────┐  ║
║  │  CRITICS (estimate value — LOCAL ❌ never shared)                  │  ║
║  │   Q1: Linear(obs+latent → 256 → 256 → 1)                          │  ║
║  │   Q2: same architecture, separate weights (twin critics)           │  ║
║  └────────────────────────────────────────────────────────────────────┘  ║
║                                                                          ║
║  ┌────────────────────────────────────────────────────────────────────┐  ║
║  │  REPLAY BUFFER  (LOCAL ❌ raw grid data stays on device)           │  ║
║  │   200,000 transitions:  (obs, action, reward, next_obs, done)      │  ║
║  └────────────────────────────────────────────────────────────────────┘  ║
╚══════════════════════════════════════════════════════════════════════════╝

Parameter budget per client:
  LocalEncoder (PRIVATE):         4,832 / 8,864 / 25,952  params
  SharedEncoderHead (FEDERATED):    264                    params  ─┐
  VQC (FEDERATED):                   16                    params  ─┤ 280 total
  Linear head (LOCAL):            1,188                    params    federated
  Critics ×2 (LOCAL):            ~2,100                   params
```

---

### Layer 2: The Federation Protocol (Server View)

```
╔══════════════════════════════════════════════════════════════════════════╗
║                        FEDERATION SERVER                                 ║
║                      (one round = one loop)                              ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  STEP 1 ─ LOCAL TRAINING (parallel, 1 GPU each, 1,000 steps)            ║
║                                                                          ║
║  Utility A ─────────────────────────────────────── Utility B            ║
║  (13-bus, cuda:0)                                   (34-bus, cuda:1)    ║
║       │  train: obs→action→reward                        │              ║
║       │  update: LocalEnc, SharedHead, VQC, Critics       │              ║
║       │  log: mean_reward, VQC_grad_norm                  │              ║
║       └──────────────────────┬──────────────────────┘    │              ║
║                              │                      Utility C           ║
║                              │                   (123-bus, cuda:2)      ║
║                              │                        │                 ║
║                              └─────────────┬──────────┘                 ║
║                                            │                            ║
║  STEP 2 ─ COLLECT SHARED WEIGHTS                                        ║
║                                                                          ║
║        A sends:  { shared_head: state_dict, vqc: tensor[2,8] }         ║
║        B sends:  { shared_head: state_dict, vqc: tensor[2,8] }         ║
║        C sends:  { shared_head: state_dict, vqc: tensor[2,8] }         ║
║                                                                          ║
║        Total bytes per round: 280 × 4 × 2 = 2,240 bytes                ║
║        vs Classical SAC FL:  110,724 × 4 × 2 = 885,792 bytes           ║
║                                                                          ║
║  STEP 3 ─ FEDAVG (uniform average)                                      ║
║                                                                          ║
║        SharedHead_avg[k] = (1/3) × (A[k] + B[k] + C[k])  for each key ║
║        VQC_avg           = (1/3) × (A_vqc + B_vqc + C_vqc)            ║
║                                                                          ║
║  STEP 4 ─ BROADCAST BACK                                                ║
║                                                                          ║
║        Each client receives:  SharedHead_avg + VQC_avg                  ║
║        Each client keeps:     LocalEncoder (unchanged)                   ║
║                               Critics (unchanged)                        ║
║                               Replay buffer (unchanged)                  ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

### Layer 3: The Three Feeders Side by Side

```
┌─────────────────────┬─────────────────────┬─────────────────────┐
│   Utility A         │   Utility B         │   Utility C         │
│   13-bus (small)    │   34-bus (medium)   │   123-bus (large)   │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ obs_dim:  42        │ obs_dim:  105        │ obs_dim:  372        │
│ LocalEnc: 4,832 p   │ LocalEnc: 8,864 p   │ LocalEnc: 25,952 p  │
│ Device:   cuda:0    │ Device:   cuda:1    │ Device:   cuda:2    │
│ Reward ~: −300      │ Reward ~: −65       │ Reward ~: −5,400    │
│ Grad norm:  HIGH    │ Grad norm: LOW      │ Grad norm: MEDIUM   │
├─────────────────────┼─────────────────────┼─────────────────────┤
│       SHARED — identical weights, identical architecture        │
│  SharedEncoderHead: 264 params   VQC: 16 params (2×8 tensor)   │
└─────────────────────────────────────────────────────────────────┘
```

---

### Layer 4: Data Flow — What Travels vs What Stays

```
                    ╔══════════════════╗
                    ║    FL SERVER     ║
                    ║  (aggregation)   ║
                    ╚════════╤═════════╝
              ┌──────────────┼──────────────┐
              │              │              │
         ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
         │ 280 ×f32│←───│FedAvg   │───►│ 280 ×f32│
         │ per     │    │uniform  │    │ per     │
         │ client  │    │average  │    │ client  │
         └─────────┘    └─────────┘    └─────────┘
              ↑                              ↑
         SharedHead+VQC                SharedHead+VQC
         (280 params)                  (280 params)
         TRAVELS ✅                    TRAVELS ✅
              │                              │
    ┌─────────┴──────────┐       ┌──────────┴─────────┐
    │  Utility A Device  │       │  Utility C Device  │
    │                    │       │                    │
    │  LocalEncoder ❌   │       │  LocalEncoder ❌   │
    │  Critics      ❌   │       │  Critics      ❌   │
    │  Replay Buffer❌   │       │  Replay Buffer❌   │
    │  Raw obs data ❌   │       │  Raw obs data ❌   │
    └────────────────────┘       └────────────────────┘
         STAYS LOCAL                   STAYS LOCAL
```

---

## PART 2 — THREE-WAY COMPARISON

### Base Paper → QE-SAC → QE-SAC-FL (This Work)

```
╔══════════════╦══════════════════════════╦══════════════════════════════╗
║              ║  BASE PAPER (QE-SAC)     ║  THIS WORK (QE-SAC-FL)       ║
║  Property    ║  Lin et al. 2025         ║  Muyleang 2026               ║
╠══════════════╬══════════════════════════╬══════════════════════════════╣
║ Problem      ║ VVC on 1 feeder          ║ VVC on 3 feeders — no data   ║
║              ║ (single utility)         ║ sharing across competitors   ║
╠══════════════╬══════════════════════════╬══════════════════════════════╣
║ Encoder      ║ MLP CAE (flat vector)    ║ LocalEncoder (private, per   ║
║              ║ 42→64→32→8               ║ client) + SharedEncoderHead  ║
║              ║ retrain every 500 steps  ║ (federated, 264 params)      ║
╠══════════════╬══════════════════════════╬══════════════════════════════╣
║ VQC          ║ 8 qubits, 2 layers       ║ IDENTICAL — 8 qubits, 16 p   ║
║              ║ 16 params, param-shift   ║ (never changed)              ║
╠══════════════╬══════════════════════════╬══════════════════════════════╣
║ Federation   ║ None (single client)     ║ FedAvg on SharedHead + VQC   ║
║              ║                          ║ 280 params per round         ║
╠══════════════╬══════════════════════════╬══════════════════════════════╣
║ Safety       ║ Soft reward penalty      ║ Soft penalty (same)          ║
║              ║ violations CAN happen    ║ Hard constraint = QE-SAC+    ║
╠══════════════╬══════════════════════════╬══════════════════════════════╣
║ Data privacy ║ Not applicable           ║ LocalEncoder + buffer NEVER  ║
║              ║                          ║ leave client device          ║
╠══════════════╬══════════════════════════╬══════════════════════════════╣
║ Best result  ║ −5.39 reward             ║ −165.0 / −15.2 / −4034.5    ║
║              ║ (OpenDSS env)            ║ (+50/+77/+25% vs local)      ║
╠══════════════╬══════════════════════════╬══════════════════════════════╣
║ Novel        ║ QE-SAC works for VVC     ║ heterogeneous FL problem — new quantum FL flaw   ║
║ findings     ║ 185× fewer params        ║ CSA — gradient scale bias    ║
║              ║ VQC noise robust         ║ PAD — alignment breaks under ║
║              ║                          ║ partial participation        ║
╠══════════════╬══════════════════════════╬══════════════════════════════╣
║ Comm. cost   ║ Not applicable           ║ 395× less than classical FL  ║
╠══════════════╬══════════════════════════╬══════════════════════════════╣
║ Clients      ║ 1 (single utility)       ║ 3 (heterogeneous feeders)    ║
║              ║ same env train/test      ║ 42/105/372 dim obs_space     ║
╚══════════════╩══════════════════════════╩══════════════════════════════╝
```

---

## PART 3 — THE NOVEL FINDINGS IN ONE DIAGRAM

```
╔══════════════════════════════════════════════════════════════════════════╗
║              THE THREE DISCOVERIES — VISUAL PROOF                        ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  EXPERIMENT 1: Unaligned FL                                              ║
║  ─────────────────────────                                               ║
║  Each client: full private encoder → VQC (only VQC federated)           ║
║                                                                          ║
║   13-bus  ■■■■■■■■■■■■■■■■■■■■■■■■■■■  -331.4  (local)                 ║
║           ■■■■■■■■■■■■■■■■■■■■■■■■■    -336.6  (unaligned FL) WORSE    ║
║                                                                          ║
║   34-bus  ■■■■■  -65.5  (local)                                         ║
║           ■■■■   -69.6  (unaligned FL) WORSE                            ║
║                                                                          ║
║  123-bus  ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■  -5364.4  (local)    ║
║           ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ -5420.5  (FL) WORSE ║
║                                                                          ║
║  → DISCOVERY 1: heterogeneous FL problem  (all three clients degraded by federation)        ║
║                                                                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  EXPERIMENT 2: Aligned FL — 50 rounds vs 200 rounds                     ║
║  ────────────────────────────────────────────────                        ║
║                                                                          ║
║   Client      Round 50    Round 200    Local     Who benefits?           ║
║   ────────────────────────────────────────────────────────              ║
║   13-bus       -326.3 ✅   -339.5 ❌   -331.4    small wins EARLY       ║
║   34-bus        -85.0 ❌    -69.3 ❌    -65.5    medium NEVER wins      ║
║   123-bus     -5402.5 ❌  -5251.4 ✅  -5364.4    large wins LATE        ║
║                                                                          ║
║              Round 50            Round 200                              ║
║              ←── small wins ──→  ←── large wins ──→                    ║
║              ┌──────────────────┬──────────────────┐                   ║
║              │  13-bus PASS     │  123-bus PASS    │                   ║
║              │  123-bus FAIL    │  13-bus FAIL     │                   ║
║              │  34-bus FAIL     │  34-bus FAIL     │                   ║
║              └──────────────────┴──────────────────┘                   ║
║                       benefit rotates — never all at once               ║
║                                                                          ║
║  → DISCOVERY 2: CSA  (gradient scale imbalance, not data content)       ║
║                                                                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  EXPERIMENT 3: Partial Participation (2/3 clients per round)             ║
║  ────────────────────────────────────────────────────────               ║
║                                                                          ║
║  Round r:    A + B participate  → SharedHead ← A+B objective            ║
║  Round r+1:  B + C participate  → SharedHead ← B+C objective            ║
║  Round r+2:  A + C participate  → SharedHead ← A+C objective            ║
║                    ↑ oscillates every round                             ║
║                                                                          ║
║  Result: ALL clients worse than local-only                              ║
║  Signature: HIGH VQC grad norms (0.000721) + LOWER rewards              ║
║             = gradients in oscillating directions, not learning         ║
║                                                                          ║
║  → DISCOVERY 3: PAD  (FedAvg partial-participation proof doesn't apply  ║
║                       to coupled split-encoder architectures)           ║
║                                                                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  EXPERIMENT 4: Personalised FL — THE SOLUTION                           ║
║  ────────────────────────────────────────────                            ║
║                                                                          ║
║  Phase 1:  50 rounds aligned FL  → all clients share warm-start         ║
║  Phase 2:  5,000 local fine-tune → each client adapts to own feeder     ║
║                                                                          ║
║   13-bus   ████████████████████████████  -165.0   +50.2% vs local ✅   ║
║            (local: ██████████████████    -331.4)                        ║
║                                                                          ║
║   34-bus   ██  -15.2   +76.8% vs local ✅                               ║
║            (local: █████  -65.5)                                        ║
║                                                                          ║
║  123-bus   ████████████████████████████████████████  -4034.5  +24.8% ✅║
║            (local: ████████████████████████████████████████████-5364.4)║
║                                                                          ║
║  → ALL THREE CLIENTS IMPROVE SIMULTANEOUSLY                              ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## PART 4 — MAIN OBJECTIVES AND EXPECTED OUTCOMES

```
╔══════════════════════════════════════════════════════════════════════════╗
║           OBJECTIVE MAP — QE-SAC-FL                                      ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  PRIMARY OBJECTIVE                                                       ║
║  ─────────────────                                                       ║
║  Enable three competing utilities to train a shared quantum VVC          ║
║  policy without sharing raw grid data.                                   ║
║                                                                          ║
║  ┌──────────────────────────────────────────────────────────────────┐   ║
║  │  OBJECTIVE 1: Prove that naive quantum FL fails (heterogeneous FL problem)           │   ║
║  │  Status: ✅  Evidence: all 3 clients worse with unaligned FL      │   ║
║  │  Why novel: no FL paper identifies this failure mode              │   ║
║  └──────────────────────────────────────────────────────────────────┘   ║
║                                                                          ║
║  ┌──────────────────────────────────────────────────────────────────┐   ║
║  │  OBJECTIVE 2: Solve heterogeneous FL problem with minimal communication overhead     │   ║
║  │  Status: ✅  Solution: SharedEncoderHead (280 params federated)   │   ║
║  │  Result:  395× less communication than classical federated SAC    │   ║
║  └──────────────────────────────────────────────────────────────────┘   ║
║                                                                          ║
║  ┌──────────────────────────────────────────────────────────────────┐   ║
║  │  OBJECTIVE 3: Achieve +reward for ALL clients simultaneously     │   ║
║  │  Status: ✅  Personalised FL: +50/+77/+25% on all 3 clients      │   ║
║  │  How: FL warm-start provides better initialisation than random    │   ║
║  └──────────────────────────────────────────────────────────────────┘   ║
║                                                                          ║
║  ┌──────────────────────────────────────────────────────────────────┐   ║
║  │  OBJECTIVE 4: Characterise deployment risks                      │   ║
║  │  Status: ✅  CSA + PAD discovered and documented                  │   ║
║  │  Impact: PAD means full participation required for quantum FL     │   ║
║  └──────────────────────────────────────────────────────────────────┘   ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## PART 5 — THE COMPLETE RESEARCH SCOPE

```
╔══════════════════════════════════════════════════════════════════════════╗
║                     FULL RESEARCH SCOPE                                  ║
║                                                                          ║
║  PAST  ──────────────────────────────────────────────────────────────   ║
║                                                                          ║
║  Base paper (Lin et al. 2025)                                           ║
║    QE-SAC: quantum RL for VVC, 1 feeder, no federation                 ║
║    Proved: reward ≈ classical SAC, 185× fewer params                   ║
║                                                                          ║
║  Our baseline experiments                                               ║
║    H1 ✅: QE-SAC 6.75× more stable than classical SAC (variance)       ║
║    H2 ✅: CAE co-adaptation is the source of stability                  ║
║    GNN encoder built (1,457 params vs 11,414 CAE)                      ║
║                                                                          ║
║  PRESENT  ────────────────────────────────────────────────────────────  ║
║                                                                          ║
║  QE-SAC-FL (this work — federated learning)                             ║
║    ✅ heterogeneous FL problem discovered and named                                          ║
║    ✅ CSA discovered and named                                           ║
║    ✅ PAD discovered and named                                           ║
║    ✅ SharedEncoderHead architecture built and verified                  ║
║    ✅ Personalised FL: +25–77% on all 3 clients                         ║
║    ✅ H3: 395× communication reduction (mathematical proof)             ║
║                                                                          ║
║  FUTURE  ──────────────────────────────────────────────────────────     ║
║                                                                          ║
║  Paper 1 (writing now)                                                  ║
║    → Submit: IEEE Transactions on Smart Grid                            ║
║    → 5-seed runs for statistical significance                           ║
║    → Gradient-normalised FedAvg (fix CSA)                              ║
║    → FedProx on SharedHead (fix PAD)                                   ║
║                                                                          ║
║  Paper 2 — QE-SAC+ (April–June 2026)                                   ║
║    → GNN encoder replaces MLP CAE                                       ║
║    → Lagrangian safety constraint (hard guarantee, not soft penalty)    ║
║    → Transfer learning: train on 13-bus → deploy on 123-bus             ║
║       (FL personalised warm-start motivates this directly)              ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## PART 6 — EXPERIMENT BIG PICTURE

```
╔══════════════════════════════════════════════════════════════════════════╗
║                    ALL EXPERIMENTS — WHAT EACH PROVES                    ║
╠═══════════════════════╦══════════════╦══════════════════════════════════╣
║  Experiment           ║  Condition   ║  What it proves                  ║
╠═══════════════════════╬══════════════╬══════════════════════════════════╣
║  E1: Local only       ║  3 clients,  ║  Baseline. Each client's best    ║
║                       ║  no FL       ║  result without any cooperation. ║
╠═══════════════════════╬══════════════╬══════════════════════════════════╣
║  E2: Unaligned FL     ║  FedAvg on   ║  heterogeneous FL problem exists. Standard quantum   ║
║                       ║  VQC only    ║  FL hurts all 3 clients.         ║
║                       ║              ║  This is NEW — no paper found it ║
╠═══════════════════════╬══════════════╬══════════════════════════════════╣
║  E3: Aligned FL 50r   ║  FedAvg on   ║  heterogeneous FL problem is fixed. Small feeder    ║
║                       ║  SharedHead  ║  benefits first (CSA early).     ║
║                       ║  + VQC       ║  Only 13-bus passes H1.          ║
╠═══════════════════════╬══════════════╬══════════════════════════════════╣
║  E4: Aligned FL 200r  ║  Same but    ║  CSA confirmed. Benefit reverses ║
║                       ║  200 rounds  ║  — now only 123-bus passes H1.   ║
║                       ║              ║  This is NEW — no paper found it ║
╠═══════════════════════╬══════════════╬══════════════════════════════════╣
║  E5: Partial FL       ║  2/3 clients ║  PAD exists. All 3 clients worse ║
║                       ║  per round   ║  than local. McMahan's proof does ║
║                       ║              ║  not apply here. NEW finding.     ║
╠═══════════════════════╬══════════════╬══════════════════════════════════╣
║  E6: Personalised FL  ║  50r aligned ║  SOLUTION. All 3 clients better. ║
║                       ║  + 5K local  ║  +50/+77/+25%. Main result.      ║
║                       ║  fine-tune   ║  Bypasses both heterogeneous FL problem and CSA.     ║
╠═══════════════════════╬══════════════╬══════════════════════════════════╣
║  H3: Comm cost        ║  Param count ║  395× less than classical FL.    ║
║  (mathematical)       ║  × bytes     ║  Mathematical proof. Always true ║
╠═══════════════════════╬══════════════╬══════════════════════════════════╣
║  S1–S5: Verification  ║  Code checks ║  Architecture correct. Privacy   ║
║  (notebook cells)     ║  on weights  ║  preserved. FedAvg works right.  ║
╚═══════════════════════╩══════════════╩══════════════════════════════════╝
```

---

## PART 7 — WHY EACH REFERENCE SUPPORTS THIS (NOT AGAINST IT)

```
╔══════════════════════════════════════════════════════════════════════════╗
║  REFERENCE         WHAT THEY PROVE        WHY WE CITE THEM              ║
╠══════════════════════════════════════════════════════════════════════════╣
║  McMahan 2017      FedAvg converges        We DISPROVE their claim holds ║
║  (FedAvg)          with partial dropout    for split-encoder quantum FL  ║
║                                            → This proves PAD is NEW      ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Li 2020           FedProx adds prox       They fix data heterogeneity   ║
║  (FedProx)         term for non-IID        We fix encoder incompatibility ║
║                    data distributions      → heterogeneous FL problem ≠ FedProx problem      ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Karimireddy 2020  SCAFFOLD fixes          They fix gradient variance    ║
║  (SCAFFOLD)        gradient drift          from DATA, not from obs_dim   ║
║                    via control variates    → CSA ≠ SCAFFOLD problem      ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Zhao 2018         Non-IID data hurts      They study DATA content diff  ║
║                    FedAvg                  We study obs_dim scale diff   ║
║                                            → CSA is a NEW category       ║
╠══════════════════════════════════════════════════════════════════════════╣
║  McClean 2018      Barren plateaus         Motivates why 123-bus has     ║
║  (Barren plateau)  vanish with qubit       near-zero grad norms (ISSUE_  ║
║                    count                   002) — a known challenge      ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Lin 2025          QE-SAC works for VVC    Our STARTING POINT. We extend ║
║  (base paper)      1 feeder, 16 params     to 3 federated clients.       ║
║                                            VQC architecture unchanged.   ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## PART 8 — THE ONE-PAGE SUMMARY FOR ANYONE

```
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║   WHAT IS THE PROBLEM?                                                   ║
║   Three power companies need smarter voltage control.                    ║
║   They cannot share data — they are competitors.                        ║
║                                                                          ║
║   WHAT DID WE TRY FIRST?                                                 ║
║   Apply standard federated learning to quantum RL agents.               ║
║   Every company got WORSE. (This had never been reported before.)       ║
║                                                                          ║
║   WHY DID IT FAIL?                                                       ║
║   Each company's encoder learned to compress observations in its        ║
║   own way. After averaging, the quantum circuit received nonsense.      ║
║   We named this: Quantum Latent Space Incompatibility (heterogeneous FL problem).           ║
║                                                                          ║
║   HOW DID WE FIX IT?                                                     ║
║   We split the encoder into a private part (stays local) and a         ║
║   shared part (goes to the server). Now all encoders compress into      ║
║   the same 8-dimensional space before reaching the quantum circuit.     ║
║                                                                          ║
║   DID THE FIX WORK PERFECTLY?                                            ║
║   Not immediately. We found two more new problems:                      ║
║   - CSA: Small companies benefit early, large companies benefit late.   ║
║     Never all at once.                                                  ║
║   - PAD: If any company misses a training round, alignment breaks.      ║
║                                                                          ║
║   WHAT IS THE FINAL RESULT?                                              ║
║   Personalised FL: federate first, then fine-tune locally.              ║
║   Result: +50%, +77%, +25% reward improvement on all 3 companies.      ║
║   Communication cost: 395× less than classical federated RL.            ║
║                                                                          ║
║   WHY IS THIS PUBLISHABLE?                                               ║
║   Three failure modes that no paper has found or named.                 ║
║   One strong solution with experimental proof.                          ║
║   One mathematical proof. Target: IEEE Transactions on Smart Grid.      ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

*See also:*
- *`FULL_RESEARCH_COMPENDIUM.md` — complete technical reference (1,100+ lines)*
- *`RESULTS_SUMMARY.md` — all numbers in one place*
- *`HYPOTHESES.md` — all 9 hypotheses with status*
- *`notebooks/qe_sac_fl_experiment.ipynb` — 56 cells, fully run*
