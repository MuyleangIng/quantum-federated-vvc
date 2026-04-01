# Research Big Picture & Proposed Solution
## Quantum RL for Safe, Generalizable Volt-VAR Control

---

## THE BIG PICTURE

### The World Problem

The power grid is changing fast.

```
OLD GRID (simple)                    NEW GRID (complex)
─────────────────                    ──────────────────
Power flows one way                  Power flows both ways
  substation → homes                   homes (solar) → grid too

Load is predictable                  Load is unpredictable
  same pattern every day               EV charging, battery storage,
                                       cloud cover changes solar output

One operator controls everything     Thousands of small devices
  a few large generators               scattered across the feeder
```

This shift breaks classical grid control. Voltage now fluctuates
dozens of times per day instead of slowly drifting once or twice.
Human operators and rule-based systems cannot react fast enough.

---

### The Technical Problem

Every house and building on a distribution feeder must receive voltage
inside a safe band: **[0.95 pu, 1.05 pu]** (roughly ±5% of nominal).

Outside that band:
- Too low  → appliances fail, motors overheat
- Too high → equipment damage, safety hazard

To keep voltage in range, operators control:

```
DEVICE              ACTION              EFFECT
──────────────────────────────────────────────────────
Capacitor bank      ON / OFF            Injects reactive power → raises voltage
Voltage regulator   Tap 0-32            Fine-tunes voltage ratio on a branch
Battery storage     Charge level 0-32   Absorbs or injects real power
```

**This control problem has three hard properties:**

| Property | Why it makes the problem hard |
|---|---|
| **Real-time** | Decisions every 1–5 minutes, 24 hours a day |
| **Combinatorial** | Caps × taps × battery levels = millions of combinations |
| **Non-linear** | Voltage depends on all devices simultaneously (coupled) |

---

### Why Current Solutions Fail

```
APPROACH            WHAT IT DOES              WHY IT FAILS
─────────────────────────────────────────────────────────────────
Rule-based          Fixed rules per device    Cannot adapt to new
                    (e.g. cap ON if V < 0.97)  load patterns or faults

OPF solver          Optimise full AC model    Too slow for real-time
(classical)         → exact optimal solution   (seconds to minutes per solve)

Classical DRL       Learn policy from         Works but needs millions
(SAC, PPO)          experience                 of parameters; hard to
                                               retrain when grid changes

Supervised ML       Learn from solved         Needs expensive labelled
                    examples                   data (solver must run
                                               thousands of times first)
```

**The gap:** We need something that is:
- Fast at inference (real-time)
- Does not need labelled data
- Uses few parameters (easy to update/retrain)
- Handles new grids without full retraining

---

## THE SOLUTION SPACE

### Where QE-SAC Sits

```
                    NEEDS LABELS?
                    Yes              No
                    ──────────────────────────────────
LARGE MODEL         Supervised ML    Classical DRL (SAC)
                    (Pineda 2021)     (baseline)

SMALL MODEL         —                QE-SAC  ← HERE
                                     (this work)
```

QE-SAC is the only approach that is simultaneously:
- Label-free (learns from environment interaction)
- Compact (~5K parameters vs ~900K for classical SAC)
- Physics-aware (CAE adapts to grid dynamics)
- Quantum-ready (VQC runs on real quantum hardware when available)

---

### The QE-SAC Solution — How It Works

```
PROBLEM                          SOLUTION COMPONENT
────────────────────────────────────────────────────────────────

State is high-dimensional        Co-Adaptive Autoencoder (CAE)
(hundreds of voltage/load        Compresses grid state → 8 numbers
 readings at every timestep)     Retrains every 500 steps to stay
                                 aligned with current conditions

8 numbers must drive             Variational Quantum Circuit (VQC)
good decisions                   8 qubits encode the 8 latent values
                                 Quantum superposition explores all
                                 action combinations simultaneously
                                 Only 16 trainable parameters

Decisions are discrete           Linear + Softmax output head
(tap positions, ON/OFF)          Maps VQC output → action probabilities
                                 SAC entropy maximisation ensures
                                 exploration of all device combinations

Learning signal is sparse        Twin Q-critics (classical MLP)
(reward only at end of step)     Estimate long-term value of actions
                                 Guide the quantum actor toward
                                 high-reward regions
```

---

## THE PROPOSED RESEARCH

### What the Paper Proved (Done)

- QE-SAC matches classical SAC reward on the **same feeder it trained on**
- Zero voltage violations on 13-bus and 123-bus
- 185× fewer parameters than classical SAC
- Robust to quantum noise up to λ = 1%

### What the Paper Did NOT Prove (Your Opportunity)

```
OPEN QUESTION                    WHY IT MATTERS FOR DEPLOYMENT
──────────────────────────────────────────────────────────────────
Does it generalise to new        Every utility has different feeders.
feeders it never trained on?     You cannot retrain from scratch each time.

Can voltage safety be            Utilities need GUARANTEED safe operation.
guaranteed (not just hoped       A soft penalty is not good enough for
for via reward shaping)?         real grid deployment.

Does grid topology help          The MLP CAE ignores that buses are
the compression?                 connected. A GNN encoder that knows
                                 the graph structure may compress better.
```

---

### The Proposed Solution — Three Contributions

---

#### Contribution 1: Generalisation via Transfer Learning

**Problem:** QE-SAC is trained and tested on the same feeder. In practice,
a utility has dozens of feeders of different sizes.

**Solution:**
```
PHASE 1: Train QE-SAC on 13-bus feeder
         → Learn general voltage control behaviour

PHASE 2: Freeze VQC weights
         Only allow CAE to retrain (fast, cheap)

PHASE 3: Deploy on 34-bus and 123-bus
         CAE adapts compression to new feeder state space
         VQC reuses learned quantum policy
```

**Hypothesis:** The VQC learns a general mapping from
"compressed voltage state" → "control action" that is feeder-agnostic.
Only the compression (CAE) needs to adapt to a new feeder.

**What you measure:**
- Reward on unseen feeder (higher = better transfer)
- Voltage violations on unseen feeder (target = 0)
- How many CAE update steps needed to adapt (fewer = better)

---

#### Contribution 2: Hard Safety Guarantee via Constrained SAC

**Problem:** Voltage limits are in the reward as a soft penalty.
The agent can still violate them if it gains reward elsewhere.

**Solution — Lagrangian Constrained SAC:**

```
Standard SAC:
  Maximise  E[reward]

Constrained SAC (your proposal):
  Maximise  E[reward]
  Subject to  E[voltage violations per step] ≤ 0

How it works:
  → Add a Lagrange multiplier λ (starts at 0)
  → Every time voltage is violated, λ increases
  → λ makes the voltage penalty grow automatically
  → Training stops when violations = 0
  → λ is the "self-tuning safety dial"
```

**What you measure:**
- Does constraint satisfaction hold? (violations must = 0)
- What is the reward cost of the constraint? (small = good)
- Does it still generalise to new feeders?

---

#### Contribution 3: GNN Encoder to Replace MLP CAE

**Problem:** The MLP CAE treats the grid state as a flat vector.
It does not know that bus 3 is connected to bus 4.
This means it has to learn grid topology from data alone — inefficient.

**Solution — Graph Neural Network encoder:**

```
CURRENT CAE (MLP):
  [V_1, P_1, Q_1, V_2, P_2, Q_2, ..., V_13, P_13, Q_13]
       ↓  flat vector, no topology
  [z_1, z_2, ..., z_8]

PROPOSED GNN encoder:
  Bus features:  [V_i, P_i, Q_i] at each node
  Edge features: [r_ij, x_ij] on each branch
       ↓  message passing: each bus aggregates neighbour info
  [z_1, z_2, ..., z_8]  ← topology-aware compression
```

**Why better:**
- Compression reflects physical connectivity
- Automatically scales to any feeder size (13 → 34 → 123-bus same model)
- Enables zero-shot transfer — GNN has never seen a feeder but
  understands the graph structure immediately

**What you measure:**
- Sample efficiency (convergence speed vs MLP CAE)
- Transfer performance (same GNN on 13/34/123-bus)
- Parameter count (GNN encoder should be smaller than MLP CAE)

---

## THE FULL PROPOSED SYSTEM

```
┌─────────────────────────────────────────────────────────────┐
│                    QE-SAC+ (Your Version)                   │
│                                                             │
│  Grid State                                                 │
│  (any feeder size)                                          │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────┐                                            │
│  │ GNN Encoder │  topology-aware compression                │
│  │ (new)       │  scales to any feeder                      │
│  └─────────────┘                                            │
│       │  8-dim latent                                       │
│       ▼                                                     │
│  ┌─────────────┐                                            │
│  │  8-qubit    │  quantum policy                            │
│  │  VQC        │  16 parameters                             │
│  └─────────────┘                                            │
│       │  action probabilities                               │
│       ▼                                                     │
│  ┌─────────────────────────────┐                            │
│  │ Constrained SAC             │  safety guarantee          │
│  │ (Lagrangian, new)           │  E[V_viol] ≤ 0             │
│  └─────────────────────────────┘                            │
│       │                                                     │
│       ▼                                                     │
│  Safe, generalizable voltage control                        │
│  on any feeder, with zero violations guaranteed             │
└─────────────────────────────────────────────────────────────┘
```

---

## COMPARISON TABLE — What You Add vs the Paper

| Aspect | Paper (QE-SAC) | Your Work (QE-SAC+) |
|---|---|---|
| Encoder | MLP CAE | GNN encoder |
| Safety | Soft penalty | Hard constraint (Lagrangian) |
| Generalisation | Same feeder only | Train once, deploy anywhere |
| Evaluation | 13-bus + 123-bus (same feeder) | Cross-feeder transfer test |
| Parameter count | ~5K actor | ~5K actor (same VQC) |
| Voltage violations | ~0 (not guaranteed) | 0 (mathematically guaranteed) |
| Topology awareness | None | Full graph structure |

---

## WHAT TO DO — Step by Step

```
WEEK 1–2   Implement Constrained SAC (Lagrangian)
           → Modify trainer.py to add λ multiplier
           → Train on 13-bus, verify 0 violations

WEEK 3–4   Implement GNN encoder
           → Replace CAE in qe_sac_policy.py with PyG GNN
           → Verify same latent dim (8), same VQC unchanged

WEEK 5–6   Transfer experiment
           → Train on 13-bus, evaluate on 34-bus and 123-bus
           → Compare: paper QE-SAC vs your QE-SAC+

WEEK 7–8   Full comparison table
           → Classical SAC, QC-SAC, QE-SAC (paper), QE-SAC+ (yours)
           → Metrics: reward, violations, params, transfer gap

WEEK 9–10  Write results and analysis
           → Your contribution: safe + generalizable quantum VVC
```

---

## EMPIRICAL HYPOTHESES (from baseline results, 2026-04-01)

These hypotheses emerged from the 13-bus baseline run (3 seeds, 50K steps).
They are **testable claims** to verify and report in the paper.

---

### H1 — QE-SAC converges more stably than Classical SAC across random seeds

**Observation:** QE-SAC reward std = ±0.66 vs Classical SAC std = ±4.42 (6.7× lower).
The original paper reports only mean reward — it never discusses seed variance.

**Hypothesis:** The VQC's tiny parameter space (16 params) acts as implicit
regularisation — it cannot overfit to seed-specific noise, forcing a general policy.
Classical SAC's large MLP actor is more sensitive to random initialisation.

**How to verify:** Run 10 seeds in Task 4.2. Report mean ± std for all agents.
If the gap holds at 10 seeds, this is a standalone novel finding.

**Expected paper claim:**
> *"QE-SAC exhibits significantly lower training variance than Classical SAC
> across random seeds (±0.66 vs ±4.42), suggesting the constrained VQC
> parameter space acts as implicit regularisation."*

---

### H2 — CAE co-adaptation drives stability, not the VQC alone

**Hypothesis:** If the CAE is frozen after warmup (no retraining every 500 steps),
training variance increases and reward degrades. This would prove the co-adaptation
mechanism — not the quantum circuit alone — is responsible for stable convergence.

**How to verify:** Add `freeze_cae=True` option to `QESACAgent`. Run 10 seeds with
frozen CAE vs co-adaptive CAE. Compare mean reward and std across seeds.

**Expected paper claim:**
> *"Ablation shows that CAE co-adaptation is the primary source of QE-SAC's
> training stability — freezing the encoder increases reward variance by X×."*

---

### H3 — Lagrangian constraint reduces training variance further

**Hypothesis:** Constrained SAC (Task 2) removes the reward-vs-violation tradeoff.
With one fewer competing objective, the optimizer follows a more consistent gradient
direction → even lower variance across seeds than unconstrained QE-SAC.

**How to verify:** Compare std of reward across 10 seeds:
unconstrained QE-SAC vs constrained QE-SAC+ (Task 4.5 ablation).

**Expected paper claim:**
> *"The Lagrangian safety constraint reduces reward variance by X× compared to
> unconstrained QE-SAC, as the optimizer no longer faces a competing
> reward-violation objective."*

---

### Summary table

| Hypothesis | Source | Task to verify | Cost |
|---|---|---|---|
| H1: VQC → lower variance | Baseline results (already observed) | Task 4.2 (10 seeds) | Free — just run more seeds |
| H2: CAE drives stability | Architectural reasoning | Task 4.5 ablation | Add `freeze_cae` flag |
| H3: Constraint → lower variance | Theoretical prediction | Task 4.5 ablation | Already planned |

---

## THE ONE CONTRIBUTION STATEMENT

> **This work proposes QE-SAC+: a safe and generalizable quantum
> reinforcement learning agent for Volt-VAR control that (1) guarantees
> zero voltage violations via Lagrangian constraint satisfaction,
> (2) transfers to unseen feeders via a topology-aware GNN encoder,
> (3) maintains the parameter efficiency of the original QE-SAC
> with identical VQC architecture, and (4) demonstrates that the
> constrained VQC parameter space produces significantly more stable
> training convergence than classical deep RL across random seeds.**

---

## FILES THAT NEED TO BE BUILT

```
src/qe_sac/
├── constrained_sac.py    ← NEW: Lagrangian SAC with voltage constraint
├── gnn_encoder.py        ← NEW: GNN replacement for MLP CAE
├── qe_sac_plus.py        ← NEW: QE-SAC+ agent (GNN + VQC + constrained SAC)
└── transfer_eval.py      ← NEW: cross-feeder evaluation utilities

(all existing files stay unchanged — this is additive)
```
