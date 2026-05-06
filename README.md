# Federated Quantum Reinforcement Learning for Multi-Utility Volt-VAR Control

**Identifying and Solving Structural Failure Modes in Quantum Federated RL**

**Researcher:** Ing Muyleang
**Institution:** Pukyong National University — Quantum Computing Laboratory
**Status:** Experiments complete — paper writing phase
**Target venue:** IEEE Transactions on Smart Grid (IF 8.9)

---

## What This Project Does

Three competing power utilities each own a private distribution feeder and
**cannot share raw grid data**. This project trains a shared quantum
reinforcement learning policy (VQC — Variational Quantum Circuit) that
improves Volt-VAR Control (VVC) for all three utilities simultaneously —
without any data ever leaving each company.

During implementation, **three novel failure modes** of quantum federated
learning were discovered, named, and solved experimentally.

---

## Three Novel Discoveries

### 1. heterogeneous FL problem — Quantum Latent Space Incompatibility
Naive FedAvg on quantum RL agents makes every client **worse** than
training alone. Each client's private encoder independently learns
incompatible latent representations — after averaging, the VQC receives
meaningless inputs from every client.

> *No existing FL paper has identified this failure mode.*

### 2. CSA — Client Size Asymmetry
Even after fixing heterogeneous FL problem, pure aligned FL benefits small feeders early
and large feeders late — never all clients simultaneously. Gradient
scale imbalance from different observation dimensions drives the
SharedHead to favour different clients at different training rounds.

> *Classical FL heterogeneity research studies data content, not obs_dim scale.*

### 3. PAD — Partial Alignment Drift
Classical FedAvg is robust to partial client participation (McMahan 2017).
Quantum FL with the aligned architecture is not — even 1/3 client dropout
causes all clients to perform worse than training alone.

> *Partial participation robustness does not transfer to split-encoder architectures.*

---

## Main Result

**Personalised Federated Quantum RL: FL warm-start + local fine-tuning**

| Client | Local only | Personalised FL | Improvement |
|---|---|---|---|
| 13-bus (42-dim obs) | −331.4 | **−165.0** | **+50.2%** |
| 34-bus (105-dim obs) | −65.5 | **−15.2** | **+76.8%** |
| 123-bus (372-dim obs) | −5364.4 | **−4034.5** | **+24.8%** |

All three utilities improve simultaneously with **395× less communication**
than federated classical SAC.

---

## Architecture

```
obs → [ LocalEncoder ] → [ SharedEncoderHead ] → [ VQC ] → action
          PRIVATE              FEDERATED            FEDERATED
       stays on client        264 params            16 params
       (feeder-specific)      same for all          same for all

Total federated per round: 280 params = 1,120 bytes
Classical federated SAC:   110,724 params = 443,000 bytes
Communication reduction:   395× to 6,920×
```

---

## Project Structure

```
power-system/
├── src/
│   ├── qe_sac/                   # Base quantum RL agent (QE-SAC)
│   │   ├── vqc.py                # 8-qubit VQC (PennyLane)
│   │   ├── qe_sac_policy.py      # QESACAgent
│   │   └── trainer.py            # Local training loop
│   └── qe_sac_fl/                # Federated quantum RL (this work)
│       ├── fed_config.py         # All hyperparameters
│       ├── federated_trainer.py  # FederatedTrainer + all conditions
│       ├── aligned_encoder.py    # LocalEncoder, SharedEncoderHead, FedAvg
│       ├── aligned_agent.py      # AlignedQESACAgent
│       ├── env_34bus.py          # 34-bus and 123-bus environments
│       └── docs/                 # Research documentation
│           ├── FULL_RESEARCH_COMPENDIUM.md   ← complete technical reference
│           ├── HYPOTHESES.md
│           ├── RESULTS_SUMMARY.md
│           ├── ISSUE_001_LATENT_INCOMPATIBILITY.md   ← heterogeneous FL problem
│           ├── ISSUE_002_BARREN_PLATEAU.md
│           ├── ISSUE_003_PARTIAL_PARTICIPATION.md    ← PAD
│           ├── FINDING_001_CLIENT_SIZE_TRADEOFF.md   ← CSA
│           └── SOLUTION_001_ALIGNED_FEDERATION.md
├── notebooks/
│   └── qe_sac_fl_experiment.ipynb  # Full experiment notebook (56 cells)
├── artifacts/
│   └── qe_sac_fl/                  # All saved results and figures
├── PROFESSOR_REPORT.md             # Advisor meeting report
├── ADVISOR_BRIEFING.md             # Pre-meeting 1-page brief
└── QE_SAC_FL_PROPOSAL.md           # Research proposal
```

---

## Quick Start

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Run the full experiment (3 GPUs recommended, ~3-4 hours)
cd notebooks
jupyter notebook qe_sac_fl_experiment.ipynb

# 3. Quick smoke test (CPU, ~5 minutes)
python3 -c "
from src.qe_sac_fl.fed_config import quick_config
from src.qe_sac_fl.federated_trainer import FederatedTrainer
trainer = FederatedTrainer(quick_config())
results = trainer.run_all_conditions()
"
```

## Reproducing Paper Results

```python
from src.qe_sac_fl.fed_config import paper_config
from src.qe_sac_fl.federated_trainer import FederatedTrainer

cfg = paper_config()          # 50 rounds × 1,000 steps, 3 GPUs
trainer = FederatedTrainer(cfg)
results = trainer.run_all_conditions()
# Saves all results to artifacts/qe_sac_fl/
```

---

## Hardware

Experiments run on 3× NVIDIA RTX 4090 (one GPU per client).
Each client trains in parallel — wall time ~3-4 hours for full 50-round run.

Quantum simulation: PennyLane `default.qubit` (CPU statevector).
No real quantum hardware required.

---

## Key Configuration

All hyperparameters in `src/qe_sac_fl/fed_config.py`:

| Parameter | Value | Notes |
|---|---|---|
| VQC qubits | 8 | Do not change — keeps comparison with base QE-SAC valid |
| Federated rounds | 50 (standard), 200 (extended) | |
| Local steps per round | 1,000 | 50K total steps per client |
| Batch size | 512 | GPU-efficient |
| Learning rate | 3e-4 | Standard SAC |
| Federated params | 280 | SharedEncoderHead (264) + VQC (16) |

---

## Results Summary

```
Condition                  13-bus    34-bus    123-bus    All > local?
──────────────────────────────────────────────────────────────────────
Local only (baseline)      -331.4     -65.5    -5364.4         —
Unaligned FL               -336.6     -69.6    -5420.5        NO   ← heterogeneous FL problem
Aligned FL, 50 rounds      -326.3     -85.0    -5402.5        NO   ← CSA
Aligned FL, 200 rounds     -339.5     -69.3    -5251.4        NO   ← CSA reversal
Partial FL (2/3 clients)   -341.4     -79.8    -5402.9        NO   ← PAD
Personalised FL            -165.0     -15.2    -4034.5       YES   ← SOLUTION
──────────────────────────────────────────────────────────────────────
```

---

## Citation

> Muyleang, I. (2026). *Privacy-Preserving Federated Quantum Reinforcement
> Learning for Multi-Utility Volt-VAR Control: Identifying and Solving
> Quantum Latent Space Incompatibility.* Pukyong National University,
> Quantum Computing Laboratory.

---

## License

Research code — Pukyong National University, Quantum Computing Laboratory.
