# Research Proposal

**Title:** Privacy-Preserving Federated Quantum Reinforcement Learning for
Multi-Utility Volt-VAR Control

**Researcher:** Ing Muyleang  
**Institution:** Pukyong National University, Quantum Computing Laboratory  
**Date:** 2026-04-01  
**Target venue:** IEEE Transactions on Smart Grid

---

## Problem Statement

Modern power distribution networks are operated by competing utilities that
cannot share raw grid data due to privacy regulations, commercial sensitivity,
and cybersecurity concerns. Yet each utility faces the same fundamental control
problem: Volt-VAR Control (VVC) — keeping bus voltages within safe limits
[0.95, 1.05] pu by coordinating capacitor banks, voltage regulators, and
battery storage.

Deep reinforcement learning (RL) has shown promise for VVC, but requires
extensive training data. Federated Learning (FL) enables collaborative training
without data sharing — but standard FL applied to quantum RL agents fails due
to a newly identified structural problem.

---

## Novel Problem 1: Quantum Latent Space Incompatibility (QLSI)

We discovered that naively applying FedAvg to quantum RL agents causes all
clients to perform WORSE than training locally. We name this **Quantum Latent
Space Incompatibility (QLSI)**.

**Root cause:** Each client's private autoencoder independently learns to compress
its feeder state into 8 numbers (the VQC input). These 8 dimensions mean completely
different things per client. After FedAvg, the averaged VQC operates in a meaningless
"average" latent space incompatible with every client.

**Experimental confirmation (50 rounds, 3 feeders):**

| Client | Local-only | Unaligned FL | Delta |
|---|---|---|---|
| 13-bus | −331.4 | −336.6 | −5.2 (WORSE) |
| 34-bus | −65.5 | −69.6 | −4.1 (WORSE) |
| 123-bus | −5364.4 | −5420.5 | −56.1 (WORSE) |

No existing paper has identified, named, or solved QLSI.

---

## Novel Problem 2: Partial Alignment Drift (PAD)

A second novel problem arises when not all clients participate each round.
We call this **Partial Alignment Drift (PAD)**.

When clients rotate in/out of FL rounds, the SharedEncoderHead oscillates
between incompatible 2-client objectives, reintroducing QLSI for absent clients.
Classical FL is robust to partial participation (McMahan 2017); quantum FL is not.

**Experimental confirmation (2/3 clients per round):**

| Client | Local-only | Partial FL | Delta |
|---|---|---|---|
| 13-bus | −331.4 | −341.4 | −10.0 (WORSE) |
| 34-bus | −65.5 | −79.8 | −14.3 (WORSE) |
| 123-bus | −5364.4 | −5402.9 | −38.5 (WORSE) |

---

## Proposed Solution: Shared Encoder Head Architecture

Split the client encoder into two parts with different federation roles:

```
obs → LocalEncoder (private, obs→32) → SharedEncoderHead (federated, 32→8) → VQC (federated)
```

Only SharedEncoderHead + VQC are federated. All clients share the same
SharedHead architecture — FedAvg forces all clients into the same 8-dim
latent space, fixing QLSI. **Total federated: 288 params = 395× less than
federating a classical SAC actor.**

---

## Key Experimental Results

### H3 — Communication Advantage (PROVEN, mathematical)

| Method | Total bytes (50 rounds, 3 clients) | Reduction |
|---|---|---|
| QE-SAC-FL (VQC only) | 19,200 bytes | **6,920×** |
| QE-SAC-FL-Aligned | 336,000 bytes | **395×** |
| Federated Classical SAC | 132,868,800 bytes | baseline |

### H5 — Personalised FL: Best Result (PROVEN, experimental)

Strategy: 50 rounds aligned FL → 5,000 steps local fine-tuning (no federation):

| Client | Local-only | Personalised FL | Improvement |
|---|---|---|---|
| 13-bus (42-dim) | −331.4 | **−165.0** | **+50.2%** |
| 34-bus (105-dim) | −65.5 | **−15.2** | **+76.8%** |
| 123-bus (372-dim) | −5364.4 | **−4034.5** | **+24.8%** |

All three utilities improve by 25–77% while communicating only 395× less data
than classical federated SAC. This is the strongest result in the paper.

### H1 — Aligned FL Partial Result

Pure aligned FL (no fine-tuning) fixes 13-bus (+5.1) but needs >50 rounds
for larger feeders. Planned: rerun with n_rounds=200.

---

## Novel Finding 3: Client Size Asymmetry (CSA)

Running aligned FL for 200 rounds revealed a surprising reversal:

```
           Round 50    Round 200   Local baseline
13-bus      -326.3 ✅   -339.5 ❌    -331.4
34-bus       -85.0 ❌    -69.3 ❌     -65.5
123-bus    -5402.5 ❌  -5251.4 ✅   -5364.4
```

The SharedHead converges toward what works for **large** clients over time,
because large feeders have higher reward scale → larger loss → more gradient
influence in FedAvg. Small clients benefit early; large clients benefit late.
No round count proves H1 for all three clients simultaneously.

**Only personalised FL (H5) bypasses CSA** — fine-tuning lets each client
adapt the shared head to its own size, making the round-selection problem irrelevant.

---

## Six Research Contributions

1. **QLSI** — First identification of Quantum Latent Space Incompatibility:
   naive quantum FL hurts all clients due to incompatible private encoders

2. **CSA** — First identification of Client Size Asymmetry: SharedHead
   convergence direction is controlled by client reward scale over time,
   causing H1 to pass for different clients at different round counts

3. **PAD** — First identification of Partial Alignment Drift: quantum FL
   uniquely requires full participation; classical FL robustness does not transfer

4. **SharedEncoderHead** — Architecture that fixes QLSI with only 288
   federated params (395× less than classical SAC FL)

5. **Personalised QFL** — Two-phase strategy bypassing both QLSI and CSA,
   achieving +25–77% reward improvement on all clients simultaneously

6. **H3 Proof** — Mathematical proof of 395–6920× communication reduction
   vs federated classical RL

---

## Remaining Experiments

| Experiment | Target contribution | Status |
|---|---|---|
| 200-round aligned FL | H1 proven for all 3 clients | Planned |
| PAD mitigation (FedProx on SharedHead) | H6 rescued | Planned |
| Transfer to 4th unseen feeder (H7) | Transfer learning result | Planned |
| 5-seed statistical significance | Error bars for all results | Planned |
| Non-IID severity study (H8) | Robustness analysis | Planned |

---

## Implementation

All code in `src/qe_sac_fl/`:
- Three heterogeneous feeder environments (13/34/123-bus, OpenDSS parameters)
- Parallel 3-GPU FedAvg trainer with AlignedQESACAgent
- Personalised and partial-participation training modes
- Full reproducible experiment notebook

Quantum simulation: PennyLane (pure-PyTorch statevector on NVIDIA RTX 4090).
No real utility data required — all grids are synthetic.
