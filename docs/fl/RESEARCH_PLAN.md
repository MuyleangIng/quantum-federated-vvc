# QE-SAC-FL — Extended Research Plan

**Project:** Federated Quantum RL for Volt-VAR Control  
**Researcher:** Ing Muyleang — Pukyong National University, QCL  
**Plan date:** 2026-04-01  
**Target venue:** IEEE Transactions on Smart Grid / Nature Quantum Information

---

## Overview

This document extends the base three hypotheses (H1–H3) with six additional
research directions (H4–H9). Each hypothesis targets a distinct novel
contribution with clear references showing the gap in existing literature.

```
Experiment road-map
═══════════════════════════════════════════════════════════
DONE    H1  Federated VQC > Local-only (after alignment fix)
DONE    H2  Faster convergence with federation
DONE    H3  ✅ 395× communication reduction vs classical FL
───────────────────────────────────────────────────────────
PLAN    H4  Barren plateau: FL regularises gradient landscape
PLAN    H5  Personalised FL: FL init + local fine-tune wins
PLAN    H6  Partial participation: robustness to client dropout
PLAN    H7  Transfer learning: global VQC → new unseen feeder
PLAN    H8  Non-IID severity: how bad can data skew get?
PLAN    H9  Round efficiency: where is the FL breakeven point?
═══════════════════════════════════════════════════════════
```

---

## H4 — Barren Plateau Regularisation via Federation

### Claim
Federated averaging over diverse clients regularises the VQC weight landscape,
reducing barren plateau risk compared to single-client training.

### Motivation
Barren plateaus (gradients exponentially vanishing with qubit count) are the
central trainability problem in quantum ML. Classical FL research shows that
gradient diversity across clients can act as implicit regularisation [1].
No paper has studied whether FL *reduces* barren plateaus for quantum policies.

### Measurement
- VQC gradient norm ||∇θ|| per round, per condition (local vs FL vs aligned FL)
- Gradient variance across clients before vs after FedAvg
- Plot: gradient norm vs round for all 3 clients × 3 conditions

### Expected result
Aligned FL should show *higher* sustained gradient norms than local-only,
because diverse grid topologies provide gradient signals from different
directions of the loss landscape.

### Paper references
- [1] Li et al. (2020). "Federated Optimization in Heterogeneous Networks."
  ICLR 2020. (FedProx — heterogeneous gradient analysis)
- [2] Cerezo et al. (2021). "Cost function dependent barren plateaus in
  shallow parametrized quantum circuits." Nature Comm. 12, 1791.
- [3] McClean et al. (2018). "Barren plateaus in quantum neural network
  training landscapes." Nature Comm. 9, 4812.
- [4] Holmes et al. (2022). "Connecting Ansatz Expressibility to Gradient
  Magnitudes and Barren Plateaus." PRX Quantum 3, 010313.

### Novel gap
No existing work has studied the interaction between FL gradient diversity
and barren plateau occurrence in quantum policy networks. H4 is the first
paper to ask: *does federation help quantum RL escape barren plateaus?*

---

## H5 — Personalised Federated Quantum RL

### Claim
A personalised strategy — aligned FL for N rounds then local fine-tuning
for M steps — outperforms both (a) local-only and (b) pure aligned FL,
because each client adapts the shared quantum policy to its own feeder.

### Motivation
Personalised FL (pFL) is one of the hottest areas of FL research [5][6][7].
In classical settings, a global model rarely beats a personalised one on
heterogeneous clients. The question is whether pFL transfers to quantum
policies. The aligned FL architecture is specifically suited for this:
LocalEncoder adapts privately while SharedHead+VQC capture global knowledge.

### Measurement
For each client, after 50 aligned FL rounds:
  - Branch A: freeze SharedHead, fine-tune VQC locally for 5000 steps
  - Branch B: unfreeze all, fine-tune entire actor locally for 5000 steps
  - Branch C: no fine-tune (pure aligned FL)
  
Compare final reward of A, B, C vs local-only baseline.

### Expected result
Branch B (personalise everything) should beat local-only by >5% for at least
2/3 clients, because the FL warm-start is better than random initialisation.
Branch A (VQC only) may be worse — forcing the VQC to adapt while keeping the
shared head frozen may create a new latent space conflict.

### Paper references
- [5] Fallah et al. (2020). "Personalized Federated Learning with Theoretical
  Guarantees: A Model-Agnostic Meta-Learning Approach." NeurIPS 2020.
- [6] Dinh et al. (2020). "pFedMe: Personalized Federated Learning with
  Moreau Envelopes." NeurIPS 2020.
- [7] Tan et al. (2022). "Towards Personalized Federated Learning." IEEE TNNLS.
- [8] Collins et al. (2021). "Exploiting Shared Representations for
  Personalized Federated Learning." ICML 2021.

### Novel gap
H5 is the first study of personalised FL for quantum RL agents. The
LocalEncoder/SharedHead split in SOLUTION_001 provides a natural interface
for personalisation that has no equivalent in classical FL architectures.

---

## H6 — Partial Participation Robustness

### Claim
QE-SAC-FL-Aligned is robust to client dropout: randomly excluding one client
per round still produces better-than-local-only reward after 50 rounds.

### Motivation
In real utility deployment, one client may be offline (planned maintenance,
communication failure, regulatory hold). The FL system must be robust to
partial participation. Classical FL papers show FedAvg degrades gracefully
with dropout [9]. Whether this holds for quantum FL (with only 16 VQC params)
is unknown — small models may be more sensitive to missing gradient signal.

### Measurement
- Run aligned FL where each round randomly selects 2/3 clients
- Compare final rewards to: (a) local-only, (b) full aligned FL (3/3 clients)
- Run 5 seeds to get mean ± std for statistical significance

### Expected result
Partial participation (2/3 clients per round) should still outperform
local-only on at least 2/3 clients. The 16 VQC params are a compact
model — FedAvg on 2 clients should still capture cross-grid knowledge.
Expect ~5–10% degradation vs full participation.

### Paper references
- [9] McMahan et al. (2017). "Communication-Efficient Learning of Deep
  Networks from Decentralized Data." AISTATS 2017. (original FedAvg)
- [10] Yang et al. (2021). "Achieving Linear Speedup with Partial Worker
  Participation in Non-IID Federated Learning." ICLR 2021.
- [11] Gu et al. (2021). "Fast Federated Learning in the Presence of
  Arbitrary Device Unavailability." NeurIPS 2021.

### Novel gap
H6 tests reliability of quantum FL under realistic deployment conditions.
First study of client dropout robustness for quantum policy federation.

---

## H7 — Transfer Learning: Global VQC → Unseen Feeder

### Claim
A VQC pre-trained via aligned FL on 3 feeders reaches good performance on
a 4th unseen feeder faster (fewer steps) than training from scratch.

### Motivation
Transfer learning is central to practical ML deployment. Utilities regularly
build new feeders or acquire networks through mergers — they need to train
new agents quickly. Classical transfer learning for RL is well-studied [12][13].
For quantum RL, the VQC has only 16 parameters — it is unclear whether
these 16 params encode transferable knowledge or are too compressed.

### Measurement
- Train aligned FL on 13-bus, 34-bus, 123-bus (50 rounds)
- Take global VQC + global SharedHead weights
- Init new agent for a MODIFIED 13-bus feeder (different load profile, seed=999)
- Fine-tune 10K steps — compare steps to reward threshold vs random init

### Expected result
Transfer should provide a better starting point: expect 20–40% fewer steps
to reach the same reward threshold. The SharedHead encodes a general
"how to read a feeder's latent state" — this should transfer across feeder sizes.

### Paper references
- [12] Tan et al. (2018). "A Survey on Deep Transfer Learning." ICANN 2018.
- [13] Zhu et al. (2023). "Transfer Learning in Deep Reinforcement Learning:
  A Survey." IEEE TPAMI.
- [14] Yosinski et al. (2014). "How transferable are features in deep
  neural networks?" NeurIPS 2014.
- [15] Weng et al. (2024). "Transfer Learning for Power System Volt-VAR
  Optimization." IEEE PES. (classical transfer, no quantum)

### Novel gap
H7 is the first demonstration of quantum policy transfer learning for
power grid control. The compactness of VQC (16 params) may make quantum
policies BETTER transfer vehicles than classical 110K-param actors.

---

## H8 — Non-IID Severity Study

### Claim
QE-SAC-FL-Aligned degrades gracefully as data heterogeneity (feeder size
disparity, load shape difference) increases, while classical FL degrades faster.

### Motivation
Non-IID data is the central challenge of FL [16][17]. The three feeders
(13, 34, 123 bus) already have very different observation dimensions and
topology. QLSI (ISSUE_001) is related to but distinct from classical non-IID
drift — fixing QLSI does not fix non-IID reward divergence.

### Measurement
- Construct 5 synthetic non-IID levels by scaling load profiles differently
  per client (severity: 0% → 50% load shift)
- At each severity, measure: final reward gap between clients, convergence speed
- Compare: aligned FL vs FedProx (classical regularisation) vs local-only

### Expected result
Aligned FL should outperform classical FL at all severity levels because
QLSI is the dominant failure mode at high severity. At low severity, classical
and quantum FL may be similar.

### Paper references
- [16] Zhao et al. (2018). "Federated Learning with Non-IID Data." arXiv:1806.00582.
- [17] Li et al. (2021). "FedBN: Federated Learning on Non-IID Features via
  Local Batch Normalization." ICLR 2021.
- [18] Karimireddy et al. (2020). "SCAFFOLD: Stochastic Controlled Averaging
  for Federated Learning." ICML 2020.

### Novel gap
First systematic non-IID study for quantum federated RL in power systems.
Quantifies the boundary between QLSI (structural) and classical FL drift.

---

## H9 — Round Efficiency: Where is the FL Breakeven?

### Claim
Aligned FL surpasses local-only performance within the first 10 rounds (10,000
steps per client) — earlier than classical FL achieves breakeven.

### Motivation
FL is only worth deploying if the communication overhead pays off quickly.
Classical FL breakeven analysis shows 20–100 rounds typically needed [19][20].
With only 16–288 parameters federated, quantum FL should achieve breakeven
faster. Knowing the breakeven round is critical for deployment planning.

### Measurement
- Per client, per round: aligned_FL_reward[round] > local_only_reward[round]?
- Find first round where aligned FL is better for ALL 3 clients simultaneously
- Compute: communication cost at breakeven = breakeven_round × bytes_per_round

### Expected result
Breakeven within 5–15 rounds for at least 2/3 clients.
Classical FL breakeven (if run) would need 30–50+ rounds for same reward level.
This proves quantum FL gives faster knowledge transfer per byte.

### Paper references
- [19] Kairouz et al. (2019). "Advances and Open Problems in Federated
  Learning." Found. Trends Mach. Learn. 14(1–2), 2021.
- [20] Li et al. (2020). "On the Convergence of FedAvg on Non-IID Data."
  ICLR 2020.
- [21] Wang et al. (2022). "A Field Guide to Federated Optimization." arXiv.

### Novel gap
H9 quantifies FL breakeven for quantum policies. The result (in bytes, not
just rounds) provides a deployability argument unique to quantum FL.

---

## Priority and Timeline

```
Priority 1 (run immediately — uses existing code):
  H4  Barren plateau analysis         → uses existing grad_norm logs
  H9  Round breakeven                 → uses existing reward logs
  H1  Aligned FL validation           → already implemented

Priority 2 (need new trainer methods — 1–2 days):
  H5  Personalised FL                 → add run_personalized()
  H6  Partial participation           → add run_partial_participation()

Priority 3 (need new env / longer runs — 1 week):
  H7  Transfer to unseen feeder       → new modified 13-bus env
  H8  Non-IID severity study          → synthetic load scaling
```

---

## Novel Contributions Summary

| # | Contribution | First in literature? |
|---|---|---|
| ISSUE_001 | Quantum Latent Space Incompatibility (QLSI) | ✅ Yes |
| SOLUTION_001 | Shared Encoder Head to fix QLSI | ✅ Yes |
| H3 | 395× communication reduction | ✅ Yes |
| H4 | FL reduces barren plateau risk | ✅ Yes |
| H5 | Personalised quantum FL for grid control | ✅ Yes |
| H6 | Quantum FL robustness to client dropout | ✅ Yes |
| H7 | Quantum policy transfer for power grids | ✅ Yes |
| H8 | Non-IID severity analysis for quantum FL | ✅ Yes |
| H9 | FL breakeven in bytes (quantum vs classical) | ✅ Yes |

A paper with 3+ of these is a strong IEEE T-SG / Nature Quantum paper.
A paper with all 9 is a survey-level contribution.
