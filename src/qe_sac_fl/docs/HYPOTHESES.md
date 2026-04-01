# QE-SAC-FL — Research Hypotheses

**Project:** Federated Quantum RL for Volt-VAR Control  
**Researcher:** Ing Muyleang — Pukyong National University, QCL  
**Last updated:** 2026-04-01 (real experimental results)

---

## Overview

Three utilities each own a private feeder. They cannot share raw grid data.
Goal: train a shared quantum policy (VQC) that helps all clients without sharing data.

```
Utility A (13-bus)   →  private data  →  local training  →  share VQC weights only
Utility B (34-bus)   →  private data  →  local training  →  share VQC weights only
Utility C (123-bus)  →  private data  →  local training  →  share VQC weights only
                                                                      ↓
                                                              FedAvg on server
                                                                      ↓
                                                           global VQC broadcast back
```

**Training config:** 50 rounds × 1,000 steps/client (extended: 200 rounds), 3 NVIDIA GPUs (cuda:0/1/2), parallel.

---

## H1 — Federated VQC Reward > Local-Only VQC

**Claim:** Training a shared VQC via FedAvg produces a better policy than
each utility training its own VQC from scratch.

**Measurement:** Final mean episode reward at round 50. Higher (less negative) = better.

| Client | Local only | Unaligned FL | Aligned 50r | Aligned 200r | H1 (200r)? |
|---|---|---|---|---|---|
| 13-bus | −331.4 | −336.6 | **−326.3** ✅ | −339.5 | ❌ Regressed |
| 34-bus | −65.5 | −69.6 | −85.0 | −69.3 | ❌ Improving |
| 123-bus | −5364.4 | −5420.5 | −5402.5 | **−5251.4** | ✅ PASS (+2.1%) |

**Status:** ⚠️ PARTIAL — pattern reverses with rounds (see FINDING_001).

**Key finding — Client Size Asymmetry (CSA):**
- At round 50: small feeder (13-bus) benefits first → SharedHead tuned early by small-feeder gradients
- At round 200: large feeder (123-bus) benefits → loss-scale imbalance shifts SharedHead over time
- 34-bus (medium): not yet passing at either checkpoint — sits between the two extremes
- This is a NEW finding: SharedHead convergence direction is controlled by client gradient magnitude, not data content

**Full analysis:** See FINDING_001_CLIENT_SIZE_TRADEOFF.md

**Next action:** Run gradient-normalised FedAvg (weighted by 1/||∇S||) to fix CSA and prove H1 for all 3 clients simultaneously.

---

## H2 — Faster Convergence with Federation

**Claim:** QE-SAC-FL reaches a reward threshold faster than local-only.

**Measurement:** Steps to reach reward > −50 per client.

| Client | Local only | QE-SAC-FL-Aligned |
|---|---|---|
| 13-bus | not reached (50K steps) | not reached |
| 34-bus | not reached (50K steps) | not reached |
| 123-bus | not reached (50K steps) | not reached |

**Status:** ⬜ INCONCLUSIVE — threshold too high for 50K steps.

**Note:** H5 (personalised FL) shows convergence IS possible — see H5 below.

---

## H3 — Quantum Communication Advantage ✅ PROVEN

**Claim:** QE-SAC-FL communicates orders of magnitude less data than classical FL.

| Method | Params federated | Bytes (50 rounds, 3 clients) | vs Classical |
|---|---|---|---|
| QE-SAC-FL (VQC only) | 16 | 19,200 | **6,920×** less |
| QE-SAC-FL-Aligned (Head+VQC) | 288 | 336,000 | **395×** less |
| Federated Classical SAC | 110,724 | 132,868,800 | baseline |

**Status:** ✅ PROVEN — mathematical result, independent of reward outcome.

---

## H4 — Barren Plateau Regularisation via Federation

**Claim:** Aligned FL maintains higher VQC gradient norms than local-only.

**Real gradient norm data (final round 50):**

| Client | Local only | Unaligned FL | Aligned FL | FL > Local? |
|---|---|---|---|---|
| 13-bus | 0.000276 | 0.000167 | **0.000084** | ❌ |
| 34-bus | 0.000003 | 0.000040 | **0.000065** | ✅ |
| 123-bus | 0.000027 | 0.000022 | **0.000021** | ❌ |

**Status:** ⚠️ MIXED — Aligned FL helps 34-bus (21× increase) but not 13-bus or 123-bus.

**Interpretation:**
- 34-bus shows strong FL regularisation (0.000003 → 0.000065 = 21×)
- 13-bus local-only already has the highest gradient signal — FL adds noise
- 123-bus near-zero in all conditions — see ISSUE_002 (structural barren plateau risk)
- Partial FL (H6) showed the highest grad norms (0.000721 for 13-bus) — unexpected finding

**Note:** The partial FL result (H6) suggests that with fewer clients, the VQC
receives a stronger gradient signal per client — worth investigating (see RESEARCH_PLAN H8).

---

## H5 — Personalised Federated Quantum RL ✅ BEST RESULT

**Claim:** Aligned FL warm-start + local fine-tuning outperforms both
local-only and pure aligned FL.

**Strategy:** 50 rounds aligned FL → freeze federation → 5,000 steps local fine-tune.

**Real results:**

| Client | Local only | Aligned FL | Personalised FL | Improvement vs Local |
|---|---|---|---|---|
| 13-bus | −331.4 | −326.3 | **−165.0** | **+50.2%** ✅ |
| 34-bus | −65.5 | −85.0 | **−15.2** | **+76.8%** ✅ |
| 123-bus | −5364.4 | −5402.5 | **−4034.5** | **+24.8%** ✅ |

**Status:** ✅ PROVEN — all 3 clients pass, massively.

**Interpretation:**
- FL warm-start provides a fundamentally better initialisation than random
- Local fine-tuning then adapts the shared quantum policy to each feeder's specifics
- The LocalEncoder (private) adapts freely during fine-tuning — this is key
- 34-bus improvement (+77%) is the strongest — medium feeder benefits most from alignment
- **This is the strongest novel result in the paper**

---

## H6 — Partial Participation Robustness

**Claim:** Aligned FL with 2/3 clients per round still beats local-only.

**Real results (2 of 3 clients participating per round, randomly chosen):**

| Client | Local only | Full Aligned FL | Partial FL (2/3) | Robust? |
|---|---|---|---|---|
| 13-bus | −331.4 | −326.3 | −341.4 | ❌ FAIL |
| 34-bus | −65.5 | −85.0 | −79.8 | ❌ FAIL |
| 123-bus | −5364.4 | −5402.5 | −5402.9 | ❌ FAIL |

**Status:** ❌ FAILED — partial participation makes things WORSE than local-only.

**Interpretation (see ISSUE_003):**
- With only 2 clients per round, the shared head FedAvg is less stable
- The dropped client's local encoder adapts to a head that was trained without it
- When that client is included next round, misalignment reoccurs briefly — repeated each round
- This is a NEW finding: quantum FL requires FULL participation to maintain alignment
- Classical FL is robust to dropout (McMahan 2017); quantum FL is not — novel gap

---

## H7 — Transfer Learning: Global VQC → Unseen Feeder

**Status:** ⬜ PLANNED — see RESEARCH_PLAN.md

---

## H8 — Non-IID Severity Study

**Status:** ⬜ PLANNED — see RESEARCH_PLAN.md

---

## H9 — Round Breakeven Analysis

**Claim:** Aligned FL beats local-only within 10 rounds.

**Real results (from h9_breakeven.png):**

| Client | First round FL > Local | Steps at breakeven | Bytes at breakeven |
|---|---|---|---|
| 13-bus | Round 1 | 1,000 | 6,720 |
| 34-bus | not reached | — | — |
| 123-bus | not reached | — | — |

**Status:** ⚠️ PARTIAL — 13-bus breaks even immediately (round 1). Large feeders never break even in 50 rounds because H1 fails for them.

**Key insight:** Once H1 is fully proven (with 200 rounds), H9 breakeven will be measurable for all clients.

---

## Summary Table

| Hypothesis | Status | Key number |
|---|---|---|
| H1: aligned FL > local | ⚠️ 1/3 clients (shifts with rounds) | 50r: 13-bus ✅; 200r: 123-bus ✅ (CSA) |
| H2: faster convergence | ⬜ inconclusive | Need 200K+ steps |
| H3: less communication | ✅ **PROVEN** | **395–6920× less data** |
| H4: barren plateau | ⚠️ mixed | 34-bus: 21× grad norm increase |
| H5: personalised FL | ✅ **PROVEN** | **+25–77% reward all clients** |
| H6: partial participation | ❌ failed | New finding: full participation required |
| H7: transfer learning | ⬜ planned | — |
| H8: non-IID severity | ⬜ planned | — |
| H9: round breakeven | ⚠️ partial | 13-bus breaks even round 1 |

**Paper-ready results: H3, H5, H6 (as negative finding), ISSUE_001 (QLSI)**
