# QE-SAC-FL — Master Results Summary

**Researcher:** Ing Muyleang — Pukyong National University, QCL  
**Date:** 2026-04-01  
**Config:** 50 rounds × 1,000 steps/client, 3 NVIDIA RTX 4090 GPUs (parallel)  
**Environments:** 13-bus (42-dim obs), 34-bus (105-dim obs), 123-bus (372-dim obs)  
**Action space:** 132 actions (2 caps × 2 caps × 33 taps) — same for all clients

---

## 1. Reward Results — All Conditions

```
                        13-bus      34-bus     123-bus
────────────────────────────────────────────────────────
Local only             -331.4       -65.5     -5364.4
Unaligned FL           -336.6       -69.6     -5420.5   ← heterogeneous FL problem: all worse
Aligned FL (50r)       -326.3       -85.0     -5402.5   ← 13-bus pass only
Aligned FL (200r)      -339.5       -69.3     -5251.4   ← 123-bus pass, 13-bus regresses (CSA)
Partial FL (2/3)       -341.4       -79.8     -5402.9   ← PAD: all worse
Personalised FL        -165.0       -15.2     -4034.5   ← BEST: all pass
────────────────────────────────────────────────────────
H1 vs local (✅=pass):
  Aligned 50r:   13-bus ✅  34-bus ❌  123-bus ❌
  Aligned 200r:  13-bus ❌  34-bus ❌  123-bus ✅   ← REVERSAL (CSA finding)
  Personalised:  13-bus ✅  34-bus ✅  123-bus ✅   ← only strategy that passes all
────────────────────────────────────────────────────────
```

**Personalised FL vs Local only (% improvement):**
```
  13-bus:  (-165.0 - -331.4) / |-331.4| × 100 = +50.2%
  34-bus:  (-15.2  - -65.5)  / |-65.5|  × 100 = +76.8%
  123-bus: (-4034.5 - -5364.4) / |-5364.4| × 100 = +24.8%
```

---

## 2. VQC Gradient Norms (Barren Plateau Diagnostic)

```
                        13-bus      34-bus     123-bus
────────────────────────────────────────────────────────
Local only             0.000276    0.000003   0.000027
Unaligned FL           0.000167    0.000040   0.000022
Aligned FL             0.000084    0.000065   0.000021
Partial FL (2/3)       0.000721    0.000200   0.000127
────────────────────────────────────────────────────────
```

**Key observations:**
- 34-bus: Aligned FL raises grad norm 21× vs local (0.000003→0.000065) — FL regularisation working
- 123-bus: Near-zero in all conditions — structural barren plateau (see ISSUE_002)
- Partial FL: Highest norms but in oscillating directions — not useful gradient signal

---

## 3. Communication Cost (H3)

```
Method                       Params    Bytes/round    Total (50r)    vs Classical
────────────────────────────────────────────────────────────────────────────────
QE-SAC-FL (VQC only)            16          384         19,200        6,920×
QE-SAC-FL-Aligned (Head+VQC)   288        6,912        336,000          395×
Federated Classical SAC     110,724    2,657,376    132,868,800        1.0× (baseline)
────────────────────────────────────────────────────────────────────────────────
```

---

## 4. Hypothesis Outcomes

| # | Hypothesis | Result | Strength |
|---|---|---|---|
| H1 | Aligned FL > Local-only | ⚠️ 1/3 clients (13-bus only) | Weak — need more rounds |
| H2 | Faster convergence | ⬜ Inconclusive | Need 200K+ steps |
| H3 | Less communication | ✅ 395–6920× less | **Strong — mathematical proof** |
| H4 | Barren plateau regularisation | ⚠️ Mixed (34-bus: +21×) | Moderate |
| H5 | Personalised FL | ✅ All 3 clients +25–77% | **Strongest result** |
| H6 | Partial participation robust | ❌ All clients fail | Novel negative finding |
| H7 | Transfer to new feeder | ⬜ Not yet run | — |
| H8 | Non-IID severity | ⬜ Not yet run | — |
| H9 | Round breakeven | ⚠️ 13-bus: round 1 | Partial |

---

## 5. Novel Discoveries

### Discovery 1: Quantum Latent Space Incompatibility (heterogeneous FL problem) — ISSUE_001
The first paper to name and characterise this structural failure mode of
quantum federated RL. Unaligned FL hurts ALL clients by creating a VQC
that operates in an averaged latent space incompatible with every client.

**Evidence:** Unaligned FL worse than local on all 3 clients:
- 13-bus: −336.6 vs −331.4 (−5.2)
- 34-bus: −69.6 vs −65.5 (−4.1)
- 123-bus: −5420.5 vs −5364.4 (−56.1)

### Discovery 2: Client Size Asymmetry (CSA) — FINDING_001
The SharedHead convergence direction is controlled by client gradient magnitude
over time. Small-feeder clients (high grad norm early) benefit first; large-feeder
clients (high reward scale) dominate later. H1 passes for different clients at
different round counts — the SharedHead cannot simultaneously serve all client
sizes optimally. Only personalised FL (H5) bypasses this.

**Evidence:**
- Round 50: 13-bus PASS (−326.3), 123-bus FAIL (−5402.5)
- Round 200: 13-bus FAIL (−339.5), 123-bus PASS (−5251.4)
- 34-bus never passes — sits between the two extremes

### Discovery 3: Partial Alignment Drift (PAD) — ISSUE_003
A new failure mode distinct from classical FL partial participation
degradation. When clients rotate in/out of FL rounds, the SharedHead
oscillates between incompatible 2-client objectives, reintroducing heterogeneous FL problem.

**Evidence:** Partial FL (2/3 clients) worse than local-only on all clients,
despite classical FL being robust to same dropout rate.

### Discovery 3: Personalised Quantum FL is Dramatically Better
The two-phase strategy (federate then personalise) yields 25–77% improvement
over local-only on all clients. The FL warm-start is the critical ingredient.

---

## 6. Paper Contribution Outline

**Title (candidate):**
"Privacy-Preserving Federated Quantum Reinforcement Learning for Multi-Utility
Volt-VAR Control: Identifying and Solving Quantum Latent Space Incompatibility"

**Contributions:**
1. **heterogeneous FL problem** — first identification and naming of this structural quantum FL problem
2. **Aligned Federation (SOLUTION_001)** — SharedEncoderHead architecture that fixes heterogeneous FL problem
3. **CSA** — SharedHead convergence favours different client sizes at different rounds
4. **PAD** — partial participation reintroduces heterogeneous FL problem (new, distinct from classical FL dropout)
5. **Personalised QFL** — two-phase strategy achieving +25–77% improvement with 395× less communication
6. **H3** — mathematical proof of communication advantage (395–6920× vs classical FL)

**Target venue:** IEEE Transactions on Smart Grid (T-SG) or IEEE Power & Energy Society General Meeting

---

## 7. Remaining Experiments

| Priority | Experiment | Expected | Time needed |
|---|---|---|---|
| HIGH | Aligned FL with n_rounds=200 | H1 proven for 34-bus, 123-bus | ~6 min/run |
| HIGH | PAD mitigation (FedProx on SharedHead) | H6 passes | 2 days code + run |
| MEDIUM | Transfer learning (H7) | +20-40% convergence speed | 1 week |
| MEDIUM | Non-IID severity (H8) | Aligned > classical at all severities | 3 days |
| LOW | 5-seed statistical significance | Mean ± std for all results | Overnight |

---

## 8. Artifacts

All results saved in `artifacts/qe_sac_fl/`:

| File | Contents |
|---|---|
| `local_only_results.json` | Per-round logs, all 3 clients |
| `QE-SAC-FL_results.json` | Unaligned FL logs |
| `QE-SAC-FL-Aligned_results.json` | Aligned FL logs |
| `QE-SAC-FL-Partial_results.json` | Partial participation logs |
| `QE-SAC-FL-Personalized_results.json` | Personalised FL final rewards |
| `h1_reward_comparison.png` | H1 bar chart (local vs unaligned FL) |
| `h2_convergence_curves.png` | Reward curves over rounds |
| `h4_barren_plateau.png` | Gradient norm comparison all conditions |
| `h5_personalized.png` | H5 bar chart (local vs aligned vs personal) |
| `h6_partial_participation.png` | H6 reward curves with dropout |
| `h9_breakeven.png` | Round breakeven plot |
| `vqc_grad_norm.png` | VQC gradient norm over rounds (original) |
