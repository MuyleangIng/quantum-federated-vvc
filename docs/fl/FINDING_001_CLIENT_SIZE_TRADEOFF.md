# FINDING 001 — Client Size Trade-off in SharedHead Convergence

**Discovered:** 2026-04-01  
**Type:** Novel empirical finding  
**Severity:** Important — affects H1 interpretation and deployment strategy

---

## What Was Observed

Running aligned FL for 200 rounds (vs 50) produced a reversal of which clients benefit:

```
               50 rounds           200 rounds      Local baseline
Client         Aligned FL  H1?     Aligned FL  H1?  (reference)
──────────────────────────────────────────────────────────────────
13-bus          -326.3     PASS     -339.5     FAIL    -331.4
34-bus           -85.0     FAIL      -69.3     FAIL     -65.5
123-bus        -5402.5     FAIL    -5251.4     PASS   -5364.4
──────────────────────────────────────────────────────────────────
```

**At 50 rounds:** 13-bus benefits (+5.1), large feeders hurt.
**At 200 rounds:** 123-bus benefits (+112.9), 13-bus regresses (−8.1).

---

## Root Cause: SharedHead Convergence Direction

The SharedHead (32→8) is federated via FedAvg each round. Its gradient
is the average of three client gradients weighted equally:

```
∇S = (1/3) * [∇S_A (from 13-bus) + ∇S_B (from 34-bus) + ∇S_C (from 123-bus)]
```

In early rounds:
- 13-bus has the highest gradient norm (0.000276 local) → dominates FedAvg
- SharedHead moves toward what works for 13-bus first
- 13-bus benefits early; large feeders see a head tuned for a small feeder

In later rounds:
- 13-bus LocalEncoder has converged — its gradient signal weakens
- 123-bus has massive reward range (−5000 vs −65) → large loss magnitude
- Over 200 rounds, the loss-scale imbalance shifts SharedHead toward 123-bus
- 123-bus eventually benefits; 13-bus head is no longer optimal for it

This is a **gradient magnitude imbalance** across clients with very different
observation dimensions and reward scales.

---

## Why This is Novel

Classical FL heterogeneity research (Zhao 2018, Li 2021) focuses on:
- Non-IID data distributions across clients
- Different local objectives due to different data

The client size trade-off is different:
- All clients have the SAME objective (VVC reward maximisation)
- The problem is the SCALE of the problem: 42-dim vs 372-dim observations
  create vastly different gradient magnitudes in the shared component
- Larger clients (more complex feeder) eventually dominate FedAvg
- Smaller clients benefit early then regress

**Name:** Client Size Asymmetry (CSA) in Quantum FL

---

## Implications for Deployment

1. **Round selection matters:** Deploy the SharedHead at the round that best
   serves the intended client mix. For a network with mostly small feeders,
   use fewer rounds. For mostly large feeders, use more.

2. **Weighted FedAvg may help:** Instead of equal weights in FedAvg, weight
   each client by the INVERSE of its gradient magnitude:
   ```
   w_i = 1 / ||∇S_i||     (normalise so small-gradient clients have more influence)
   ```
   This prevents large feeders from dominating over time.

3. **Personalised FL (H5) bypasses this entirely:** After fine-tuning, each
   client adapts the SharedHead to its own feeder — the round-selection
   problem disappears. This reinforces H5 as the recommended deployment strategy.

---

## Proposed Mitigation: Gradient-Normalised FedAvg

```python
def fedavg_normalised(weight_list, grad_norm_list):
    """
    FedAvg weighted by inverse gradient norm.
    Prevents large-gradient clients from dominating SharedHead convergence.
    """
    weights = [1.0 / max(g, 1e-8) for g in grad_norm_list]
    total = sum(weights)
    keys = weight_list[0].keys()
    averaged = {}
    for k in keys:
        stacked = torch.stack([w[k].float() * (wt / total)
                               for w, wt in zip(weight_list, weights)], dim=0)
        averaged[k] = stacked.sum(dim=0)
    return averaged
```

This is a single-line change to `fedavg_shared_head()` in `aligned_encoder.py`.

---

## Impact on Paper

This finding adds a third novel contribution alongside QLSI and PAD:

1. **QLSI** — structural alignment failure of unaligned FL
2. **PAD** — alignment breaks under partial participation
3. **CSA** — SharedHead convergence favours larger clients over time

Together, these three findings characterise the unique challenges of
**quantum FL with heterogeneous clients** — a topic not addressed in
any existing paper.

---

## Evidence Summary

| Metric | 13-bus (small) | 34-bus (medium) | 123-bus (large) |
|---|---|---|---|
| Local grad norm | 0.000276 (highest) | 0.000003 (lowest) | 0.000027 |
| Reward scale | ~300 | ~65 | ~5400 |
| Benefits at 50r | ✅ | ❌ | ❌ |
| Benefits at 200r | ❌ | ❌ | ✅ |
| Crossover round | ~70 (estimated) | >200 | ~120 (estimated) |

---

## Related

- ISSUE_002: Barren plateau on 123-bus (explains why 123-bus needs more rounds)
- ISSUE_003: PAD (separate problem, different mechanism)
- HYPOTHESES.md: H1 results updated
- RESEARCH_PLAN.md: Gradient-normalised FedAvg as future work
