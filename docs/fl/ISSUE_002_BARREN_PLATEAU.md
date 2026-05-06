# ISSUE 002 — Barren Plateau Risk in 123-bus Client

**Discovered:** 2026-04-01  
**Severity:** Medium — limits 123-bus trainability, does not block results  
**Status:** MONITORED — real data collected, mitigations planned  
**Related:** H4 (barren plateau hypothesis), RESEARCH_PLAN.md

---

## What Was Observed

After 50 rounds of training, the VQC gradient norm for all clients is small,
with 123-bus consistently the weakest:

```
Real experimental data (round 50):
  Condition          13-bus grad    34-bus grad    123-bus grad
  ─────────────────────────────────────────────────────────────
  Local only         0.000276       0.000003       0.000027
  Unaligned FL       0.000167       0.000040       0.000022
  Aligned FL         0.000084       0.000065       0.000021
  Partial FL (2/3)   0.000721       0.000200       0.000127
```

**Key finding:** 123-bus gradient norms are near-zero in ALL conditions.
This is a structural issue, not a training instability.

**Surprising finding:** Partial FL (2/3 clients) shows the HIGHEST gradient norms
across all conditions, including for 123-bus. See Note below.

---

## Root Cause Analysis

Barren plateaus in quantum circuits arise from two mechanisms:

### Mechanism 1: High-Dimensional Observation → Lossy Compression
```
123-bus:  obs_dim=372  →  LocalEncoder (372→64→32)  →  SharedHead (32→8)  →  VQC
```
The 372-dim observation is compressed 46× before the VQC sees it. This aggressive
compression may wash out feeder-specific voltage signal — the VQC receives a
poorly-conditioned input with low information density.

Compare:
- 13-bus: 42-dim → 32 (compression ratio 1.3×) — minimal loss
- 34-bus: 105-dim → 32 (compression ratio 3.3×) — moderate loss  
- 123-bus: 372-dim → 32 (compression ratio 11.6×) — heavy loss

### Mechanism 2: Large Feeder Reward Variance
The 123-bus feeder has much larger reward magnitude (−5000 range vs −65 for 34-bus).
High reward variance → high actor loss gradient → but the VQC gradient is
only the VQC's contribution to the actor gradient. If the critic and MLP head
dominate the backward pass, the VQC sees very little gradient.

### Mechanism 3: Expressibility vs Trainability Trade-off
Our circuit: 8 qubits × 2 layers = 16 trainable RX parameters.
For a 372-dim problem, 16 params may be too few to express useful policies.
The VQC is forced into a subspace that doesn't cover the relevant policy directions.
(Holmes et al. 2022, PRX Quantum 3, 010313)

---

## Why 123-bus is NOT in a True Barren Plateau

Evidence that training IS happening:
```
123-bus reward progression (local-only condition):
  Round 1:   ~-5800
  Round 10:  ~-5600
  Round 25:  ~-5450
  Round 50:  -5364.4

That's +435 reward improvement over 50 rounds despite near-zero VQC gradients.
```

The VQC gradients are small but non-zero. The MLP critic and actor head are
learning even when the VQC learns slowly. This is not a global barren plateau —
it is a **local gradient starvation** problem specific to the VQC component.

---

## H4 Finding: What Does FL Do to Barren Plateau Risk?

From the real data:

| Client | Local grad norm | Aligned FL grad norm | Change |
|---|---|---|---|
| 13-bus | 0.000276 | 0.000084 | −70% (worse) |
| 34-bus | 0.000003 | 0.000065 | **+2067% (much better)** |
| 123-bus | 0.000027 | 0.000021 | −22% (slightly worse) |

**Interpretation:**
- 34-bus shows strong FL regularisation: the near-zero local gradient is raised 21× by FL
- 13-bus local grad is already the highest — FL averaging pulls it down (regression to mean)
- 123-bus: FL makes minimal difference (gradient is near-zero regardless)

**H4 conclusion:** FL provides gradient regularisation for clients that were in
a local barren region (34-bus). It does not help clients with structural gradient
starvation (123-bus). H4 is partially confirmed.

---

## The Partial FL Surprise

Partial FL (2/3 clients per round) shows the HIGHEST gradient norms:
```
13-bus partial:  0.000721  (vs 0.000084 aligned, 0.000276 local)
34-bus partial:  0.000200  (vs 0.000065 aligned, 0.000003 local)
123-bus partial: 0.000127  (vs 0.000021 aligned, 0.000027 local)
```

High gradient norms from partial FL do NOT mean better training.
This is the signature of Partial Alignment Drift (PAD — see ISSUE_003):
- The VQC is being pulled in inconsistent directions each round
- Large gradient norm = large but unstable updates
- The reward is still WORSE than local-only despite high grad norms

**Lesson:** High VQC gradient norm is necessary but not sufficient for good performance.
The gradient direction matters more than magnitude when SharedHead is misaligned.

---

## Proposed Mitigations

### 1. Increase LocalEncoder Depth for 123-bus
Current: Linear(372→64) → ReLU → Linear(64→32) → ReLU  
Proposed: Linear(372→128) → ReLU → Linear(128→64) → ReLU → Linear(64→32) → ReLU

The extra layer gives the encoder more capacity to compress 372-dim → 32-dim
without losing voltage control signal.

### 2. Local Cost Functions for VQC
Instead of measuring all 8 qubits with global <Z_0 ⊗ Z_1 ⊗ ... ⊗ Z_7>,
use local 2-qubit operators: <Z_0 ⊗ Z_1>, <Z_2 ⊗ Z_3>, etc.
Local cost functions have polynomially (not exponentially) vanishing gradients.
(Ref: Cerezo et al. 2021, Nature Comm 12:1791)

### 3. VQC Warm-Start from 13-bus
Pre-train VQC on the 13-bus feeder (smallest, highest gradient) for 10K steps,
then use those weights as the initial global VQC for all clients.
Avoids random initialisation in flat regions.

### 4. More VQC Layers for 123-bus
Current: N_LAYERS=2 for all clients.
Proposed: N_LAYERS=3 for 123-bus only.
More layers = more expressibility for the large feeder.
Trade-off: deeper circuits are more susceptible to true barren plateaus (test carefully).

### 5. Gradient Rescaling in FedAvg
Before FedAvg, rescale each client's VQC gradient by 1/||∇θ_client||.
Prevents 13-bus (highest gradient) from dominating the aggregate direction.
Ensures 123-bus gradient has equal influence despite near-zero magnitude.

---

## Priority

Mitigation 1 (deeper LocalEncoder) is the easiest to implement and most likely
to help — try this first before changing the VQC architecture.
Mitigation 2 (local cost functions) is the most theoretically grounded.

---

## Related

- ISSUE_001: heterogeneous FL problem — separate problem (latent space incompatibility)
- ISSUE_003: PAD — explains why partial FL has high but useless gradient norms
- H4: Barren plateau hypothesis — real data above
- RESEARCH_PLAN.md: H4 future work section
