# ISSUE 003 — Partial Participation Breaks Alignment

**Discovered:** 2026-04-01  
**Severity:** Medium — deployment limitation, not a training bug  
**Status:** OPEN — documented, mitigation planned  
**Related:** ISSUE_001 (QLSI), SOLUTION_001 (Aligned Federation)

---

## What Happened

Running aligned FL with only 2/3 clients participating per round (one randomly
dropped each round) produced WORSE reward than local-only training:

```
Results (round 50, partial participation dropout_rate=0.33):
  Client         Local only    Partial FL (2/3)    Delta
  13-bus          -331.4        -341.4              -10.0  ← WORSE
  34-bus           -65.5         -79.8              -14.3  ← WORSE
  123-bus        -5364.4       -5402.9              -38.5  ← WORSE
```

This is surprising because classical FL (FedAvg) is well-known to be
robust to partial participation (McMahan et al. 2017).

---

## Why Quantum FL is Different from Classical FL

In classical FL, partial participation degrades performance gradually because:
- Missing clients → their gradient direction is absent from FedAvg
- Server still averages the present clients → roughly correct direction
- Missing client catches up when it rejoins next round

In quantum FL with aligned federation, partial participation causes a NEW
alignment problem — call it **Partial Alignment Drift (PAD)**:

```
Round t: Clients A and B participate, C is dropped.
  → SharedHead is updated using A+B gradients only
  → New SharedHead is optimised for A and B's local encoders
  → Client C's LocalEncoder was adapted to the PREVIOUS SharedHead

Round t+1: Clients A and C participate, B is dropped.
  → Client C loads the new SharedHead — it has drifted from what C's
    LocalEncoder was trained on
  → Client C is now misaligned (QLSI re-introduced for C)
  → Client A's LocalEncoder is also disrupted by C's misaligned gradients

Round t+2: Clients B and C participate, A is dropped.
  → Same cycle repeats for B
```

The result: every round, at least one client operates with a misaligned
SharedHead. The alignment that SOLUTION_001 establishes is destroyed every
round because the dropped client's encoder diverges from the SharedHead.

---

## Mathematical Intuition

With full participation (3/3 clients per round), FedAvg on SharedHead minimises:

```
L_shared = (1/3) * [L_A(h_A, S) + L_B(h_B, S) + L_C(h_C, S)]
```

where `S` = SharedHead, `h_X` = ClientX's LocalEncoder.

With 2/3 participation, each round minimises a different 2-client objective:
```
Round t:   L_AB = (1/2) * [L_A(h_A, S) + L_B(h_B, S)]   (C dropped)
Round t+1: L_AC = (1/2) * [L_A(h_A, S) + L_C(h_C, S)]   (B dropped)
Round t+2: L_BC = (1/2) * [L_B(h_B, S) + L_C(h_C, S)]   (A dropped)
```

The SharedHead oscillates between 3 different 2-client objectives rather than
converging to a single 3-client optimum. The LocalEncoders cannot keep up with
the oscillating SharedHead — alignment never stabilises.

---

## Evidence

Gradient norms under partial FL are surprisingly HIGH:

```
Condition          13-bus grad norm    34-bus grad norm    123-bus grad norm
Local only         0.000276            0.000003            0.000027
Aligned FL (3/3)   0.000084            0.000065            0.000021
Partial FL (2/3)   0.000721            0.000200            0.000127
```

Partial FL has 3–8× HIGHER gradient norms than full aligned FL. This is the
signature of an unstable training loop — large gradients, but in inconsistent
directions. The VQC is being pulled in different directions each round as the
SharedHead oscillates, producing high gradient magnitude but poor convergence.

---

## Why This is Novel

Classical FL literature (McMahan 2017, Yang 2021, Gu 2021) shows:
- FedAvg is robust to partial participation under standard assumptions
- Missing clients introduce bias but convergence still holds

PAD is different:
- The problem is not missing gradient signal (classical FL issue)
- The problem is that partial participation creates rotating alignment targets
- The LocalEncoder-SharedHead coupling makes quantum FL fundamentally more
  sensitive to participation patterns than classical FL

**Name:** Partial Alignment Drift (PAD)

---

## Impact on Paper

This is a valuable negative result with clear practical implications:

- **For deployment:** Utility companies must guarantee full participation
  each FL round, or implement a PAD mitigation strategy
- **As a contribution:** First paper to identify PAD as distinct from
  classical partial participation degradation
- **Future work:** Motivates PAD-robust aggregation methods

---

## Proposed Mitigations (Future Work)

### 1. FedProx Regularisation on SharedHead
Add a proximal term to prevent SharedHead from drifting too far from each
client's last-known SharedHead:

```
L_client = L_RL + (μ/2) * ||S - S_global_prev||²
```

This limits how much the SharedHead can move per round — reduces oscillation.
(Ref: Li et al. 2020, ICLR — FedProx)

### 2. Momentum-Based Aggregation (FedAvgM)
Apply server-side momentum to SharedHead updates. Momentum smooths out the
oscillation between 2-client objectives.
(Ref: Hsieh et al. 2020, ICLR)

### 3. Frozen SharedHead During Dropout
When a client is dropped, freeze its last-known SharedHead contribution in
the FedAvg calculation (weighted average with stale weights):

```
S_new = (1/3) * [S_A_new + S_B_new + S_C_stale]
```

The stale weight prevents the SharedHead from drifting away from C's encoder.

### 4. LocalEncoder Adaptation Window
After rejoining, give the dropped client K steps of LocalEncoder-only training
(SharedHead frozen) before resuming full federation. This lets the LocalEncoder
re-align with the drifted SharedHead before contributing to FedAvg.

---

## Related Files

| File | Role |
|---|---|
| `src/qe_sac_fl/federated_trainer.py` | `run_partial_participation()` implementation |
| `docs/ISSUE_001_LATENT_INCOMPATIBILITY.md` | Original QLSI problem |
| `docs/SOLUTION_001_ALIGNED_FEDERATION.md` | Solution that PAD affects |
| `docs/RESEARCH_PLAN.md` | H6 hypothesis and future mitigations |
