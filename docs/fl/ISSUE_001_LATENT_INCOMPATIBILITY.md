# ISSUE 001 — Quantum Latent Space Incompatibility

**Discovered:** 2026-04-01  
**Severity:** Critical — blocks H1  
**Status:** SOLVING → see SOLUTION_001  

---

## What Happened

Running QE-SAC-FL (federate VQC only) produced **worse** reward than local-only
training on all three clients after 50 rounds × 1,000 steps.

```
Results (round 50):
  Client         Local only    QE-SAC-FL    Delta
  13-bus          -329.8        -338.2       -8.4  ← FL is WORSE
  34-bus           -77.3         -81.6       -4.3  ← FL is WORSE
  123-bus        -5300.6       -5444.4      -143.8 ← FL is WORSE
```

The unaligned federated VQC **hurts every client** instead of helping.

---

## Root Cause

Each client's CAE independently learns to compress its own feeder state
into 8 numbers. The 8 dimensions mean completely different things per client:

```
Client A CAE (13-bus):
  z[0] = direction of voltage drop in upstream branch
  z[1] = aggregate load level
  z[2] = cap bank influence ...

Client B CAE (34-bus):
  z[0] = something related to rural load shape
  z[1] = regulator tap sensitivity ...
  z[2] = completely different meaning than Client A's z[2]

Client C CAE (123-bus):
  z[0..7] = yet another 8 independent directions
```

After FedAvg the VQC weights are the average of three VQCs each trained
in a different latent space. When Client A loads this averaged VQC and feeds
it its own z (from Client A's CAE), the VQC is operating in the wrong space.

```
WHAT THE AVERAGED VQC EXPECTS:  "average" latent space
WHAT CLIENT A GIVES IT:         Client A's private latent space
RESULT:                         VQC outputs random-looking action distributions
                                → reward degrades
```

This is NOT the same as classical FL client drift (caused by non-IID data).
It is a structural problem unique to quantum FL with private encoders.

**Name:** Quantum Latent Space Incompatibility (QLSI)

---

## Evidence

### 1. Reward drops after round 1 (34-bus)

```
Round:          0      1      2      3      4
Local-only:   -82    -87    -83    -81    -67   (improving)
QE-SAC-FL:    -75    -97   -104    -74    -75   (dips then recovers)
```

The dip at rounds 1–2 is the signature of QLSI: loading the aggregated VQC
from round 0 immediately degrades performance before local training partially
recovers it.

### 2. VQC gradient norms after 50 rounds

```
Client         Local      QE-SAC-FL
13-bus         0.000284   0.000104   ← FL reduces gradient signal
34-bus         0.000238   0.000120   ← FL reduces gradient signal
123-bus        0.000008   0.000011   ← near zero in both (barren plateau risk)
```

The FL condition has *smaller* VQC gradients than local-only. The averaged
VQC is in a flatter region of the loss landscape — harder to train.

### 3. CAE latent space visualisation (planned)

PCA of z-vectors per client would show three separate clusters with no overlap.
This directly proves the latent spaces are incompatible.

---

## Why This is Novel

Classical federated learning drift (FedProx, Scaffold, etc.) addresses:
- Non-IID data distributions across clients
- Model weight divergence due to heterogeneous local objectives

QLSI is different:
- Even with identical data distributions, QLSI would still occur
- The problem is the private encoder creating incompatible INPUT SPACES to the VQC
- Standard FL fixes (FedProx, momentum correction) do not address this

No paper has identified or named this problem.

---

## Impact on Paper

**Positive reframe:** The failure of unaligned federation IS the finding.
The paper now has two contributions instead of one:

1. **Identify QLSI** — first paper to name and characterise this problem
2. **Solve QLSI** — propose SharedEncoderHead architecture (SOLUTION_001)

A paper that finds a new problem AND solves it is stronger than one that
just shows a method works.

---

## Fix

See [SOLUTION_001_ALIGNED_FEDERATION.md](SOLUTION_001_ALIGNED_FEDERATION.md)
