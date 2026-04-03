# QE-SAC Implementation Guide
## Quantum-Enhanced SAC for Volt-VAR Control

---

## What This Is

This implements **QE-SAC** (Quantum-Enhanced Soft Actor-Critic) from:

> Lin et al. (2025), *"QE-SAC: Quantum Reinforcement Learning for Volt-VAR Control
> in Distribution Systems"*, IEEE Open Access Journal of Power and Energy.
> DOI: https://doi.org/10.1109/OAJPE.2025.3534946

The core idea: replace the large MLP actor inside SAC with a tiny
**Classical Autoencoder → 8-qubit Quantum Circuit** pipeline,
achieving ~1% of the classical parameter count at comparable performance.

---

## Architecture: How It Works

```
High-dim grid state  s  (42-dim for 13-bus)
         │
         ▼
┌─────────────────────┐
│  Classical          │   Encoder: s → 64 → 32 → 8
│  Autoencoder (CAE)  │   Output scaled to [-π, π]
│  (co-adaptive)      │   Retrained every C=500 gradient steps
└─────────────────────┘
         │  latent s' (8-dim)
         ▼
┌─────────────────────┐
│  8-qubit VQC        │   Angle encoding: RY(s'_i)|0⟩ on each qubit
│  (PennyLane)        │   × 2 variational layers:
│                     │     - CNOT(i, i+1) nearest-neighbour entanglement
│                     │     - RX(ζ_k) trainable rotation per qubit
│                     │   Measurement: ⟨Z_i⟩ → output in [-1, 1]^8
│                     │   Gradient: parameter-shift rule
└─────────────────────┘
         │  vqc_out (8-dim)
         ▼
┌─────────────────────┐
│  Linear + Softmax   │   8 → n_actions → action probabilities
└─────────────────────┘
         │
         ▼
  Per-device action (capacitor ON/OFF, regulator tap, battery SoC)
```

**Critics (classical):** Two MLP Q-networks, same as standard SAC.
**Replay buffer:** 1M transitions.
**SAC hyperparams:** γ=0.99, lr=1e-4, batch=256, ρ=0.005.

---

## Paper vs Our Implementation

### Architecture Comparison

| Component | Paper | Our Implementation |
|---|---|---|
| VQC qubits | 8 | 8 ✓ |
| VQC layers | 2 | 2 ✓ |
| Encoding | Angle (RY) | Angle (RY) ✓ |
| Entanglement | CNOT nearest-neighbour | CNOT nearest-neighbour ✓ |
| Trainable gates | RX(ζ_k) per qubit per layer | RX(ζ_k) ✓ |
| Gradient method | Parameter-shift | Parameter-shift ✓ |
| CAE latent dim | 8 | 8 ✓ |
| CAE hidden | not specified | 64 → 32 |
| CAE update interval | C = 500 steps | C = 500 ✓ |
| Simulator | PennyLane default.qubit | PennyLane default.qubit ✓ |
| Critics | Twin MLP | Twin MLP ✓ |
| Replay buffer | 1M | 1M ✓ |

### Environment Comparison

| Aspect | Paper (PowerGym/OpenDSS) | Our Implementation |
|---|---|---|
| Simulator | OpenDSS 3-phase unbalanced AC | Linearised DistFlow (DC approx) |
| 13-bus topology | Exact IEEE 13-bus feeder | Approximate IEEE 13-bus params |
| 123-bus topology | Exact IEEE 123-bus feeder | Random radial (123 nodes) |
| Devices | Caps, regulators, batteries | Caps, regulators (batteries=0) |
| Episode length | 24 steps (full day) | 24 steps ✓ |
| Load variation | OpenDSS load profiles | Uniform noise ±10% |

> **Why we didn't use PowerGym:** The Siemens PowerGym package has no
> pip-installable distribution (no `setup.py` or `pyproject.toml`).
> Our DistFlow environment preserves the same observation/action/reward
> interface so all agent code is simulator-agnostic.

### Parameter Count Comparison

#### 13-bus system

| Component | Paper | Ours | Root cause of gap |
|---|---|---|---|
| QE-SAC actor | **4,872** | **10,575** | We count decoder; paper counts inference path only |
| QE-SAC (inference only) | **4,872** | **5,445** | Hidden dim difference (our 64→32 vs paper unknown) |
| Classical SAC | **899,729** | **86,309** | Paper's obs_dim ≈ 3,219 (OpenDSS); ours = 42 (DistFlow) |
| Reduction ratio | 185× | 8× | Scales with the classical SAC size |

#### Root Cause 1 — Classical SAC is too small (86K vs 899K)

Back-calculating from the paper's 899,729 params with a 2×256 MLP actor:

```
899,729 = obs_dim × 256 + 256        # layer 1
        + 256 × 256 + 256             # layer 2
        + 256 × 37  + 37              # output (37 actions)
→  obs_dim ≈ 3,219
```

The paper uses **PowerGym/OpenDSS** which gives a full 3-phase unbalanced state:
per-phase voltages + angles + currents + loads at every bus = **thousands of values**.
Our DistFlow environment gives only 42 values (voltages + loads + device states).
Because the classical MLP is smaller, the compression ratio looks smaller too (8× vs 185×).

#### Root Cause 2 — QE-SAC is too large (10.5K vs 4.8K)

Our actor has two parts counted together, but the paper counts **inference-time only**:

```
Our total actor (10,575):
  CAE encoder  (42→64→32→8) :  5,096  ← used at inference
  CAE decoder  (8→32→64→42) :  5,130  ← used ONLY during CAE training, NOT inference
  VQC weights  (2×8)         :     16  ← used at inference
  Linear head  (8→37)        :    333  ← used at inference
                               ------
  Inference-only path        :  5,445  ← comparable to paper's 4,872

Remaining gap (5,445 − 4,872 = 573 params):
  Due to hidden layer sizes. Paper likely uses smaller hidden dims in the encoder.
  Exact architecture not specified in the paper.
```

**The VQC itself always has exactly 16 parameters** (2 layers × 8 qubits) — this matches the paper exactly. The count difference is entirely in the CAE hidden layers.

#### 123-bus system

| Component | Paper | Ours |
|---|---|---|
| QE-SAC actor | **42,329** | **55,158** |
| Classical SAC | 3,213,427 | 200,850 |

Same two root causes apply at 123-bus scale.

### Reward Function

| Term | Paper | Ours |
|---|---|---|
| f_vv (voltage violation) | α · Σ(V - V_lim)² | α=100 · Σ(V - V_lim)² ✓ |
| f_cl (capacitor switching) | β · Σ\|cap changes\| | β=1 · Σ\|changes\| ✓ |
| f_pl (power loss) | γ · total loss | γ=0.1 · P_loss ✓ |
| V limits | [0.95, 1.05] pu | [0.95, 1.05] pu ✓ |

### Performance Results

The paper reports (avg 10 seeds):

| Algorithm | 13-bus Reward | 13-bus VViol | 123-bus Reward | 123-bus VViol | Params (123-bus) |
|---|---|---|---|---|---|
| **QE-SAC** | **−5.39** | **0** | **−9.53** | **0** | 42,329 |
| Classical SAC | −5.41 | 0.01 | −9.72 | 0 | 3,213,427 |
| QC-SAC (fixed PCA) | −5.91 | 0.02 | −10.78 | 0.01 | ~42K |

Our results will differ because:
1. We use a simplified DistFlow environment (not 3-phase OpenDSS)
2. Training budget in the demo notebook is shorter (5K steps vs paper's full run)
3. Fewer seeds (3 vs 10)

To get closer to paper numbers: increase `N_STEPS` to 200K+ and `N_SEEDS` to 10 in the notebook.

---

## File-by-File Guide

### `src/qe_sac/env_utils.py`
VVC gymnasium environments. Key classes:

```python
env = VVCEnv13Bus(episode_len=24, load_noise=0.1, seed=0)
obs, info = env.reset()   # obs shape: (42,)
obs, reward, done, _, info = env.step(action)
# info = {"voltage": V_pu_array, "P_loss": float, "v_viol": int}
```

State vector layout (13-bus, 42-dim):
```
[0:13]  — bus voltages in pu
[13:26] — normalised active load P per bus
[26:39] — normalised reactive load Q per bus
[39:41] — capacitor bank status (0=OFF, 1=ON)
[41]    — voltage regulator tap normalised to [0,1]
```

Action space: `MultiDiscrete([2, 2, 33])`
- `action[0]` — capacitor bank 1 (bus 8): 0=OFF, 1=ON
- `action[1]` — capacitor bank 2 (bus 11): 0=OFF, 1=ON
- `action[2]` — voltage regulator tap: 0–32 → ratio [0.9, 1.1]

### `src/qe_sac/autoencoder.py`
Co-adaptive CAE. Key usage:

```python
from src.qe_sac.autoencoder import CAE, train_cae

cae = CAE(input_dim=42)       # latent dim always 8
z = cae.encode(obs_tensor)    # z in [-π, π], shape (8,)
x_hat, z = cae(obs_tensor)    # forward: returns (reconstruction, latent)

# Retrain on recent buffer observations (called every 500 steps)
loss = train_cae(cae, observations_array, n_steps=50)
```

### `src/qe_sac/vqc.py`
8-qubit PennyLane VQC:

```python
from src.qe_sac.vqc import VQCLayer

vqc = VQCLayer(noise_lambda=0.0)  # noiseless
out = vqc(latent_tensor)          # shape (8,), values in [-1, 1]
print(vqc.n_params)               # 16 = 2 layers × 8 qubits
```

Circuit structure per call:
```
  Input: latent s' = [s'_0, ..., s'_7] ∈ [-π, π]

  Preparation:  RY(s'_0)  RY(s'_1)  ...  RY(s'_7)

  Layer 1:
    Entangle:  CNOT(0,1) CNOT(1,2) ... CNOT(6,7)
    Rotate:    RX(ζ_0)   RX(ζ_1)   ...  RX(ζ_7)

  Layer 2:  (same structure, different ζ weights)

  Measure:  <Z_0>  <Z_1>  ...  <Z_7>
```

### `src/qe_sac/qe_sac_policy.py`
Full agent:

```python
from src.qe_sac.qe_sac_policy import QESACAgent

agent = QESACAgent(
    obs_dim=42, n_actions=37,
    lr=1e-4, gamma=0.99, tau=0.005, alpha=0.2,
    buffer_size=1_000_000, noise_lambda=0.0, device='cpu'
)
action = agent.select_action(obs_tensor)        # numpy scalar
agent.store(obs, action, reward, next_obs, done)
logs = agent.update(batch_size=256, cae_update_interval=500)
agent.save('artifacts/qe_sac/agent.pt')
```

### `src/qe_sac/trainer.py`
Training loop:

```python
from src.qe_sac.trainer import QESACTrainer, compare_agents

trainer = QESACTrainer(agent, env, batch_size=256,
                        cae_update_interval=500, warmup_steps=1000)
metrics = trainer.train(n_steps=50_000)
print(metrics.summary())
```

---

## Running the Experiment

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Run all tests (24 tests, ~25s)
python -m pytest tests/test_qesac_env.py -v

# 3. Quick parameter count check
python -c "
from src.qe_sac.env_utils import VVCEnv13Bus
from src.qe_sac.qe_sac_policy import QESACAgent
from src.qe_sac.sac_baseline import ClassicalSACAgent
env = VVCEnv13Bus()
obs_dim = env.observation_space.shape[0]
n_act   = int(env.action_space.nvec.sum())
qe = QESACAgent(obs_dim, n_act)
cl = ClassicalSACAgent(obs_dim, n_act)
print(f'QE-SAC: {qe.param_count():,} params')
print(f'SAC:    {cl.param_count():,} params')
"

# 4. Full experiment (open notebook)
jupyter notebook notebooks/qe_sac_experiment.ipynb
```

---

## Known Differences vs the Paper

| Difference | Impact | Fix |
|---|---|---|
| DistFlow vs OpenDSS | Voltages less accurate | Install PowerGym when available |
| 42-dim vs ~200-dim state | Smaller classical MLP → smaller ratio | Use full OpenDSS env |
| Demo: 5K steps | Lower convergence | Increase `N_STEPS` to 200K+ |
| Demo: 3 seeds | Higher variance | Increase `N_SEEDS` to 10 |
| No batteries | Missing one device type | Add `_n_bats > 0` to env class |

---

## Noise Robustness

The VQC is tested under **depolarising noise** at λ = 0.1%, 0.5%, 1.0%:

```python
from src.qe_sac.noise_model import evaluate_noise_robustness
results = evaluate_noise_robustness(clean_vqc, n_samples=100)
# Returns mean/std of |output_clean - output_noisy| per noise level
```

The paper shows performance is stable at λ ≤ 1%, meaning the circuit
is robust enough for near-term quantum hardware.
