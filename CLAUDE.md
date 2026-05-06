# Claude Role — QE-SAC-FL Research Assistant
## Pukyong National University, Quantum Computing Lab

---

## Who You Are Helping

**Ing Muyleang** — PhD researcher, Pukyong National University (QCL).
Advisor: Myeongseong (Professor).
Goal: Publish federated quantum RL paper to IEEE Transactions on Smart Grid.

**Communication rules — read every session:**
- User types with heavy typos. NEVER ask for clarification — infer intent and act.
- Short, direct responses. No trailing summaries.
- Lead with results, not explanations.
- Do not name the method unless user decides — use "This Work" or "[PROPOSED]".

---

## What This Project Is

Extending Lin et al. (2025) QE-SAC (DOI: 10.1109/OAJPE.2025.3534946) to a
**federated multi-utility setting** across heterogeneous grid topologies.

**Three core contributions:**
1. Aligned encoder — solves Quantum Latent Space Incompatibility (heterogeneous FL problem)
2. 383× communication reduction vs classical SAC-FL (288 params/client/round)
3. Barren plateau prevention — VQC gradients stable under alignment

---

## Key Architecture (always remember this)

```
obs [B, obs_dim]
  → LocalEncoder (PRIVATE — MLP or GNN variant)
  → h [B, 32]
  → SharedEncoderHead [32→8, FEDERATED, 264 params]
  → z [B, 8] ∈ (-π, π)
  → VQC [8 qubits, 16 params, FEDERATED]
  → q [B, 8]
  → N action heads (PRIVATE)

Total federated: 280 params = 1.1 KB per client per round
```

**Two encoder variants being compared:**
- MLP: `LinearEncoder(obs_dim→64→32)` — flat, no topology awareness
- GNN: `BusGNN(node_features, adj_matrix)` — topology-aware, handles faults

---

## Key Numbers — Memorise These

| Result | Value |
|---|---|
| Federated params | **280 params = 1.1 KB/round** |
| vs Classical SAC-FL | **383× reduction** |
| Effect size B (n=5) | **d=+1.74, p=0.0089** |
| Effect size C (n=5) | **d=+1.24, p=0.0252** |
| Effect size C OpenDSS | **d=+1.97, p=0.054** |
| Personalized FL gain B | **+4.549** |
| VQC barren plateau | naive FL collapses 10⁻¹→10⁻³ by round 100 |

---

## Experiment Status (update when things finish)

| Experiment | Status | Log | Output |
|---|---|---|---|
| 5-seed FL (linearized) | ✅ DONE | fl_500k.log | artifacts/qe_sac_fl/ |
| Hidden dim ablation | ✅ DONE | ablation.log | artifacts/qe_sac_fl/ablation/ |
| Personalized FL (H5) | ✅ DONE | fl_personalized.log | artifacts/qe_sac_fl/seed*_personalized.json |
| OpenDSS real validation | ✅ DONE | fl_opendss_real.log | artifacts/qe_sac_fl_opendss/ |
| MLP vs GNN comparison | 🔄 RUNNING | logs/fl_gnn_comparison.log | artifacts/qe_sac_fl_gnn/ |

**Check running experiments with:**
```bash
tail -f logs/fl_gnn_comparison.log
```

---

## Critical Technical Rules (never break these)

1. **VQC backend = default.qubit ONLY** — do not switch to lightning.qubit.
   Reason: fair comparison with Lin et al. (2025) requires identical simulator.

2. **VQC weights shape = (2, 8)** — NOT flat (16,).
   Any FedAvg on VQC must use `view(-1, *([1]*(stacked.dim()-1)))`.

3. **reward_scale MUST be set in ClientConfig in every run script.**
   A=50.0, B=10.0, C=750.0. Do not rely on FedConfig defaults.

4. **GNN adj_matrix self-loops required** — `adj += eye(n_buses)` before normalisation.

---

## File Map — Most Important Files

```
src/qe_sac_fl/
  aligned_encoder.py    — LocalEncoder, SharedEncoderHead, AlignedCAE, fedavg_shared_head
  gnn_encoder.py        — BusGNN, GNNLocalEncoder, GNNAlignedCAE
  aligned_agent.py      — AlignedQESACAgent, GNNAlignedQESACAgent
  federated_trainer.py  — FederatedTrainer, RewardScaledEnv, _make_env, _fedavg
  fed_config.py         — FedConfig, ClientConfig

scripts/
  run_fl_500k.py              — main 5-seed FL (done)
  run_fl_opendss_real.py      — real OpenDSS FL (done)
  run_fl_gnn_comparison.py    — MLP vs GNN (RUNNING)
  verify_results.py           — statistical verification
  run_dynamic_topology.py     — fault injection experiment (pending)

artifacts/
  QE_SAC_FL_Technical_Report.md    — full technical report with figures
  QE_SAC_FL_Progress_Apr12.pptx   — progress slides
  qe_sac_fl/verification/          — n=5 stats, plots
  qe_sac_fl_opendss/               — real OpenDSS results
  qe_sac_fl_gnn/                   — MLP vs GNN results (filling now)

TODO.md                            — full 32-task todo list with descriptions
```

---

## Task Priority Right Now

1. **Task 25** — Write advisor reply (barren plateau + KB reduction + multi-topology)
2. **Task 21** — Prove barren plateau with VQC circuit diagram
3. **Task 22** — Communication cost KB comparison figure
4. **Task 31** — Wait for MLP vs GNN results (running, PID 2602905)
5. **Task 32** — Dynamic topology fault injection experiment
6. **Tasks 14-17** — Paper sections (Related Work → Conclusion)

See TODO.md for full list with descriptions.

---

## When Starting a New Session

1. Read this file first (CLAUDE.md)
2. Check `tail -20 logs/fl_gnn_comparison.log` for experiment status
3. Check TODO.md for current task list
4. Check `artifacts/QE_SAC_FL_Technical_Report.md` for latest results
5. Ask user what they want to work on — do NOT assume

---

## Advisor Context

Myeongseong asked about:
- **Barren plateau** — needs circuit diagram + gradient norm proof
- **KB reduction** — the 383× is the biggest contribution, make it visual
- **Multiple grid topologies** — GNN comparison addresses this directly

Email format user prefers (for advisor emails):
- Subject: "Weekly Progress Report — [topic]"
- Structure: What implemented → Why it matters → Key results table → Next steps
- Keep concise, professional, honest about limitations
