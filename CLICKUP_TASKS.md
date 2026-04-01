# ClickUp Task Board — QE-SAC+ Research
**Project:** Quantum RL for Safe Volt-VAR Control
**Duration:** 2026-04-01 → 2026-06-30  (3 months)
**Researcher:** Ing Muyleang — Pukyong National University, QCL

---

## MILESTONES

| # | Milestone | Due Date | Status |
|---|---|---|---|
| M1 | All baselines running & verified | 2026-04-14 | ⬜ Not started |
| M2 | Safety constraint (Constrained SAC) verified | 2026-04-30 | ⬜ Not started |
| M3 | GNN encoder complete & tested | 2026-05-15 | ⬜ Not started |
| M4 | Full QE-SAC+ trained (10 seeds, 13-bus + 123-bus) | 2026-05-30 | ⬜ Not started |
| M5 | Transfer learning results finalized | 2026-06-10 | ⬜ Not started |
| M6 | Paper draft ready for review | 2026-06-30 | ⬜ Not started |

---

## TASK 1 — Environment & Baseline Finalization
**Description:**
Finalize the simulation environment and confirm all baseline agents
run correctly end-to-end. This is the foundation everything else builds on.
**Due:** 2026-04-14
**Priority:** URGENT
**Milestone:** M1

### Subtasks

| # | Subtask | Description | Due |
|---|---|---|---|
| 1.1 | Run full Classical SAC training (13-bus) | Train Classical SAC for 50K steps on VVCEnv13Bus. Log reward curve and voltage violations. Save checkpoint to artifacts/qe_sac/. | 2026-04-05 |
| 1.2 | Run full QE-SAC training (13-bus) | Train QE-SAC for 50K steps on VVCEnv13Bus. Confirm CAE retraining fires every 500 steps. Compare reward vs Classical SAC. | 2026-04-07 |
| 1.3 | Reproduce parameter count analysis | Print exact parameter breakdown for QE-SAC inference path vs Classical SAC. Document in QE_SAC_IMPLEMENTATION.md. | 2026-04-08 |
| 1.4 | Run both agents on 123-bus | Repeat 1.1 and 1.2 on VVCEnv123Bus. Confirm obs_dim=380 and action space work correctly. | 2026-04-12 |
| 1.5 | Write baseline results table | Fill in actual reward and VViol numbers from our runs into QE_SAC_RESEARCH_SUMMARY.md. | 2026-04-14 |

---

## TASK 2 — Constrained SAC (Safety Guarantee)
**Description:**
Implement Lagrangian-based constrained SAC to give a hard mathematical
guarantee that voltage violations never occur. This is the first new
contribution beyond the original paper.
**Due:** 2026-04-30
**Priority:** HIGH
**Milestone:** M2

### Subtasks

| # | Subtask | Description | Due |
|---|---|---|---|
| 2.1 | Study Lagrangian RL theory | Read: "Constrained Policy Optimization" (Achiam 2017) and "WCSAC" paper. Understand how λ multiplier works in SAC. Write a 1-page summary note. | 2026-04-16 |
| 2.2 | Implement constrained_sac.py | Create src/qe_sac/constrained_sac.py. Add Lagrange multiplier λ to actor loss. Update rule: λ += lr_λ × (mean_vviol − 0). Start λ = 0. | 2026-04-20 |
| 2.3 | Integrate constraint into QESACAgent | Modify qe_sac_policy.py or create QESACAgentConstrained. Constraint: E[voltage violations per step] ≤ 0. | 2026-04-22 |
| 2.4 | Train & verify safety on 13-bus | Run 5 seeds × 50K steps. Confirm: (a) violations reach 0, (b) reward is comparable to unconstrained. Plot λ curve over training. | 2026-04-26 |
| 2.5 | Write unit tests for constraint | Add tests to test_qesac_env.py. Test: λ increases when violations > 0, λ decreases when violations = 0. | 2026-04-28 |
| 2.6 | Document and compare | Add constrained vs unconstrained comparison table to RESEARCH_BIG_PICTURE.md. | 2026-04-30 |

---

## TASK 3 — GNN Encoder (Topology-Aware Compression)
**Description:**
Replace the flat MLP autoencoder (CAE) with a Graph Neural Network
that understands the physical connection structure of the distribution feeder.
Same 8-dim latent output — VQC is unchanged.
**Due:** 2026-05-15
**Priority:** HIGH
**Milestone:** M3

### Subtasks

| # | Subtask | Description | Due |
|---|---|---|---|
| 3.1 | Install PyTorch Geometric (PyG) | pip install torch-geometric. Verify it works with current PyTorch 2.5. Add to requirements.txt. | 2026-05-01 |
| 3.2 | Build feeder graph data structure | Create graph builder for VVCEnv13Bus and VVCEnv123Bus. Node features: [V, P, Q]. Edge features: [r, x]. Use torch_geometric.data.Data. | 2026-05-04 |
| 3.3 | Implement GNNEncoder in gnn_encoder.py | Create src/qe_sac/gnn_encoder.py. Architecture: 2-layer GCN or GraphSAGE → global mean pool → Linear → 8-dim latent → tanh × π. | 2026-05-07 |
| 3.4 | Replace CAE with GNN in QE-SAC+ | Create src/qe_sac/qe_sac_plus.py. Swap CAE encoder for GNNEncoder. Keep VQC and head identical. Verify output shape (8,) still goes into VQC. | 2026-05-10 |
| 3.5 | Write unit tests for GNN encoder | Test: output shape (8,), output in [-π, π], works on both 13-bus and 123-bus graphs, gradients non-zero. | 2026-05-12 |
| 3.6 | Train QE-SAC+ with GNN on 13-bus | Run 5 seeds × 50K steps. Compare vs QE-SAC (MLP CAE): convergence speed, final reward, parameter count. | 2026-05-15 |

---

## TASK 4 — QE-SAC+ Full Integration & Training
**Description:**
Combine all three contributions into one agent (QE-SAC+):
GNN encoder + VQC + Constrained SAC.
Run full training across all systems and seeds.
**Due:** 2026-05-30
**Priority:** HIGH
**Milestone:** M4

### Subtasks

| # | Subtask | Description | Due |
|---|---|---|---|
| 4.1 | Create QESACPlusAgent | Combine GNNEncoder + VQCLayer + constrained SAC critics into single class in qe_sac_plus.py. Clean API: same train/evaluate interface as QESACAgent. | 2026-05-17 |
| 4.2 | Full training run — 13-bus (10 seeds) | Train QE-SAC+ on VVCEnv13Bus. 10 seeds × 100K steps. Log: reward, vviol, λ curve, CAE loss. Save all checkpoints. | 2026-05-21 |
| 4.3 | Full training run — 123-bus (10 seeds) | Same as 4.2 on VVCEnv123Bus. Expected: longer convergence. Monitor barren plateau (gradient norms). | 2026-05-25 |
| 4.4 | Noise robustness test | Run evaluate_noise_robustness() on trained QE-SAC+ VQC. λ = 0.1%, 0.5%, 1.0%. Compare vs original QE-SAC. | 2026-05-27 |
| 4.5 | Ablation: constraint & encoder | Train QE-SAC+ (no constraint) vs QE-SAC+ (with constraint). Also compare MLP CAE vs GNN encoder. Show reward cost of safety is small. | 2026-05-30 |
| 4.6 | VQC qubit & layer ablation | Make VQCLayer configurable. Sweep n_qubits ∈ {4,8,12,16} and n_layers ∈ {1,2,3,4} on 13-bus (3 seeds × 50K steps each). Log gradient norms to detect barren plateau. Main agent stays at 8 qubits / 2 layers. Save to artifacts/qe_sac/vqc_ablation_*.json. | 2026-05-30 |

---

## TASK 5 — Transfer Learning Evaluation
**Description:**
Test whether QE-SAC+ trained on a small feeder can control a larger
feeder it has never seen. This is the key generalisation claim.
**Due:** 2026-06-10
**Priority:** MEDIUM
**Milestone:** M5

### Subtasks

| # | Subtask | Description | Due |
|---|---|---|---|
| 5.1 | Implement transfer_eval.py | Create src/qe_sac/transfer_eval.py. Function: transfer_evaluate(agent, source_env, target_env, freeze_vqc=True, n_adapt_steps=500). | 2026-06-01 |
| 5.2 | Transfer: 13-bus → 123-bus | Freeze VQC weights. Re-initialise GNN encoder on 123-bus graph. Run 500 GNN adaptation steps. Evaluate reward and vviol. | 2026-06-04 |
| 5.3 | Zero-shot test (no adaptation) | Evaluate frozen VQC directly on 123-bus with NO GNN retraining. Measure reward drop vs trained-from-scratch. | 2026-06-06 |
| 5.4 | Compare transfer agents | Table: (a) trained-from-scratch, (b) zero-shot, (c) 500-step adapted, (d) classical SAC from scratch. | 2026-06-08 |
| 5.5 | Add transfer results to summary doc | Update RESEARCH_BIG_PICTURE.md with transfer table and conclusion. | 2026-06-10 |

---

## TASK 6 — Final Comparison & Writing
**Description:**
Produce the complete results table comparing all agents, write the
paper draft, and prepare final presentation for advisor.
**Due:** 2026-06-30
**Priority:** MEDIUM
**Milestone:** M6

### Subtasks

| # | Subtask | Description | Due |
|---|---|---|---|
| 6.1 | Final results table (all agents) | Compare: Classical SAC / QC-SAC / QE-SAC (paper) / QE-SAC+ (ours). Metrics: reward, vviol, params, transfer gap, safety guarantee. | 2026-06-13 |
| 6.2 | Training curve figures | Matplotlib plots: reward curves (all agents, 10 seeds, mean±std), λ curve, parameter count bar chart, noise robustness chart. | 2026-06-16 |
| 6.3 | Write paper Introduction & Related Work | Cover: VVC problem, DRL for power systems, quantum ML, prior QRL papers. ~2 pages. | 2026-06-19 |
| 6.4 | Write paper Methodology section | Describe QE-SAC+ architecture, constrained SAC derivation, GNN encoder design. Include architecture figure. ~3 pages. | 2026-06-22 |
| 6.5 | Write paper Results & Discussion | Report all numbers from Tasks 4 & 5. Discuss why QE-SAC+ beats baselines. ~2 pages. | 2026-06-25 |
| 6.6 | Full paper review & advisor submission | Combine all sections. Check formatting for IEEE Transactions on Smart Grid. Submit draft to advisor. | 2026-06-30 |

---

## QUICK OVERVIEW TIMELINE

```
APRIL 2026
──────────────────────────────────────────────────
Week 1  Apr 01–07   Task 1: Run baselines (SAC, QE-SAC)
Week 2  Apr 08–14   Task 1: Verify results, parameter analysis   ← M1
Week 3  Apr 15–22   Task 2: Implement Constrained SAC
Week 4  Apr 23–30   Task 2: Train + verify safety guarantee      ← M2

MAY 2026
──────────────────────────────────────────────────
Week 5  May 01–07   Task 3: Install PyG, build feeder graph
Week 6  May 08–15   Task 3: GNN encoder complete + tested        ← M3
Week 7  May 16–22   Task 4: QE-SAC+ integrated, 13-bus run
Week 8  May 23–30   Task 4: 123-bus + ablations + noise test     ← M4

JUNE 2026
──────────────────────────────────────────────────
Week 9  Jun 01–10   Task 5: Transfer learning experiments        ← M5
Week 10 Jun 11–20   Task 6: Final results + figures
Week 11 Jun 21–30   Task 6: Paper writing + advisor review       ← M6
```

---

## STATUS LEGEND

| Symbol | Meaning |
|---|---|
| ✅ | Completed |
| 🔄 | In Progress |
| ⬜ | Not Started |
| 🔴 | Blocked |
| ⭐ | Milestone |
