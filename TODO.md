# QE-SAC-FL — Master Todo List
**Project**: Federated Quantum-Enhanced SAC for Heterogeneous Volt-VAR Control
**Author**: Ing Muyleang — Pukyong National University, QCL
**Last updated**: April 13, 2026

---

## ✅ Completed

### 1. Set up IEEE 14-bus test case with PyPSA/pandapower
Configure the IEEE 14-bus standard test case. Define network topology, bus data,
line parameters, generator setpoints, and load profiles. Verify the network loads
correctly and is ready for power flow analysis.

### 2. Set up IEEE 30-bus test case
Configure the IEEE 30-bus standard test case with PyPSA or pandapower. Define all
30 buses, 41 branches, 6 generators, and load data. Ensure topology matches the
IEEE 30-bus benchmark specification for use in OTS experiments.

### 3. Validate power flow solutions on both test cases
Run AC power flow on IEEE 14-bus and 30-bus cases. Compare voltage magnitudes,
angles, and line flows against published benchmark values (MATPOWER / IEEE PES).
Confirm convergence and numerical accuracy before proceeding to OTS formulation.

### 4. Formulate Optimal Transmission Switching (OTS) problem
Formulate the OTS problem as a mixed-integer program. Each transmission line has
a binary switch variable (on/off). Objective: minimise total generation cost or
active power losses subject to power balance, line flow limits, voltage limits,
and N-1 security constraints.

### 5. Set up MILP solver for baseline OTS comparison
Implement the MILP baseline for OTS using a solver (CPLEX, Gurobi, or CBC via
PuLP/Pyomo). Serves as the classical optimal benchmark that the RL agent is
compared against. Record solve time, objective value, and switching decisions
for both 14-bus and 30-bus cases.

### 6. Run 5-seed experiments for all FL conditions
Run all three FL conditions (local_only, naive_fl, aligned_fl) for seeds 0–4.
Each seed uses different random initialisations and load profiles. Collect
per-seed reward curves and save to artifacts/. Run verify_results.py after
completion to get n=5 statistical analysis with Cohen's d and
Bonferroni-corrected p-values.
**Result**: B: d=+1.74 p=0.0089 (*) | C: d=+1.24 p=0.0252

### 7. Implement gradient-normalised FedAvg to fix CSA
Implement gradient-normalised federated averaging to address the Client Staleness
Asymmetry (CSA) problem — where clients with different topology complexity produce
gradients of very different magnitudes. Normalise each client's gradient
contribution before aggregation so that simpler topologies (13-bus) do not
dominate over complex ones (123-bus).

### 9. Verify QFA and FL environment imports and client builds
Verify that all QFA and FL environment imports resolve correctly across all three
clients (13-bus, 34-bus, 123-bus). Confirm AlignedQESACAgent, AlignedCAE,
FederatedTrainer, and environment constructors build without errors on all target
devices. Run a 2-round smoke test to confirm end-to-end execution.

### 10. Run all 3 FL conditions: local_only, QE-SAC-FL, QE-SAC-FL-Aligned
Execute the full FL experiment pipeline for all three conditions:
- **local_only** — each client trains independently, no communication
- **QE-SAC-FL** — naive FedAvg on full VQC weights, demonstrates heterogeneous FL problem problem
- **QE-SAC-FL-Aligned** — FedAvg on SharedEncoderHead + VQC only (288 params)

### 11. Check OpenDSS real-physics validation results
Ran scripts/run_fl_opendss_real.py (12.1 hours). All 3 FL conditions on real
VVCEnvOpenDSS (actual 3-phase AC power flow, not linearized). Results saved to
artifacts/qe_sac_fl_opendss/.
**Key finding**: Client C d=+1.97, p=0.054 on real physics. FL provides more
stable convergence than local-only (local variance ±5-6 vs FL ±0.5).

---

## 🔄 In Progress

### 8. Run and finalise workflow of QFA and Agent
Complete the end-to-end workflow for the Quantum Federated Agent (QFA). Ensure
AlignedQESACAgent integrates correctly with FederatedTrainer, reward scaling,
CAE pre-training, and FL round loop. Finalise agent checkpointing, logging,
and result serialisation. Document the full pipeline for reproducibility.

### 12. Analyse OpenDSS real-physics results and update technical report
Interpret the real OpenDSS FL results (n=3 seeds):
- local_only has huge variance (±5–6) — some seeds converge, some get stuck
- FL conditions converge consistently (variance ±0.5) — FL stabilises training
- Client C: aligned vs naive d=+1.97, p=0.054 (close to significant)
Update QE_SAC_FL_Technical_Report.md Section 5 with OpenDSS results table
and interpretation.

### 25. Write response to advisor explaining barren plateau, KB reduction, and multi-topology plan
Write clear email reply to Myeongseong covering his three main questions:
1. **BARREN PLATEAU** — what it is (Var[∇θ]~O(2^{-n})), why heterogeneous FL problem causes it
   in FL (misaligned inputs → contradictory gradients → cancellation → flat
   landscape), show gradient norm figure as proof
2. **COMMUNICATION REDUCTION (KB)** — 383x reduction: SharedHead (272p) +
   VQC (16p) = 288 params = 1.1 KB vs classical 430 KB. Hard architectural
   fact, not a statistical result.
3. **MULTI-TOPOLOGY** — plan to add 57-bus as 4th client, already have
   13/34/123-bus working, aligned encoder handles any obs_dim.

### 28. Fix seed0_local_only OpenDSS result (wrong reward_scale from crashed run)
The seed0_local_only.json was manually recovered from a crashed run that used
reward_scale=50.0 (values ~-0.87). The rerun uses reward_scale=1.0 (values
~-9 to -14). The file is on a completely different scale and corrupts the
summary statistics.
**Fix**: Delete artifacts/qe_sac_fl_opendss/seed0_local_only.json and rerun
seed 0 local_only condition only. Resume logic skips all other files.

---

## 📋 Pending — Advisor Feedback (Priority)

### 21. Prove barren plateau with VQC circuit diagram and gradient experiment
Advisor asked specifically about barren plateau. Three deliverables:
1. **VQC circuit diagram** — draw 8-qubit circuit: RY(z_i) encoding gates +
   CNOT(i, i+1) entanglement ring + RX(θ_i) trainable gates + PauliZ
   measurement. Save to artifacts/figures/vqc_circuit.png
2. **Gradient collapse experiment** — train VQC with misaligned inputs (naive
   FL) vs aligned inputs (aligned FL), plot ‖∇θ‖ vs round on same axis.
   Show the collapse clearly. Save to artifacts/figures/barren_plateau_proof.png
3. **Math explanation** — Var[∂L/∂θ_k] ~ O(2^{-n}) from McClean et al. 2018.
   With 8 qubits, gradients are ~1/256 of what a classical network would see.

### 22. Build communication cost comparison table and figure (KB classical vs quantum)
Advisor flagged communication reduction as a big contribution — make it undeniable:
1. **Breakdown table** — exactly where 383x comes from:
   Classical SAC actor MLP (2×256) = 110,724 params = 430 KB
   SharedHead (32×8+8=264) + VQC (16) = 280 params = 1.09 KB
2. **Bar chart on log scale** — Classical SAC-FL (430 KB) | Naive QE-SAC-FL
   (29 KB) | Aligned QE-SAC-FL (1.1 KB)
3. **Cumulative bandwidth** over 500 rounds × 3 clients:
   Classical = 1.29 GB | Ours = 3.3 MB
4. **Real-world framing** — at 100 KB/s SCADA link: classical needs 4.3 s/round,
   ours needs 0.01 s/round. Save to artifacts/figures/comm_cost_comparison.png

### 23. Extend FL to IEEE 57-bus topology as 4th client
Implement VVCEnv57Bus in src/qe_sac_fl/env_57bus.py using linearized DistFlow
(same pattern as env_34bus.py). IEEE 57-bus: 57 buses, 80 branches, obs_dim ~180,
4 cap banks + 3 regulators. Add to FedConfig as Utility_D.
This directly answers the reviewer question about generalisability.
**Blocked by**: nothing — can start now

### 24. Run 4-topology FL experiment (13, 34, 57, 123-bus) and compare results
After 57-bus env is ready (Task 23), run full FL with 4 clients. Compare:
- Does adding a 4th topology hurt existing clients A, B, C?
- Does the 57-bus client (D) benefit from alignment?
- Communication cost still 288 params regardless of topology — only bandwidth
  scales linearly with N clients.
Expected: aligned FL generalises to 4 topologies; naive FL gets worse with more
clients (more heterogeneous FL problem interference).
Output: artifacts/qe_sac_fl_4topology/
**Blocked by**: Task 23

### 26. Add scalability analysis — comm cost vs N clients and grid size
Show that 288-param federation is independent of grid size (obs_dim). Create:
- Table: N clients (2,3,4,5,10) × Classical KB/round vs QE-SAC-FL KB/round vs ratio
- Key argument: adding a new client (e.g. 57-bus, obs_dim=180) does NOT increase
  federated parameter count — only private LocalEncoder grows, stays local.
- Formula: comm_cost = N_clients × 2 × 288 × 4 bytes (upload + download, float32)
This is a key scalability argument for the paper.

---

## 📋 Pending — Paper Writing

### 13. Run OpenDSS validation with 5 seeds for stronger statistics
Re-run scripts/run_fl_opendss_real.py with SEEDS=[0,1,2,3,4] for n=5.
Note: seed0_local_only must be rerun (Task 28). Resume logic skips existing files.
With d=+1.97 on Client C (n=3), n=5 is expected to push p below 0.05.
**Blocked by**: Task 28 (fix seed0 first)

### 14. Write paper Section 2 — Related Work
Cover: (1) Volt-VAR control with RL — cite Lin et al. 2025 as base,
(2) Federated learning for power systems — privacy-preserving multi-utility,
(3) Quantum machine learning / VQC for optimisation,
(4) Barren plateau in quantum FL — cite McClean et al. 2018, Cerezo et al. 2021,
(5) Heterogeneous FL — FedAvg limitations with non-IID data.
~1.5 pages, ~15 citations.

### 15. Write paper Section 3 — Problem Formulation
Formally define the multi-utility federated VVC problem:
(1) MDP formulation — state space, action space (MultiDiscrete), reward function
(2) Federated learning setup — K clients, communication rounds, privacy model
(3) heterogeneous FL problem problem definition — formal statement of why naive FedAvg fails
(4) Communication cost model — bytes per round formula
Use mathematical notation consistent with Lin et al. 2025.

### 16. Write paper Section 4 — Proposed Method (QE-SAC-FL)
Use QE_SAC_FL_Technical_Report.md Sections 2 and 7 as draft. Cover:
(1) AlignedCAE architecture — LocalEncoder + SharedEncoderHead + LocalDecoder
(2) Why the split solves heterogeneous FL problem — geometric argument
(3) Federation protocol — FedAvg on 288 params, EMA server momentum
(4) Full QE-SAC-FL algorithm as pseudocode
(5) Communication cost analysis (383x reduction)

### 17. Write paper Section 5 — Experiments and Results
Use technical report Section 5 as draft. Include:
(1) Experimental setup — 3 clients, 3 conditions, 5 seeds, 500K steps
(2) Main results table (n=5 linearized env)
(3) Figures 1–4 (learning curves, delta dist, gradient norms, ablation)
(4) OpenDSS real-physics validation as separate subsection
(5) Personalized FL results (+4.549 on B — strongest single number)
(6) Communication efficiency table (383x)
Frame stats honestly — effect sizes + bootstrap CI primary, p-values secondary.

### 18. Generate final paper figures in publication quality
Regenerate all figures at 300 DPI, IEEE column width (~3.5 inches):
- Figure 1: Learning curves
- Figure 2: Delta distributions
- Figure 3: VQC gradient norms (barren plateau)
- Figure 4: Hidden dim ablation
- Figure 5: OpenDSS validation
- Figure 6: Communication cost comparison (from Task 22)
- Figure 7: VQC circuit diagram (from Task 21)
Use matplotlib IEEE-style (Times New Roman, no grid, tight layout).
Save as PNG + PDF to artifacts/qe_sac_fl/figures/.

### 27. Generate updated progress slides (Apr 13)
Regenerate QE_SAC_FL_Progress slides with new content:
(1) OpenDSS real-physics results slide
(2) Barren plateau circuit diagram slide
(3) Communication cost KB comparison bar chart
(4) Multi-topology roadmap slide (current: 13/34/123, planned: +57-bus)
(5) Updated Next Actions slide
Output: artifacts/QE_SAC_FL_Progress_Apr13.pptx

### 19. Send paper draft to advisor Myeongseong for review
Compile complete paper draft (Sections 1–5 + conclusion + references) and
send to Myeongseong for feedback. Include:
(1) PDF of draft
(2) Updated progress slides (QE_SAC_FL_Progress_Apr13.pptx)
(3) Brief email summarising OpenDSS real-physics results and statistical status
Target date: April 22, 2026.
**Blocked by**: Task 20

### 20. Submit paper to IEEE Transactions on Smart Grid
Final submission checklist:
- All experiments complete and reproducible
- OpenDSS validation n=5 done
- Advisor approved draft
- Abstract <250 words
- All figures 300 DPI
- References formatted IEEE style
- Cover letter: contribution novelty — aligned encoder + barren plateau
  prevention + 383x comm reduction
Submit via IEEE Author Portal. Target: May 2026.
**Blocked by**: Task 19

---

## Quick Reference

| File | Purpose |
|---|---|
| `scripts/run_fl_opendss_real.py` | Real OpenDSS FL validation |
| `scripts/verify_results.py` | Statistical verification (linearized env) |
| `artifacts/QE_SAC_FL_Technical_Report.md` | Full technical report |
| `artifacts/QE_SAC_FL_Progress_Apr12.pptx` | Latest progress slides |
| `artifacts/qe_sac_fl/verification/summary.json` | n=5 stats (linearized) |
| `artifacts/qe_sac_fl_opendss/verification/summary.json` | n=3 stats (real OpenDSS) |
| `logs/fl_opendss_real.log` | OpenDSS experiment log |

## Key Numbers to Remember

| Result | Value | Notes |
|---|---|---|
| Communication reduction | **383×** | vs classical SAC-FL — hard fact |
| Federated params/round | **288 params = 1.1 KB** | SharedHead + VQC only |
| Effect size B (linearized) | **d = +1.74, p = 0.0089** | n=5, significant |
| Effect size C (linearized) | **d = +1.24, p = 0.0252** | n=5, trend |
| Effect size C (real OpenDSS) | **d = +1.97, p = 0.054** | n=3, near significant |
| Personalized FL gain B | **+4.549** | strongest single result |
| VQC gradient collapse | **10⁻¹ → 10⁻³ by round 100** | naive FL barren plateau |
