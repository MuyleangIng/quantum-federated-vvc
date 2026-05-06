# Paper References — QE-SAC-FL
## Compiled: April 2026

---

## BASE PAPER (must cite first)

**[B1] Lin et al. (2025) — QE-SAC**
- Full: Lin, Y. et al., "Quantum-Enhanced Soft Actor-Critic for Volt-VAR Control in Distribution Networks"
- DOI: 10.1109/OAJPE.2025.3534946
- IEEE Open Access Journal of Power and Energy, 2025
- **Why cite**: Our direct base — we extend QE-SAC to federated multi-utility setting
- **Where**: Introduction, Section 3 (problem formulation), Section 4 (method), Section 5 (baselines)
- **Key numbers to compare**: QE-SAC 4,896 params, our aligned_fl 280 federated params → 17× reduction over QE-SAC itself

---

## SECTION 2 — RELATED WORK

### 2A. Volt-VAR Control with RL

**[R1] Lee, Sarkar & Wang (2022) — Graph-PPO for VVC** ⭐ MOST IMPORTANT
- Full: X.Y. Lee, S. Sarkar, Y. Wang, "A graph policy network approach for Volt-Var Control in power distribution systems"
- Applied Energy, vol. 323, 2022, article 119530
- DOI: 10.1016/j.apenergy.2022.119530
- **Why cite**: 
  1. Uses same 13/34/123-bus benchmarks — direct comparison
  2. Proves GNN slower than MLP on static topology (validates our result at 50K steps)
  3. Proves GNN more robust to sensor failures: Dense-PPO −272% vs Graph-PPO −12% at 75% missing sensors on 123-bus
  4. Their best 34-bus result: −3.56 ± 0.20 (our personalized FL: −3.261 ± 0.096 — we beat them)
- **Where**: Related Work §2, Results §5 (comparison table), GNN encoder justification

**[R2] Multi-agent graph RL for decentralized VVC (2023)**
- Full: (authors TBD) "Multi-agent graph reinforcement learning for decentralized Volt-VAR control in power distribution systems"
- International Journal of Electrical Power & Energy Systems, 2023
- URL: https://www.sciencedirect.com/science/article/pii/S0142061523005884
- **Why cite**: Multi-agent + GNN + VVC, but no FL, no privacy, no comm reduction
- **Where**: Related Work §2 — gap: "prior decentralized VVC lacks privacy and formal comm budget"

**[R3] Domain knowledge-enhanced GRL for V/V control (2025)**
- URL: https://www.sciencedirect.com/science/article/abs/pii/S0306261925011390
- Applied Energy, 2025
- **Why cite**: Very recent GNN+RL for VVC — cite as concurrent work, differentiate by FL + aligned encoder
- **Where**: Related Work §2

### 2B. Quantum Machine Learning / VQC

**[Q1] Lin et al. (2025)** — see [B1] above

**[Q2] McClean et al. (2018) — Barren Plateau**
- Full: J.R. McClean, S. Boixo, V.N. Smelyanskiy, R. Babbush, H. Neven, "Barren plateaus in quantum neural network training landscapes"
- Nature Communications, vol. 9, 4812, 2018
- DOI: 10.1038/s41467-018-07090-4
- **Why cite**: Original barren plateau paper — Var[∂L/∂θ_k] ~ O(2^{-n}), with 8 qubits → 1/256 gradient magnitude
- **Where**: Section 4 (barren plateau explanation), advisor reply

**[Q3] Cerezo et al. (2021) — Barren Plateau review**
- Full: M. Cerezo et al., "Cost function dependent barren plateaus in shallow parametrized quantum circuits"
- Nature Communications, vol. 12, 1791, 2021
- DOI: 10.1038/s41467-021-21728-w
- **Why cite**: Shows barren plateau worsens with global cost functions — supports aligned encoder design
- **Where**: Section 4 (barren plateau)

### 2C. Federated Learning

**[F1] McMahan et al. (2017) — FedAvg**
- Full: H.B. McMahan, E. Moore, D. Ramage, S. Hampson, B.A. y Arcas, "Communication-Efficient Learning of Deep Networks from Decentralized Data"
- AISTATS 2017
- arXiv: 1602.05629
- **Why cite**: Original FedAvg — our _fedavg() implements this, 383× reduction is vs this baseline
- **Where**: Section 3 (FL background), Section 4 (federation protocol)

**[F2] FL with Heterogeneous Architectures + Graph HyperNetworks (2022)**
- arXiv: 2201.08459
- **Why cite**: FL with heterogeneous clients — shows standard FedAvg fails with different architectures (our heterogeneous FL problem problem)
- **Where**: Related Work §2 — motivates aligned encoder

**[F3] Federated RL for EV charging control (2024)**
- URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC12216737/
- Nature/PMC, 2025
- **Why cite**: Federated RL for power systems — adjacent work, no quantum, no aligned encoder
- **Where**: Related Work §2

### 2D. GNN for Power Systems

**[G1] PowerGNN — Suri & Mangal (2025)**
- arXiv: 2503.22721, March 2025
- **Why cite**: 
  1. Same architecture: hidden_dim=32, 2 GraphSAGE layers, mean pooling — validates our GNNLocalEncoder design
  2. GNN reduces state prediction error 73.5% vs MLP on NREL 118-bus
  3. Node features [V, θ, P, Q] per bus — same as our flat_obs_to_node_features
- **Where**: Section 4 (GNN encoder design motivation)

**[G2] Generalizable GNN for robust grid topology control (2025)**
- arXiv: 2501.07186, January 2025
- **Why cite**: 
  1. GNNs generalize better than FCNNs to out-of-distribution topologies
  2. Quote: "heterogeneous GNNs generalize better to out-of-distribution networks than FCNNs"
  3. Directly supports Task 32 (fault injection) argument
- **Where**: Section 4 (GNN encoder), Section 5 (GNN fault tolerance results)

**[G3] SAC + GNN for large-scale EV coordination (2025)**
- Nature Communications Engineering, 2025
- URL: https://www.nature.com/articles/s44172-025-00457-8
- **Why cite**: 
  1. SAC (same algorithm as ours) + GNN — proves combination is viable
  2. GNN actor: 256K params vs MLP SAC: 2.8M params — supports parameter efficiency argument
  3. GNN needs more initial training but better scalability — validates our 50K convergence result
- **Where**: Section 4 (GNN design), Section 5 (GNN convergence discussion)

**[G4] Graph RL in Power Grids: A Survey (2024)**
- arXiv: 2407.04522
- URL: https://arxiv.org/html/2407.04522v2
- **Why cite**: Survey paper — useful for Related Work overview sentence
- **Where**: Related Work §2 opening sentence

---

## SECTION 5 — RESULTS (comparison numbers)

### Your numbers vs papers

| Method | Client B reward | Params | Source |
|---|---|---|---|
| Classical-SAC | −6.716 ± 0.413 | 113,288 | Your gpu_run.log |
| SAC-AE | −5.432 ± 0.581 | 5,024 | Your gpu_run.log |
| QC-SAC | −6.168 ± 0.057 | 1,240 | Your gpu_run.log |
| QE-SAC [B1] | −6.585 ± 0.960 | 4,896 | Your gpu_run.log |
| Lee et al. best (34-bus) [R1] | −3.56 ± 0.20 | N/A | Lee 2022 Table 7 |
| **[PROPOSED] aligned_fl** | **−7.750 ± 0.621** | **280** | **Your work (n=5)** |
| **[PROPOSED] personalized FL** | **−3.261 ± 0.096** | **280** | **Your work (n=3)** |

**Personalized FL beats Lee et al. best single-agent by 0.30 reward units, with privacy + 405× less communication.**

---

## IEEE FORMAT CITATIONS (ready to paste)

```
[1] Y. Lin et al., "Quantum-Enhanced Soft Actor-Critic for Volt-VAR Control in Distribution Networks," IEEE Open Access J. Power Energy, 2025, doi: 10.1109/OAJPE.2025.3534946.

[2] X. Y. Lee, S. Sarkar, and Y. Wang, "A graph policy network approach for Volt-Var control in power distribution systems," Appl. Energy, vol. 323, p. 119530, 2022, doi: 10.1016/j.apenergy.2022.119530.

[3] J. R. McClean, S. Boixo, V. N. Smelyanskiy, R. Babbush, and H. Neven, "Barren plateaus in quantum neural network training landscapes," Nat. Commun., vol. 9, p. 4812, 2018, doi: 10.1038/s41467-018-07090-4.

[4] M. Cerezo et al., "Cost function dependent barren plateaus in shallow parametrized quantum circuits," Nat. Commun., vol. 12, p. 1791, 2021, doi: 10.1038/s41467-021-21728-w.

[5] H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, "Communication-efficient learning of deep networks from decentralized data," in Proc. AISTATS, 2017.

[6] D. Suri and M. Mangal, "PowerGNN: A topology-aware graph neural network for electricity grids," arXiv:2503.22721, Mar. 2025.

[7] (Authors TBD), "Generalizable graph neural networks for robust power grid topology control," arXiv:2501.07186, Jan. 2025.

[8] (Authors TBD), "Multi-agent graph reinforcement learning for decentralized Volt-VAR control in power distribution systems," Int. J. Electr. Power Energy Syst., 2023, doi: 10.1016/j.ijepes.2023.109573.

[9] (Authors TBD), "Federated learning with heterogeneous architectures using graph hypernetworks," arXiv:2201.08459, 2022.

[10] (Authors TBD), "Scalable reinforcement learning for large-scale coordination of electric vehicles using graph neural networks," Commun. Eng., 2025.
```

---

## TODO — verify full author names for [7], [8], [9], [10]
These need full author names before final submission. Use Google Scholar or IEEE Xplore.
