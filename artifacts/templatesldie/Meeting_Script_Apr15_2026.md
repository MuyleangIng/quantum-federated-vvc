# QE-SAC-FL — Full Presenter Script
### Biweekly Meeting — April 15, 2026 — Ing Muyleang

---

## HOW TO USE THIS SCRIPT
- 🎤 = What you say out loud
- 📌 = What to point at on the slide
- ✏️ = Draw or write on board
- ⏱ = Suggested time

---

## SLIDE 1 — Title
⏱ ~30 sec

🎤
> "Good morning Professor. My name is Ing Muyleang. Today I will present my biweekly progress on the QE-SAC-FL research — Federated Quantum Reinforcement Learning for Volt-VAR Control across heterogeneous power grid topologies.
>
> This week I completed two new experiments and I have one open question I would like your guidance on."

---

## SLIDE 2 — Index
⏱ ~30 sec

🎤
> "I have six completed experiments this period. The two new ones are the GNN vs MLP encoder comparison and the VQC circuit ablation. The latent alignment evaluation is still running and finishes tonight. I will walk through all results and then open for discussion."

📌 Point to each experiment item as you say it.

---

## SLIDE 4 — Executive Summary
⏱ ~2 min

🎤
> "Three contributions. First — the aligned encoder solves heterogeneous FL problem, which is Quantum Latent Space Incompatibility. Client B shows effect size d=+1.74 with p=0.009, which is statistically significant after Bonferroni correction. Client C shows d=+1.24, which is a strong trend.
>
> Second — 405 times smaller communication cost than classical SAC-FL. We send only 280 parameters per round — that is 1.1 KB — compared to 453 KB for classical federated SAC.
>
> Third — the aligned encoder prevents gradient instability that naive FL causes inside the VQC. Without alignment, clients produce contradictory gradients that cancel each other out."

📌 Point to the three numbered contributions as you say each one.

---

## SLIDE 5 — New Progress
⏱ ~2 min

🎤
> "The main result: aligned FL outperforms both local-only and naive FL on Client B and Client C. The strongest single number in my results is personalized FL on Client B — the reward improved from negative 8.075 with local training alone to negative 3.261, which is a 59.6% improvement.
>
> For the latent alignment evaluation, the early signal shows that local-only has negative cosine similarity around negative 0.15. This confirms that without any alignment, clients have no shared geometric structure in their latent space. Aligned FL is expected to produce the highest similarity — that result comes tonight."

📌 Point to the reward bar chart. Trace the green bar on Client B.

---

## SLIDE 6 — New Progress (cont) — Comm Cost + VQC Ablation
⏱ ~2 min

🎤
> "The 405 times communication reduction is an architectural fact — not a statistical result. We send 280 parameters per round. Over 500 rounds with 3 clients, classical SAC-FL uses 1.36 gigabytes total bandwidth. Our method uses 3.3 megabytes.
>
> For the VQC circuit ablation, I tested four circuit variants on 8 qubits. The paper circuit — 2-layer linear CNOT chain — achieves the best reward of negative 5.879. Adding more layers does not help: 4-layer gives negative 5.956. Removing entanglement gives negative 5.991. Ring CNOT is the worst at negative 6.023. This confirms that every design choice in the original architecture is justified."

📌 Point to the communication cost bar chart. Show the log scale gap.
📌 Point to the VQC ablation numbers.

---

## SLIDE 7 — New Progress (cont 2) — GNN vs MLP
⏱ ~1.5 min

🎤
> "I compared two encoder variants — MLP and GNN — using the same 280 federated parameters. The only difference is the private local encoder. MLP wins on static topology across all three clients, which matches Lee et al. 2022.
>
> For OpenDSS real-physics validation using full AC power flow, aligned FL converges stably with variance around plus or minus 0.5, while local-only shows high variance of plus or minus 5 to 6 — some seeds get stuck in a bad policy. This shows that federation acts as a stabiliser even on real physics."

📌 Point to the GNN vs MLP bar chart.

---

## SLIDE 8 — Response to Previous Feedback
⏱ ~1.5 min

🎤
> "Three points from previous feedback.
>
> On barren plateau — I reframed this as heterogeneous FL problem-induced gradient instability, which is more precise and data-supported. Naive FL produces 2.5 times higher seed variance and 1.4 times more round instability. The mechanism is: misaligned latent inputs cause contradictory VQC gradients that cancel each other out.
>
> On KB reduction — the bar chart on log scale makes the 405 times reduction visually clear. I also want to clarify — 280 params is the federated cost only, not the model size. I will come back to this in the discussion.
>
> On multiple grid topologies — all three topology sizes are used in every experiment, and the GNN comparison covers 3 topologies times 2 encoders times 3 conditions."

---


---

## SLIDE 10 — Architecture: MLP vs GNN
⏱ ~1.5 min

🎤
> "This diagram shows the full model per client. Green is what gets sent across the network — 280 parameters only. Blue stays on the client and is never shared.
>
> The key point: when the professor asks whether 1.1 KB is too small — the model is not small. The full model per client is about 135K parameters. LocalEncoder, twin critics, and actor heads are all private. Only the SharedEncoderHead and VQC are federated. So 1.1 KB is the communication cost, not the model capacity."

📌 Point to green boxes. Then point to blue boxes. Emphasise the distinction.

---

## SLIDE 11 — Discussion
⏱ ~3 min

🎤
> "I have three questions for discussion.
> Second — on the GNN. My current result shows MLP beats GNN on static topology. But GNN was designed for dynamic topology — line faults and switching events. So right now I am not comparing them fairly. My plan is to run a fault injection experiment next week to simulate line outages and test if GNN recovers better than MLP in that scenario. My question is — should I run this experiment before submission and potentially promote GNN as a 4th contribution? Or should I keep GNN as an ablation study and proceed with the paper now?
>
> Third — on paper scope. I currently have 3 topologies: 13, 34, and 123-bus. I am planning to add 57-bus as a 4th client. Is 3 topologies sufficient for IEEE TSG, or do I need 4 before submitting?"

📌 Point to each Q as you ask it.

**[Wait for professor response after each question.]**

---

## SLIDE 12 — Next Action
⏱ ~1 min

🎤
> "My plan for the next two weeks.
>
> Tonight — latent alignment eval finishes and I read the final cosine similarity numbers.
>
> This week — fix the corrupted OpenDSS seed 0 result, rerun with 5 seeds, and generate the VQC gradient collapse figure.
>
> Next week — run the fault injection experiment if you advise it, and send you a full paper draft for review.
>
> End of April — submit to IEEE Transactions on Smart Grid."

---

## SLIDE 13 — Q&A
⏱ open

🎤
> "Thank you Professor. I am open to any questions or direction, especially on the three discussion points."

---
---

# PREDICTED Q&A — FULL PREPARATION

---

## GROUP 1 — MODEL SIZE & ARCHITECTURE

### Q: "280 parameters is extremely small. How can a 16-parameter VQC learn anything useful?"

🎤
> "The 16 VQC parameters are the trainable RX rotation angles — one per qubit per layer with 2 layers on 8 qubits. The input encoding is done by the RY gates which take the 8-dimensional latent vector from the SharedEncoderHead. So the VQC is not learning from scratch — it receives a compressed and aligned 8-dimensional representation from 264 parameters in the SharedEncoderHead.
>
> The VQC's role is to apply quantum entanglement to the latent representation and output an 8-dimensional measurement via PauliZ. It is not replacing the full policy network — the actor heads on top are private and contain their own parameters. So the expressiveness comes from the full pipeline, not the VQC alone."

---

### Q: "Why not just use a classical neural network instead of a VQC? What does quantum add here?"

🎤
> "That is the most important question for this work. The practical answer is: the VQC is the federated component — the part that gets shared. A classical network of equivalent expressiveness would have many more parameters, which increases communication cost. The quantum circuit naturally produces a compact parameterisation.
>
> The research question I am investigating is whether the quantum structure — specifically entanglement — provides any representational advantage for power grid control. The ablation results show that removing entanglement makes performance worse, which suggests the quantum structure is contributing something beyond just a compact parameterisation. But I would be honest that proving a clear quantum advantage over classical is still an open question."

---

### Q: "What is heterogeneous FL problem exactly? Is this a new concept or does it appear in prior literature?"

🎤
> "heterogeneous FL problem — Quantum Latent Space Incompatibility — is a problem I identified in this work. It is not directly named in prior literature in this form, but it is related to the heterogeneous FL problem in classical federated learning where clients have different data distributions.
>
> In our case, the incompatibility is geometric: each client has a different observation space dimension because their grid topology is different. Without the aligned encoder, their latent vectors live in incompatible geometric spaces — even though they have the same dimension of 8, the directions mean different things for different clients. When FedAvg averages VQC weights trained on incompatible inputs, the averaged weights produce poor gradients for everyone.
>
> The aligned encoder forces all clients to map into a shared geometric space by jointly training with a consistency loss. This is the key innovation."

---

## GROUP 2 — STATISTICAL VALIDITY

### Q: "n=5 seeds is very small for a statistical claim. How can you report p=0.009 with only 5 samples?"

🎤
> "This is a valid concern. With n=5, the t-test has very low statistical power, and p-values can be unstable. I report p=0.009 as supporting evidence, but my primary claim is the effect size d=+1.74, which is considered a large effect by Cohen's convention.
>
> Effect size is more informative than p-values for small samples because it measures practical significance, not just statistical significance. I also plan to report bootstrap confidence intervals in the paper. I am rerunning OpenDSS with 5 seeds to strengthen the real-physics result. But honestly, the clearest evidence is the personalized FL result — Client B improving from negative 8.075 to negative 3.261 is a 59.6% improvement that is visually clear regardless of sample size."

---

### Q: "Client A shows almost no improvement. Does that weaken the claim?"

🎤
> "Client A is the 13-bus network — the simplest topology. With only 13 buses, the local training alone is already sufficient to learn a good policy. The federated benefit is most valuable when the task is complex enough that sharing information across clients adds new information.
>
> Client B with 34-bus and Client C with 123-bus are more complex, and that is where we see the strongest results. This is actually consistent with the theoretical motivation — the aligned encoder helps most when client heterogeneity is highest."

---

## GROUP 3 — GNN & TOPOLOGY

### Q: "If MLP beats GNN on static topology, why did you implement GNN at all?"

🎤
> "GNN was implemented specifically for dynamic topology scenarios where grid lines can fail or switch during operation. On static topology, the adjacency matrix never changes, so GNN has no structural advantage over MLP — the graph convolution just processes fixed features with fixed connectivity.
>
> The real value of GNN appears when a line trips or a new line connects. In that case, GNN can detect the changed adjacency and adapt, while MLP would see an unexpected input pattern and likely fail. I plan to test this with the fault injection experiment next week. If GNN shows a clear advantage there, it becomes a 4th contribution. If not, it stays as an ablation study."

---

### Q: "Is 3 topologies really enough to claim generalisation?"

🎤
> "The three topologies — 13, 34, and 123-bus — cover a range of 10 to 1 in system size, which I believe represents small distribution network, medium, and large. The aligned encoder handles any observation dimension because the LocalEncoder is private — adding a new topology only requires training a new private LocalEncoder. The shared 280 parameters do not change with topology size.
>
> However, I understand that more topologies would strengthen the claim. I am planning to add 57-bus as a 4th client. I would like your guidance on whether this is necessary before IEEE TSG submission or if 3 is sufficient."

---

## GROUP 4 — IMPLEMENTATION

### Q: "Why default.qubit and not a faster quantum simulator?"

🎤
> "For fair comparison with Lin et al. 2025, which is the base paper this work extends. They used default.qubit. If I switch to lightning.qubit, the results are not directly comparable because different simulators can produce slightly different numerical results. After publication, I can benchmark with faster backends."

---

### Q: "How long did the experiments take to run?"

🎤
> "The main 5-seed FL experiment took approximately 48 hours on a single GPU machine. OpenDSS validation with 3 seeds took 12.1 hours. GNN vs MLP comparison took about 6 hours. The VQC circuit ablation took about 3 hours. The latent alignment evaluation is about 8 hours total — it is still running tonight.
>
> The main bottleneck is the VQC simulation, which uses PennyLane default.qubit. Switching to GPU-accelerated lightning.qubit would reduce this by roughly 10 times."

---

## QUICK REFERENCE — KEY NUMBERS

| Claim | Number |
|---|---|
| Effect size Client B | d = +1.74, p = 0.009 |
| Effect size Client C | d = +1.24, p = 0.025 |
| Effect size Client C (OpenDSS) | d = +1.97, p = 0.054 |
| Personalized FL Client B gain | −8.075 → −3.261  (+59.6%) |
| Federated params | 280 params = 1.1 KB/round |
| vs Classical SAC-FL | 405× smaller communication |
| Full model per client | ~135K params (private) |
| VQC best circuit reward | −5.879 (paper circuit) |
| Seeds | n=5 (linearized), n=3 (OpenDSS) |

---

## OPENING / CLOSING PHRASES

- **To start**: "Good morning Professor. My name is Ing Muyleang..."
- **Transitioning**: "Let me now move to..." / "Before I explain this, let me first clarify..." / "The key point here is..."
- **Emphasising**: "Please pay attention to this number..." / "This is the most important result..."
- **Handling hard questions**: "That is a very valid point. The answer is..." / "That is a limitation I acknowledge..."
- **When unsure**: "That is beyond what this experiment tested, but my hypothesis would be..."
- **Closing**: "Thank you Professor. I am happy to take any questions."
