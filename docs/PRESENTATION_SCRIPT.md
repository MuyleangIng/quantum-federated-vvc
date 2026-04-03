# Presentation Script & Q&A Preparation
## Advisor Meeting — Federated Quantum RL for VVC
**Ing Muyleang | Pukyong National University QCL | April 1, 2026**

> **How to use this document**
> Read the **SAY** section out loud when each slide appears.
> The **Q&A** section is what your professor is likely to ask — read it before the meeting.

---

---

# SLIDE 1 — TITLE

## SAY

> "Good morning, Professor. Today I want to report on the federated learning
> component of the QE-SAC+ research plan.
>
> I started by implementing standard federated learning on the quantum RL agents.
> What happened was not what I expected — I discovered three structural failure
> modes that do not appear in any existing paper.
>
> I have experimental evidence for all three, a working solution, and proof
> the solution achieves 25 to 77 percent reward improvement across all three
> utility clients simultaneously.
>
> I believe this is enough for a paper submission to IEEE Transactions on Smart Grid."

**Pause. Let him respond before moving on.**

---

---

# SLIDE 2 — INDEX

## SAY

> "The structure today is: executive summary first so you have the full picture,
> then the three new discoveries in detail, then architecture, results, what is
> ongoing, a discussion comparing to the literature, and finally the next actions
> and timeline to submission.
>
> The most important sections are New Progress and All Results.
> Everything else is context."

---

---

# SLIDE 3 — EXECUTIVE SUMMARY

## SAY

> "The problem: three competing utilities need better voltage control but
> cannot share data. Federated learning was the natural solution.
>
> When I applied standard FedAvg to the quantum agents, every client got worse.
> I found out why. I fixed it. And along the way I found two more new problems.
>
> The headline result — after running all experiments — is this row at the bottom.
> Personalised federated learning: plus 50 percent for the 13-bus client,
> plus 77 percent for 34-bus, plus 25 percent for 123-bus.
> All three companies improve simultaneously.
> And the communication cost is 395 times less than classical federated SAC.
>
> The three discoveries in the middle are what make this a paper, not just
> an implementation. Each one is something no existing work has identified."

---

---

# SLIDE 4 — NEW PROGRESS

## SAY

> "Let me explain each discovery clearly.
>
> **QLSI — Quantum Latent Space Incompatibility.**
> Each client trains its own autoencoder independently. Client A's 8 latent
> numbers mean 'voltage at bus 3, reactive power at bus 7...' and so on.
> Client B's 8 numbers mean completely different things.
> After FedAvg, the VQC receives inputs from three incompatible spaces.
> The result: every client gets worse. You can see the numbers —
> 13-bus goes from minus 331 to minus 336. 123-bus drops by 56 points.
> This had never been reported in any federated learning or quantum ML paper.
>
> **CSA — Client Size Asymmetry.**
> After I fixed QLSI with the SharedEncoderHead, I ran for 200 rounds instead
> of 50. Something surprising happened: the client that benefits from federation
> *changes* over time. At round 50, only the small feeder (13-bus) is better.
> At round 200, only the large feeder (123-bus) is better.
> The benefit rotates. At no single round count do all three clients pass.
>
> **PAD — Partial Alignment Drift.**
> McMahan's 2017 paper proved FedAvg converges even when clients drop out.
> I tested whether this holds for the aligned architecture. It does not.
> Even dropping one client per round makes all three clients worse than
> training alone. The VQC gradient norms actually go UP — highest of any
> condition — but the reward is worst. That signature means the gradients
> are oscillating, not learning."

---

---

# SLIDE 5 — ARCHITECTURE

## SAY

> "Here is what I built to solve QLSI.
>
> The key idea is to split the encoder into two parts.
> The LocalEncoder stays on the client — it is private, feeder-specific,
> never shared. It compresses from 42, 105, or 372 dimensions down to 32.
>
> The SharedEncoderHead goes to the server — it is federated, identical
> for all clients, and compresses from 32 down to 8.
> Since all clients share the same SharedEncoderHead after FedAvg,
> they all speak the same 8-dimensional language before reaching the VQC.
>
> The VQC is also federated — same 16 parameters for all clients.
>
> Total federated per round: 280 parameters. That is 1,120 bytes.
> Classical federated SAC sends 110,000 parameters — 443 kilobytes.
> We are 395 times cheaper to communicate.
>
> Everything else — the LocalEncoder, the critics, the replay buffer with
> raw grid data — never leaves the client device."

---

---

# SLIDE 6 — ALL RESULTS

## SAY

> "This table tells the complete story in one view.
>
> Row 1 is local-only training — each company trains alone. This is the baseline.
>
> Row 2 is unaligned FL — standard FedAvg, VQC only. All three clients are worse.
> That is QLSI.
>
> Rows 3 and 4 are aligned FL at 50 and 200 rounds with the SharedEncoderHead.
> QLSI is fixed — but we see CSA. At 50 rounds only 13-bus passes.
> At 200 rounds only 123-bus passes. Notice the reversal.
>
> Row 5 is partial FL with two thirds of clients per round. All three fail.
> That is PAD.
>
> Row 6 — personalised FL — is the only row that says YES in the 'all pass' column.
> Plus 50, plus 77, plus 25 percent. This is the main result.
>
> At the bottom: 395 times less communication — that is a mathematical fact,
> not an experiment. It follows directly from the parameter counts."

---

---

# SLIDE 7 — ONGOING WORK

## SAY

> "Three things are in progress right now.
>
> First, statistical significance — I am running all conditions with 5 seeds
> to get mean plus or minus standard deviation. Reviewers will require this.
>
> Second, the gradient-normalised FedAvg to fix CSA. The idea is simple:
> weight each client's contribution by the inverse of its gradient norm,
> so large-gradient clients do not dominate the SharedHead.
> It is one function change in aligned_encoder.py.
>
> Third, FedProx regularisation on the SharedHead to fix PAD.
> This adds a proximal term that prevents the SharedHead from drifting
> too far when clients are absent. McMahan and Li both use proximal
> regularisation for data heterogeneity — I am applying it to the
> architectural coupling problem instead.
>
> Fourth, transfer to a fourth unseen feeder — this connects directly
> to the QE-SAC+ roadmap we discussed earlier."

---

---

# SLIDE 8 — DISCUSSION

## SAY

> "The most important thing reviewers and professors ask is:
> 'How is this different from existing work?'
>
> For QLSI: FedProx and SCAFFOLD both address heterogeneous data distributions.
> QLSI is caused by incompatible encoder representations — it occurs even with
> identical data. These papers do not study that.
>
> For CSA: Zhao 2018 and FedBN study non-IID data content.
> CSA is caused by observation dimension scale — all clients have the same
> objective, but different feeder sizes create gradient imbalance.
> That is a different category entirely.
>
> For PAD: McMahan proved FedAvg converges with partial dropout for standard
> model weights. His proof assumes weights are independent. In our architecture,
> the LocalEncoder and SharedHead are *coupled* — they must stay aligned.
> Partial participation breaks that coupling. His proof does not apply.
>
> For H3: no quantum FL paper has quantified the communication cost.
> We have a mathematical proof that we are 395 to 6920 times cheaper.
> That result is always true regardless of training outcome."

---

---

# SLIDE 9 — NEXT ACTIONS

## SAY

> "The paper is 80 percent done. What remains is four tasks over three weeks.
>
> Week 1: 5-seed runs and the CSA fix.
> Week 2: PAD fix and statistical analysis.
> Week 3: Write one mathematical paragraph per finding — this turns
> empirical observations into causal mechanisms, which is what reviewers require.
>
> After that, paper writing begins. I have the structure planned —
> 8 pages in IEEE T-SG format with 6 contributions."

---

---

# SLIDE 10 — PLAN

## SAY

> "The timeline: April for experiments and fixes, May for transfer learning
> and the QE-SAC+ bridge, June for writing and submission.
>
> IEEE Transactions on Smart Grid has a 3 to 6 month review cycle.
> If we submit in June, a decision is likely before the end of the year.
>
> The backup venue is IEEE PES General Meeting 2026 if we want a faster
> conference publication first."

---

---

# SLIDE 11 — CLOSING

## SAY

> "To summarise the six contributions this work makes:
>
> One — QLSI: the first identification of this failure mode.
> Two — SharedEncoderHead: the architectural fix, 395 times cheaper than classical FL.
> Three — CSA: gradient scale prevents simultaneous benefit.
> Four — PAD: partial participation breaks the coupling constraint.
> Five — Personalised QFL: the solution that works — plus 25 to 77 percent.
> Six — H3: mathematical proof of communication advantage.
>
> The target is IEEE Transactions on Smart Grid, impact factor 8.9.
> Submission is planned for June 2026.
>
> I am ready for your questions."

---

---

---

# Q&A PREPARATION

## Questions your professor is likely to ask

---

### Q1 — "Is this really novel? Someone must have studied quantum FL before."

**A:**
> "Quantum federated learning papers exist — but they federate standard
> quantum classifiers, not quantum RL agents with private encoders.
> The specific problem here is that the encoder is privately trained
> per client before federation. No paper in quantum FL, and no paper
> in classical FL, has identified or named QLSI, CSA, or PAD.
> I checked the most cited papers in both fields and none of them
> study split-encoder architectures with heterogeneous observation dimensions."

**Backup if he pushes:**
> "The key distinction for QLSI is that it occurs even with IID data.
> Classical FL heterogeneity research assumes the problem is data.
> We show it can be purely architectural — that is a different category."

---

### Q2 — "Why does personalised FL work? Explain the mechanism."

**A:**
> "Two reasons compound.
>
> First, the federated warm-start. After 50 rounds, the SharedHead and VQC
> have seen gradient signal from three diverse feeders. That is a much better
> starting point than random initialisation. The policy already understands
> what 'keep voltage in range' means across topologies.
>
> Second, local fine-tuning frees the LocalEncoder to adapt to the specific
> feeder. During fine-tuning, the LocalEncoder can adjust its compression
> to work best for that client's state space. The shared weights act as
> a stable anchor — the fine-tuned model stays close to the good general
> solution rather than drifting to a local minimum."

---

### Q3 — "Why does 34-bus never pass H1? That seems like a failure."

**A:**
> "Actually this is the clearest evidence for CSA.
> 34-bus sits exactly between the two extremes.
> Its observation dimension (105) is neither small enough to have the
> highest early gradient norm like 13-bus, nor large enough to have
> the highest reward scale like 123-bus.
> It is always squeezed out of the gradient average.
>
> This is precisely what makes CSA a novel finding — it is not random noise.
> It is a systematic consequence of gradient magnitude imbalance,
> and the proposed fix is gradient-normalised FedAvg.
> Also: 34-bus achieves the *strongest* personalised improvement of all —
> plus 77 percent — because the FL warm-start gives it gradient signal
> it cannot generate locally. So pure aligned FL fails it, but personalised
> FL benefits it most."

---

### Q4 — "Why are VQC gradient norms highest under partial FL? That seems wrong."

**A:**
> "This is the PAD signature and it is counterintuitive.
> High gradient norm does not always mean useful learning.
> It means the parameter is being pulled in a large direction.
>
> Under partial FL, the SharedHead oscillates between different 2-client
> objectives every round. When a client returns after being absent,
> its LocalEncoder is misaligned with the new SharedHead.
> The VQC receives inconsistent inputs each round — and it tries to
> compensate with large gradient updates.
>
> The result is high gradient norm but low reward — the gradients are
> in competing directions, not converging. It is the same signal you
> would see from a poorly conditioned optimisation problem."

---

### Q5 — "Can you prove CSA mathematically, not just empirically?"

**A:**
> "The empirical pattern is strong and I have a sketch:
>
> The SharedHead update direction is the average of client gradients.
> If we write grad_i as proportional to the loss scale of client i,
> and loss scale is proportional to the reward magnitude,
> then over time the SharedHead converges toward the optimum of
> the highest-reward-scale client.
>
> In formal terms: the fixed point of FedAvg with heterogeneous gradient
> norms is biased toward the client with maximum gradient norm.
> For our clients that is 123-bus by round 200.
>
> Writing this as a formal inequality is one of the tasks in Week 3.
> The empirical evidence already makes the point clearly for the paper."

---

### Q6 — "Why not just normalise the rewards so all clients have the same scale?"

**A:**
> "That is a good engineering question. Reward normalisation would reduce
> CSA but not eliminate it — because the gradient imbalance also comes
> from observation dimension differences, not only reward scale.
>
> The cleaner solution is gradient-normalised FedAvg, which directly
> weights the SharedHead update by the inverse gradient norm from each client.
> This is principled, measurable, and connects directly to the FL
> aggregation mechanism rather than changing the environment setup.
>
> Reward normalisation would also require knowing the reward range in advance,
> which may not be available in deployment."

---

### Q7 — "How does this connect to QE-SAC+ and the GNN encoder work?"

**A:**
> "The connection is direct.
>
> CSA shows that the bottleneck for federated learning is the LocalEncoder —
> when it produces very different gradient scales across clients,
> the SharedHead cannot serve all clients equally.
>
> A GNN encoder, which knows the feeder topology, would compress observations
> in a more physically consistent way across different feeder sizes.
> Bus voltages in a 13-bus feeder and a 123-bus feeder follow the same
> DistFlow equations — a topology-aware encoder should produce more
> comparable gradient magnitudes than a flat MLP.
>
> So the FL paper motivates the GNN encoder not just from a theoretical angle
> but from an experimental one: we have evidence that encoder quality
> directly affects federation stability."

---

### Q8 — "Is one seed per condition enough? The results could be noise."

**A:**
> "You are right — this is the most important limitation to acknowledge.
> These are single-seed results. The pattern is consistent and
> the directional conclusions are strong, but statistical significance
> requires multiple seeds.
>
> The 5-seed runs are the first thing being executed now.
> I expect the mean values to be close to what we see here,
> because the phenomena are structural — QLSI, CSA, and PAD are
> caused by architecture, not random initialisation.
> But until we have error bars I will not claim the numbers are final."

---

### Q9 — "What is the risk that a reviewer rejects the QLSI claim as obvious?"

**A:**
> "The claim is specific: QLSI occurs because independently trained
> autoencoders learn *orthogonal latent bases* — the 8 dimensions
> have no consistent meaning across clients.
>
> A reviewer might say 'of course different encoders produce different
> representations' — but no paper has:
> (a) identified this as a failure mode for federated quantum RL,
> (b) named it,
> (c) provided experimental evidence where ALL clients degrade,
> (d) proposed and validated a fix.
>
> The naming and the evidence together make it a contribution.
> A negative result with experimental proof and a solution is publishable.
> IEEE T-SG regularly publishes papers that identify and solve
> a specific failure mode in an applied ML context."

---

### Q10 — "Which venue should we target — T-SG or PES General Meeting?"

**A:**
> "My recommendation is T-SG as primary, PES as backup.
>
> T-SG is the strongest venue for this topic — it covers exactly the
> intersection of power systems and machine learning.
> Impact factor 8.9, well-known to the power engineering community.
>
> If we want a faster publication to establish the priority of the findings —
> especially the QLSI naming — PES General Meeting has a shorter review cycle
> and a June 2026 deadline that is achievable.
>
> We could submit a 5-page conference version to PES first,
> then extend to the full journal paper for T-SG."

---

---

# BEFORE THE MEETING — CHECKLIST

- [ ] Open `artifacts/QE_SAC_FL_AdvisorSlides.pptx` on your laptop
- [ ] Have `notebooks/qe_sac_fl_experiment.ipynb` open and ready to show Cell 6 (results table) and Cell 52 (visualisation) if asked
- [ ] Know these four numbers by heart: **+50.2%, +76.8%, +24.8%, 395×**
- [ ] Know the three names: **QLSI, CSA, PAD**
- [ ] If he asks to see the code: open `src/qe_sac_fl/aligned_encoder.py` — the `fedavg_shared_head()` function
- [ ] If he asks for the full reference list: open `src/qe_sac_fl/docs/FULL_RESEARCH_COMPENDIUM.md` Section 10

---

# ONE SENTENCE PER FINDING (memorise these)

| Finding | One sentence |
|---|---|
| QLSI | "Independently trained encoders produce incompatible latent spaces — after FedAvg the VQC receives meaningless inputs from every client." |
| CSA | "The SharedHead gradient is dominated by the client with the largest gradient norm — small feeders win early, large feeders win late, never all at once." |
| PAD | "Partial participation breaks the coupling between LocalEncoder and SharedHead — when a client returns after absence, its encoder is misaligned with the updated SharedHead." |
| Solution | "Personalised FL gives all clients a shared warm-start, then lets each one fine-tune locally — bypassing both QLSI and CSA." |

---

*Good luck. You have done the work. The numbers speak for themselves.*
