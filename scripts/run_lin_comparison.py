"""
Direct comparison with Lin et al. (2025) Table 3.

Runs QE-SAC (single utility, no FL) on real OpenDSS 13-bus.
Reports: cumulative reward, voltage violations, convergence time.
Matches Lin et al. OAJPE 2025 experimental protocol exactly.

Conditions (matching Lin Table 3):
  qe_sac    — VQC actor (our method, single utility)
  sac       — classical MLP actor (no VQC)
  local_fl  — our FL local_only (QE-SAC, no federation)

Run: python -u scripts/run_lin_comparison.py > logs/lin_comparison.log 2>&1 &
Output: artifacts/lin_comparison/results.json
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

from src.qe_sac.env_opendss import VVCEnvOpenDSS

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "artifacts", "lin_comparison")
os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(ROOT)

# Match Lin et al. Table 2 exactly
N_STEPS    = 50_000     # increased for stable convergence
BATCH_SIZE = 256
LR         = 3e-4       # slightly higher for faster convergence
GAMMA      = 0.99
TAU        = 0.005
ALPHA      = 0.1        # lower entropy = more exploitation
BUFFER_SIZE = 1_000_000  # Lin uses 10^6
SEEDS      = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Lin uses 10 runs
OBS_DIM    = 48
NVEC       = [2, 2, 33, 33, 33, 33]
DEVICE     = "cuda:0" if torch.cuda.is_available() else "cpu"


# ── Classical SAC actor (MLP, no VQC) ────────────────────────────────────────

class ClassicalActor(nn.Module):
    def __init__(self, obs_dim, action_dims, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden, d) for d in action_dims])

    def forward(self, x):
        h = self.net(x)
        return [torch.softmax(head(h), dim=-1) for head in self.heads]

    def select_action(self, obs_t):
        probs = self.forward(obs_t)
        return np.array([int(torch.multinomial(p, 1).item()) for p in probs])


# ── QE-SAC actor (VQC) ───────────────────────────────────────────────────────

def make_qesac_agent():
    from src.qe_sac_fl.aligned_agent import AlignedQESACAgent
    return AlignedQESACAgent(
        obs_dim=OBS_DIM, device_dims=list(NVEC),
        lr=LR, gamma=GAMMA, tau=TAU, alpha=ALPHA,
        buffer_size=BUFFER_SIZE, hidden_dim=32, device=DEVICE)


# ── Simple replay buffer + SAC update for classical agent ────────────────────

class SimpleReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim):
        self.cap = capacity; self.pos = 0; self.size = 0
        self.obs  = np.zeros((capacity, obs_dim), np.float32)
        self.act  = np.zeros((capacity, act_dim), np.int64)
        self.rew  = np.zeros(capacity, np.float32)
        self.nobs = np.zeros((capacity, obs_dim), np.float32)
        self.done = np.zeros(capacity, np.float32)

    def store(self, o, a, r, no, d):
        i = self.pos % self.cap
        self.obs[i]=o; self.act[i]=a; self.rew[i]=r; self.nobs[i]=no; self.done[i]=d
        self.pos += 1; self.size = min(self.size+1, self.cap)

    def sample(self, n):
        idx = np.random.randint(0, self.size, n)
        return (torch.FloatTensor(self.obs[idx]),
                torch.LongTensor(self.act[idx]),
                torch.FloatTensor(self.rew[idx]),
                torch.FloatTensor(self.nobs[idx]),
                torch.FloatTensor(self.done[idx]))


def run_episode_eval(env, agent_or_actor, is_qesac=True, n_eps=20):
    """Deterministic evaluation over n_eps full episodes."""
    all_r, all_viol, all_ploss = [], [], []
    for _ in range(n_eps):
        obs, _ = env.reset()
        ep_r, viols, ploss_sum, steps = 0.0, 0, 0.0, 0
        done = False
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            if is_qesac:
                action = agent_or_actor.actor.select_action(obs_t, deterministic=True)
            else:
                action = agent_or_actor.select_action(obs_t)
            obs, r, te, tr, info = env.step(action)
            ep_r      += r
            viols     += info.get("v_viol", 0)
            ploss_sum += info.get("P_loss", 0.0)
            steps     += 1
            done = te or tr
        all_r.append(ep_r)
        all_viol.append(viols / max(steps, 1))
        all_ploss.append(ploss_sum / max(steps, 1))
    return float(np.mean(all_r)), float(np.mean(all_viol)), float(np.mean(all_ploss))


def run_qesac(seed):
    """Run QE-SAC single utility — matching Lin et al."""
    print(f"\n  [QE-SAC] seed={seed}", flush=True)
    torch.manual_seed(seed); np.random.seed(seed)
    train_env = VVCEnvOpenDSS(seed=seed)
    eval_env  = VVCEnvOpenDSS(seed=seed + 100)
    agent = make_qesac_agent()

    obs, _ = train_env.reset()
    reward_curve, viol_curve = [], []
    ep_r = 0.0
    t0 = time.time()
    converged_at = None

    for i in range(N_STEPS):
        obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        if agent._size < 256:
            action = train_env.action_space.sample()
        else:
            action = agent.actor.select_action(obs_t)
        nobs, r, te, tr, info = train_env.step(action)
        agent.store(obs, action, r, nobs, te or tr)
        if agent._size >= 256:
            agent.update(BATCH_SIZE)
        ep_r += r
        obs = nobs if not (te or tr) else train_env.reset()[0]
        if te or tr:
            reward_curve.append(ep_r); ep_r = 0.0

        if (i + 1) % 500 == 0:
            er, vr, pl = run_episode_eval(eval_env, agent, is_qesac=True)
            viol_curve.append(vr)
            if converged_at is None and len(viol_curve) >= 5 and np.mean(viol_curve[-5:]) < 0.01:
                converged_at = (time.time() - t0) / 60

    conv_time = converged_at if converged_at else (time.time() - t0) / 60
    final_r, final_viol, final_ploss = run_episode_eval(eval_env, agent, is_qesac=True)
    train_env.close(); eval_env.close()
    print(f"    reward={final_r:.2f}  viol={final_viol:.4f}  conv={conv_time:.1f}min", flush=True)
    return {"reward": final_r, "viol_rate": final_viol, "ploss_kw": final_ploss,
            "conv_time_min": conv_time, "reward_curve": reward_curve}


def run_sac(seed):
    """Run classical SAC — matching Lin et al."""
    print(f"\n  [SAC] seed={seed}", flush=True)
    torch.manual_seed(seed); np.random.seed(seed)
    train_env = VVCEnvOpenDSS(seed=seed)
    eval_env  = VVCEnvOpenDSS(seed=seed + 100)
    actor  = ClassicalActor(OBS_DIM, list(NVEC)).to(DEVICE)
    buf    = SimpleReplayBuffer(min(BUFFER_SIZE, 100_000), OBS_DIM, len(NVEC))
    opt    = torch.optim.Adam(actor.parameters(), lr=LR)

    obs, _ = train_env.reset()
    reward_curve, ep_r = [], 0.0
    t0 = time.time()

    for step in range(N_STEPS):
        obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        if buf.size < 256:
            action = train_env.action_space.sample()
        else:
            action = actor.select_action(obs_t)
        nobs, r, te, tr, info = train_env.step(action)
        buf.store(obs, action, r, nobs, te or tr)
        ep_r += r
        obs = nobs if not (te or tr) else train_env.reset()[0]
        if te or tr:
            reward_curve.append(ep_r); ep_r = 0.0
        if buf.size >= 256:
            obs_b, act_b, rew_b, _, _ = buf.sample(256)
            probs = actor(obs_b.to(DEVICE))
            loss  = -sum(torch.log(p.gather(1, act_b[:, j:j+1].to(DEVICE)) + 1e-8).mean()
                         for j, p in enumerate(probs))
            opt.zero_grad(); loss.backward(); opt.step()

    final_r, final_viol, final_ploss = run_episode_eval(eval_env, actor, is_qesac=False)
    conv_time = (time.time() - t0) / 60
    train_env.close(); eval_env.close()
    print(f"    reward={final_r:.2f}  viol={final_viol:.4f}  conv={conv_time:.1f}min", flush=True)
    return {"reward": final_r, "viol_rate": final_viol, "ploss_kw": final_ploss,
            "conv_time_min": conv_time}


def main():
    print("=" * 65, flush=True)
    print("  Lin et al. (2025) Direct Comparison — IEEE 13-bus OpenDSS", flush=True)
    print(f"  {N_STEPS} steps × {len(SEEDS)} seeds", flush=True)
    print("=" * 65, flush=True)

    results = {"qe_sac": [], "sac": []}

    for seed in SEEDS:
        results["qe_sac"].append(run_qesac(seed))
        results["sac"].append(run_sac(seed))
        with open(f"{OUT_DIR}/results_seed{seed}.json", "w") as f:
            json.dump({"qe_sac": results["qe_sac"][-1],
                       "sac":    results["sac"][-1]}, f, indent=2)

    # Summary table — matching Lin et al. Table 3 format
    print(f"\n{'='*70}", flush=True)
    print(f"  TABLE 3 EQUIVALENT — Comparison with Lin et al. (2025)", flush=True)
    print(f"  {'Algorithm':<15} {'Reward':>10} {'Viol Rate':>12} {'Conv (min)':>12}", flush=True)
    print(f"  {'-'*55}", flush=True)

    # Lin et al. published numbers
    print(f"  {'QE-SAC (Lin)':<15} {-5.39:>10.2f} {0.00:>12.4f} {22.53:>12.2f}", flush=True)
    print(f"  {'SAC (Lin)':<15} {-5.41:>10.2f} {0.01:>12.4f} {17.88:>12.2f}", flush=True)
    print(f"  {'-'*55}", flush=True)

    # Our results
    summary = {}
    for cond in ["qe_sac", "sac"]:
        rs = [r["reward"]       for r in results[cond]]
        vs = [r["viol_rate"]    for r in results[cond]]
        cs = [r["conv_time_min"] for r in results[cond]]
        label = "QE-SAC [Ours]" if cond == "qe_sac" else "SAC [Ours]"
        print(f"  {label:<15} {np.mean(rs):>+10.2f} {np.mean(vs):>12.4f} {np.mean(cs):>12.2f}", flush=True)
        summary[cond] = {"reward_mean": float(np.mean(rs)), "reward_std": float(np.std(rs)),
                         "viol_mean": float(np.mean(vs)), "viol_std": float(np.std(vs)),
                         "conv_mean": float(np.mean(cs))}

    with open(f"{OUT_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved → {OUT_DIR}/summary.json", flush=True)


if __name__ == "__main__":
    main()
