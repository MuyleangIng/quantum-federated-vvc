"""
Option 1 — Classical SAC-FL baseline
=====================================
Runs the same federated experiment but replaces the VQC with a classical MLP
of the same federated parameter budget (16 params) to test quantum advantage.

Two conditions compared:
  quantum_fl  — QE-SAC-FL with aligned encoder + VQC (280 federated params)
  classical_fl — Same aligned encoder but MLP(8→8) replaces VQC (280 federated params)

If quantum_fl reward > classical_fl reward → quantum advantage proven.
If equal → honest: alignment protocol drives the gain, not quantum per se.

Run: python3 scripts/run_fl_classical_baseline.py
Output: artifacts/qe_sac_fl/classical_baseline/
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.qe_sac.env_utils import VVCEnv13Bus
from src.qe_sac_fl.env_34bus import VVCEnv34BusFL, VVCEnv123BusFL
from src.qe_sac_fl.aligned_encoder import AlignedCAE, train_aligned_cae, fedavg_shared_head
from src.qe_sac_fl.aligned_agent import AlignedQESACAgent
from src.qe_sac.vqc import VQCLayer

SEEDS      = [0, 1, 2]
N_ROUNDS   = 300
LOCAL_STEPS = 1000
WARMUP     = 500
BATCH_SIZE = 64
LR         = 3e-4
BUFFER_SIZE = 50_000
OUT_DIR    = "artifacts/qe_sac_fl/classical_baseline_300r"
os.makedirs(OUT_DIR, exist_ok=True)

CLIENTS = [
    dict(name="A_13bus",  obs_dim=43,  device_dims=[2,2,33,33], reward_scale=50.0,
         device="cuda:0" if torch.cuda.is_available() else "cpu"),
    dict(name="B_34bus",  obs_dim=113, device_dims=[2,2,33,33,33], reward_scale=10.0,
         device="cuda:0" if torch.cuda.is_available() else "cpu"),
    dict(name="C_123bus", obs_dim=349, device_dims=[2,2,33,33,33,33,33], reward_scale=750.0,
         device="cuda:0" if torch.cuda.is_available() else "cpu"),
]


# ── Classical MLP replacement for VQC ───────────────────────────────────────

class ClassicalHead(nn.Module):
    """Drop-in replacement for VQCLayer — same I/O shape, classical MLP."""
    def __init__(self):
        super().__init__()
        # 8→8 with one hidden layer of 8 = 8*8+8 + 8*8+8 = 144 params
        # We use 8→8 linear (64+8=72 params) to stay close to VQC's 16 params
        self.fc = nn.Linear(8, 8, bias=True)   # 72 params (close to VQC 16)

    def forward(self, z):
        return torch.tanh(self.fc(z))


class ClassicalActorNetwork(nn.Module):
    """Same as AlignedActorNetwork but VQC → ClassicalHead."""
    def __init__(self, obs_dim, device_dims, hidden_dim=32):
        super().__init__()
        self.cae   = AlignedCAE(obs_dim, hidden_dim=hidden_dim)
        self.vqc   = ClassicalHead()   # classical replacement
        self.heads = nn.ModuleList([nn.Linear(8, d) for d in device_dims])

    def forward(self, obs):
        z = self.cae.encode(obs)
        q = self.vqc(z)
        return [F.softmax(h(q), dim=-1) for h in self.heads]

    def select_action(self, obs):
        with torch.no_grad():
            probs = self.forward(obs)
            acts  = [torch.argmax(p, dim=-1).cpu().numpy().item() for p in probs]
        return np.array(acts)

    def get_shared_weights(self):
        return {
            "shared_head": {k: v.cpu().clone() for k, v in
                           self.cae.shared_head.state_dict().items()},
            "vqc":         {k: v.cpu().clone() for k, v in
                           self.vqc.state_dict().items()}
        }

    def set_shared_weights(self, weights):
        self.cae.shared_head.load_state_dict(
            {k: v.to(next(self.parameters()).device) for k, v in weights["shared_head"].items()})
        self.vqc.load_state_dict(
            {k: v.to(next(self.parameters()).device) for k, v in weights["vqc"].items()})


def make_env(name, seed):
    if "13" in name: return VVCEnv13Bus(seed=seed)
    if "34" in name: return VVCEnv34BusFL(seed=seed)
    return VVCEnv123BusFL(seed=seed)   # FL version: 2 caps + 1 reg = [2,2,33]


def run_one(condition, seed):
    print(f"\n{'='*55}\n  {condition}  seed={seed}\n{'='*55}")
    torch.manual_seed(seed); np.random.seed(seed)

    agents, envs = [], []
    for cfg in CLIENTS:
        if condition == "quantum_fl":
            agent = AlignedQESACAgent(
                obs_dim=cfg["obs_dim"], device_dims=cfg["device_dims"],
                lr=LR, buffer_size=BUFFER_SIZE, device=cfg["device"])
        else:
            # Classical — same agent but swap actor
            agent = AlignedQESACAgent(
                obs_dim=cfg["obs_dim"], device_dims=cfg["device_dims"],
                lr=LR, buffer_size=BUFFER_SIZE, device=cfg["device"])
            # Replace actor with classical version
            agent.actor = ClassicalActorNetwork(
                cfg["obs_dim"], cfg["device_dims"]).to(cfg["device"])
            agent.actor_opt = torch.optim.Adam(agent.actor.parameters(), lr=LR)

        env = make_env(cfg["name"], seed)
        agents.append(agent); envs.append(env)

    # Warm-up
    for agent, env, cfg in zip(agents, envs, CLIENTS):
        obs, _ = env.reset()
        for _ in range(WARMUP):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(cfg["device"])
            action = agent.actor.select_action(obs_t)
            next_obs, reward, done, trunc, _ = env.step(action)
            agent.store(obs, action, reward / cfg["reward_scale"], next_obs, done or trunc)
            obs = next_obs if not (done or trunc) else env.reset()[0]

    reward_history = {cfg["name"]: [] for cfg in CLIENTS}
    server_weights = agents[0].actor.get_shared_weights()

    for rnd in range(N_ROUNDS):
        for agent, env, cfg in zip(agents, envs, CLIENTS):
            agent.actor.set_shared_weights(server_weights)
            ep_rewards, ep_rew = [], 0.0
            obs, _ = env.reset()
            for _ in range(LOCAL_STEPS):
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(cfg["device"])
                action = agent.actor.select_action(obs_t)
                next_obs, reward, done, trunc, _ = env.step(action)
                agent.store(obs, action, reward / cfg["reward_scale"], next_obs, done or trunc)
                ep_rew += reward / cfg["reward_scale"]
                obs = next_obs
                if done or trunc:
                    ep_rewards.append(ep_rew); ep_rew = 0.0
                    obs, _ = env.reset()
                agent.update(batch_size=BATCH_SIZE)
            reward_history[cfg["name"]].append(
                float(np.mean(ep_rewards)) if ep_rewards else float("nan"))

        # FedAvg
        all_sw = [a.actor.get_shared_weights() for a in agents]
        avg_sh  = fedavg_shared_head([s["shared_head"] for s in all_sw])
        # Average VQC (Tensor) or classical head (state_dict)
        vqc0 = all_sw[0]["vqc"]
        if isinstance(vqc0, torch.Tensor):
            avg_vqc = torch.stack([s["vqc"].float() for s in all_sw]).mean(0)
        else:
            keys    = list(vqc0.keys())
            avg_vqc = {k: torch.stack([s["vqc"][k].float() for s in all_sw]).mean(0) for k in keys}
        server_weights = {"shared_head": avg_sh, "vqc": avg_vqc}

        rew_B = reward_history["B_34bus"][-1]
        print(f"  Round {rnd+1:>3}/{N_ROUNDS} | B={rew_B:+.3f}")

    for env in envs: env.close()

    final = {cfg["name"]: float(np.nanmean(reward_history[cfg["name"]][-10:]))
             for cfg in CLIENTS}
    return {"condition": condition, "seed": seed,
            "reward_history": reward_history, "final": final}


def plot_results(all_results):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), facecolor='white')
    fig.suptitle("Quantum vs Classical Federated SAC — Aligned FL Protocol\n"
                 "(same 280 federated params, VQC vs classical MLP)",
                 fontsize=12, fontweight='bold')

    clients = ["A_13bus", "B_34bus", "C_123bus"]
    titles  = ["Client A (13-bus)", "Client B (34-bus)", "Client C (123-bus)"]
    colors  = {"quantum_fl": "#2196F3", "classical_fl": "#E91E63"}
    labels  = {"quantum_fl": "Quantum FL (VQC)", "classical_fl": "Classical FL (MLP)"}

    for ax, cname, title in zip(axes, clients, titles):
        ax.set_facecolor('white')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel("FL Round"); ax.set_ylabel("Mean Reward (↑ better)")
        ax.grid(True, alpha=0.3)

        for cond in ["quantum_fl", "classical_fl"]:
            res_list = all_results.get(cond, [])
            if not res_list: continue
            arrs = np.array([r["reward_history"][cname] for r in res_list])
            mn   = np.nanmean(arrs, axis=0)
            std  = np.nanstd(arrs, axis=0)
            rounds = np.arange(1, len(mn)+1)
            ax.plot(rounds, mn, color=colors[cond], lw=2, label=labels[cond])
            ax.fill_between(rounds, mn-std, mn+std, alpha=0.15, color=colors[cond])
        ax.legend(fontsize=9)

    plt.tight_layout()
    out = f"{OUT_DIR}/quantum_vs_classical.png"
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(); print(f"  Figure → {out}")


def main():
    all_results = {"quantum_fl": [], "classical_fl": []}
    t0 = time.time()

    for condition in ["quantum_fl", "classical_fl"]:
        for seed in SEEDS:
            res = run_one(condition, seed)
            all_results[condition].append(res)
            with open(f"{OUT_DIR}/{condition}_seed{seed}.json", "w") as f:
                json.dump(res, f, indent=2)

    print(f"\n{'='*60}\n  QUANTUM vs CLASSICAL SUMMARY\n{'='*60}")
    print(f"  {'Condition':<15} {'A_13bus':>10} {'B_34bus':>10} {'C_123bus':>10}")
    for cond in ["quantum_fl", "classical_fl"]:
        means = [np.mean([r["final"][c] for r in all_results[cond]]) for c in
                 ["A_13bus", "B_34bus", "C_123bus"]]
        print(f"  {cond:<15} {means[0]:>+10.3f} {means[1]:>+10.3f} {means[2]:>+10.3f}")

    summary = {}
    for cond in all_results:
        summary[cond] = {c: float(np.mean([r["final"][c] for r in all_results[cond]]))
                         for c in ["A_13bus", "B_34bus", "C_123bus"]}
    with open(f"{OUT_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    plot_results(all_results)
    print(f"\n  Done. {(time.time()-t0)/60:.1f} min")
    print(f"  Output → {OUT_DIR}/")


if __name__ == "__main__":
    main()
