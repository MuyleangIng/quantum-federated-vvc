"""
Consolidate all experiment results into one summary table.

Reads:
    artifacts/qe_sac/results_13bus.json        (DistFlow baselines)
    artifacts/qe_sac/h1_variance_analysis.json  (H1 variance)
    artifacts/qe_sac/h2_cae_freeze_ablation.json (H2 ablation)
    artifacts/qe_sac/constrained_sac_seeds12.json (Task 2)
    artifacts/qe_sac/opendss_results_13bus.json  (OpenDSS comparison)

Saves:
    artifacts/qe_sac/full_results_summary.json

Usage:
    python scripts/consolidate_results.py
"""

import sys, json, os
sys.path.insert(0, "/root/power-system")

SAVE_DIR = "artifacts/qe_sac"


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def main():
    summary = {}

    # ─── DistFlow baselines ─────────────────────────────────────────────────
    baseline = load_json(os.path.join(SAVE_DIR, "results_13bus.json"))
    if baseline:
        summary["DistFlow_QE-SAC"] = {
            "env":        "DistFlow (42-dim)",
            "seeds":       3,
            "mean_reward": baseline["QE-SAC"]["mean_reward"],
            "std_reward":  baseline["QE-SAC"]["std_reward"],
            "n_params":    baseline["QE-SAC"]["n_params"],
            "safety":      "soft",
        }
        summary["DistFlow_Classical-SAC"] = {
            "env":        "DistFlow (42-dim)",
            "seeds":       3,
            "mean_reward": baseline["Classical SAC"]["mean_reward"],
            "std_reward":  baseline["Classical SAC"]["std_reward"],
            "n_params":    baseline["Classical SAC"]["n_params"],
            "safety":      "none",
        }

    # ─── H1 variance analysis ──────────────────────────────────────────────
    h1 = load_json(os.path.join(SAVE_DIR, "h1_variance_analysis.json"))
    if h1:
        summary["H1"] = {
            "finding":       "QE-SAC more stable than Classical SAC",
            "variance_ratio": h1["variance_ratio"],
            "qe_std":         h1["qe_sac"]["std"],
            "classical_std":  h1["classical_sac"]["std"],
            "supported":      h1["h1_supported"],
        }

    # ─── H2 CAE ablation ──────────────────────────────────────────────────
    h2 = load_json(os.path.join(SAVE_DIR, "h2_cae_freeze_ablation.json"))
    if h2:
        summary["H2"] = {
            "finding":       "CAE co-adaptation is stability source",
            "reward_drop":    h2["reward_drop"],
            "std_change":     h2["std_change"],
            "supported":      h2["h2_supported"],
        }

    # ─── Constrained SAC (Task 2) ──────────────────────────────────────────
    # Try full results first, then partial (seeds 1-2 only)
    constrained_full = load_json(os.path.join(SAVE_DIR, "task2_constrained_results.json"))
    constrained_12   = load_json(os.path.join(SAVE_DIR, "constrained_sac_seeds12.json"))

    if constrained_full:
        summary["DistFlow_Constrained-SAC"] = {
            "env":     "DistFlow (42-dim)",
            "safety":  "HARD (Lagrangian λ)",
            **constrained_full,
        }
    elif constrained_12:
        # Load seed 0 from existing checkpoints to combine
        import numpy as np
        import torch
        from src.qe_sac.constrained_sac import QESACAgentConstrained
        from src.qe_sac.env_utils import VVCEnv13Bus
        from src.qe_sac.metrics import evaluate_policy

        seed0_ckpt = os.path.join(SAVE_DIR, "qe_sac_constrained_seed0.pt")
        if os.path.exists(seed0_ckpt):
            env = VVCEnv13Bus(seed=42)
            obs_dim   = env.observation_space.shape[0]
            n_actions = int(env.action_space.nvec.prod())
            agent = QESACAgentConstrained(obs_dim=obs_dim, n_actions=n_actions)
            agent.load(seed0_ckpt)
            r0 = evaluate_policy(env, agent, n_episodes=20)

            rewards = [r0["mean_reward"]] + [constrained_12[f"seed{s}"]["mean_reward"] for s in [1,2]]
            summary["DistFlow_Constrained-SAC"] = {
                "env":        "DistFlow (42-dim)",
                "safety":     "HARD (Lagrangian λ)",
                "seeds":       3,
                "mean_reward": float(np.mean(rewards)),
                "std_reward":  float(np.std(rewards)),
                "seed_rewards": rewards,
            }

    # ─── OpenDSS results ──────────────────────────────────────────────────
    opendss = load_json(os.path.join(SAVE_DIR, "opendss_results_13bus.json"))
    if opendss:
        for name, r in opendss.items():
            summary[f"OpenDSS_{name.replace(' ', '-')}"] = {
                "env":        "OpenDSS (93-dim, real 3-phase AC)",
                **r,
            }

    # ─── Paper reference ─────────────────────────────────────────────────
    summary["Paper_QE-SAC"] = {
        "env":         "OpenDSS (3219-dim, full 3-phase AC)",
        "mean_reward": -5.39,
        "n_params":    4872,
        "safety":      "soft (penalty only)",
        "source":      "Lin et al. (2025) DOI:10.1109/OAJPE.2025.3534946",
    }
    summary["Paper_Classical-SAC"] = {
        "env":         "OpenDSS (3219-dim, full 3-phase AC)",
        "mean_reward": -5.41,
        "n_params":    899729,
        "safety":      "none",
        "source":      "Lin et al. (2025)",
    }

    # ─── Save ─────────────────────────────────────────────────────────────
    out = os.path.join(SAVE_DIR, "full_results_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved → {out}")

    # ─── Print table ──────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("  FULL RESULTS SUMMARY")
    print("="*80)

    print(f"\n{'Method':35s} {'Reward':10s} {'Std':8s} {'Params':10s} {'Safety':20s}")
    print("-"*80)
    for name, r in summary.items():
        if "mean_reward" in r:
            std_str = f"±{r.get('std_reward', 0):.2f}" if r.get('std_reward') else ""
            params  = r.get('n_params', "—")
            safety  = r.get('safety', "—")
            print(f"  {name:33s} {r['mean_reward']:10.2f} {std_str:8s} {str(params):10s} {safety}")

    if "H1" in summary:
        h1 = summary["H1"]
        print(f"\n  H1: variance_ratio = {h1['variance_ratio']:.2f}× (QE-SAC more stable)")
    if "H2" in summary:
        h2 = summary["H2"]
        print(f"  H2: frozen CAE reward_drop = {h2['reward_drop']:.2f}, std_change = {h2['std_change']:.2f}")

    print("="*80)


if __name__ == "__main__":
    main()
