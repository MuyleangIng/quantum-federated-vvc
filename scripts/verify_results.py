"""
Rigorous statistical verification for QE-SAC-FL results.

Statistical methods (following Dietterich 1998, Demšar 2006, Cohen 1988):

  1. Paired one-sided t-test
       H1: aligned_fl > local_only  (per client)
       H2: aligned_fl > naive_fl    (per client)
     Paired because the same seed drives both conditions — removes seed-level
     noise and gives the most powerful test for small n.
     One-sided because hypotheses are directional.

  2. Cohen's d (paired version)
       d = mean(d_i) / std(d_i)
     where d_i = aligned_i - local_i per seed.
     Interpretation: |d| < 0.2 negligible, 0.2-0.5 small,
                     0.5-0.8 medium, > 0.8 large.

  3. Bootstrap 95% CI (B=5000 resamples)
     Percentile bootstrap on the mean difference. More reliable than
     t-interval at small n because it makes no normality assumption.

  4. Bonferroni correction
     3 clients × 2 hypotheses = 6 simultaneous tests.
     α_corrected = 0.05 / 6 = 0.0083.

  5. Wilcoxon signed-rank test (non-parametric fallback)
     Valid when n ≥ 3. Reports W statistic and exact p.

Output: artifacts/qe_sac_fl/verification/
"""

import sys, os, json
import numpy as np
from scipy import stats

sys.path.insert(0, "/root/power-system")
os.makedirs("artifacts/qe_sac_fl/verification", exist_ok=True)

SEEDS      = [s for s in [0, 1, 2, 3, 4]
              if os.path.exists(f"artifacts/qe_sac_fl/seed{s}_aligned_fl.json")]
CONDITIONS = ["local_only", "naive_fl", "aligned_fl"]
CLIENTS    = ["Utility_A_13bus", "Utility_B_34bus", "Utility_C_123bus"]
SHORT      = {"Utility_A_13bus": "A (13bus)",
              "Utility_B_34bus": "B (34bus)",
              "Utility_C_123bus": "C (123bus)"}
COLORS     = {"local_only": "#555555", "naive_fl": "#e67e22", "aligned_fl": "#2980b9"}
CLINES     = {"local_only": ":", "naive_fl": "--", "aligned_fl": "-"}
N_BOOTSTRAP = 5000
ALPHA       = 0.05
N_TESTS     = 6   # 3 clients × 2 hypotheses
ALPHA_BONF  = ALPHA / N_TESTS   # 0.0083


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all():
    data = {c: {cl: [] for cl in CLIENTS} for c in CONDITIONS}
    for seed in SEEDS:
        for cond in CONDITIONS:
            path = f"artifacts/qe_sac_fl/seed{seed}_{cond}.json"
            with open(path) as f:
                raw = json.load(f)
            final = {log["client"]: log["reward"] for log in raw["logs"]}
            # Use the last logged value per client (final reward)
            last = {}
            for log in raw["logs"]:
                last[log["client"]] = log["reward"]
            for cl in CLIENTS:
                data[cond][cl].append(last.get(cl, np.nan))
    return data


def load_curves():
    curves = {c: {cl: {} for cl in CLIENTS} for c in CONDITIONS}
    for seed in SEEDS:
        for cond in CONDITIONS:
            path = f"artifacts/qe_sac_fl/seed{seed}_{cond}.json"
            with open(path) as f:
                raw = json.load(f)
            per_client = {cl: [] for cl in CLIENTS}
            for log in raw["logs"]:
                cl = log["client"]
                if cl in per_client:
                    per_client[cl].append((log["round"], log["reward"]))
            for cl in CLIENTS:
                curves[cond][cl][seed] = sorted(per_client[cl])
    return curves


def load_grad_norms():
    norms = {c: {cl: {} for cl in CLIENTS} for c in CONDITIONS}
    for seed in SEEDS:
        for cond in CONDITIONS:
            path = f"artifacts/qe_sac_fl/seed{seed}_{cond}.json"
            with open(path) as f:
                raw = json.load(f)
            per_client = {cl: [] for cl in CLIENTS}
            for log in raw["logs"]:
                cl = log["client"]
                if cl in per_client:
                    per_client[cl].append((log["round"], log.get("vqc_grad_norm", 0.0)))
            for cl in CLIENTS:
                norms[cond][cl][seed] = sorted(per_client[cl])
    return norms


# ---------------------------------------------------------------------------
# Statistical functions
# ---------------------------------------------------------------------------

def cohen_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Paired Cohen's d = mean(a-b) / std(a-b)."""
    diff = a - b
    sd   = np.std(diff, ddof=1)
    return float(np.mean(diff) / sd) if sd > 0 else 0.0


def bootstrap_ci(a: np.ndarray, b: np.ndarray,
                 B: int = N_BOOTSTRAP, ci: float = 0.95) -> tuple:
    """
    Percentile bootstrap CI for mean(a - b).
    Returns (lower, upper).
    """
    rng   = np.random.default_rng(42)
    diffs = a - b
    n     = len(diffs)
    boot  = np.array([
        np.mean(rng.choice(diffs, size=n, replace=True))
        for _ in range(B)
    ])
    alpha = (1 - ci) / 2
    return float(np.percentile(boot, 100 * alpha)), float(np.percentile(boot, 100 * (1 - alpha)))


def paired_ttest_one_sided(a: np.ndarray, b: np.ndarray) -> tuple:
    """
    One-sided paired t-test: H1: mean(a) > mean(b).
    Returns (t, p_one_sided).
    """
    diff = a - b
    n    = len(diff)
    t    = float(np.mean(diff) / (np.std(diff, ddof=1) / np.sqrt(n)))
    # p for one-sided (upper tail)
    p    = float(stats.t.sf(t, df=n - 1))
    return t, p


def wilcoxon_test(a: np.ndarray, b: np.ndarray) -> tuple:
    """Wilcoxon signed-rank, one-sided (a > b). Returns (W, p)."""
    diff = a - b
    if np.all(diff == 0):
        return 0.0, 1.0
    try:
        res = stats.wilcoxon(diff, alternative="greater", zero_method="wilcox")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return 0.0, 1.0


def sig_stars(p: float, bonf: bool = False) -> str:
    threshold = ALPHA_BONF if bonf else ALPHA
    if p < threshold / 5:
        return "***"
    if p < threshold:
        return "**"
    if p < threshold * 2:
        return "*"
    return "ns"


# ---------------------------------------------------------------------------
# 1. Full statistical table
# ---------------------------------------------------------------------------

def print_stat_table(data):
    n = len(SEEDS)
    print(f"\n{'='*75}")
    print(f"  STATISTICAL VERIFICATION  (n={n} seeds)")
    print(f"  Bonferroni α = {ALPHA}/{N_TESTS} = {ALPHA_BONF:.4f}  |  One-sided paired t-test")
    print(f"{'='*75}")

    print(f"\n  {'Client':<14} {'Δ mean':>8} {'95% CI':>20} {'d':>6} "
          f"{'t':>6} {'p (1-sided)':>12} {'sig':>4}  Hypothesis")
    print("  " + "-"*75)

    all_results = {}
    for cl in CLIENTS:
        a_loc = np.array(data["local_only"][cl])
        a_nai = np.array(data["naive_fl"][cl])
        a_ali = np.array(data["aligned_fl"][cl])
        all_results[cl] = {}

        for label, a, b, hyp in [
            ("ali>loc", a_ali, a_loc, "H1: aligned > local"),
            ("ali>nai", a_ali, a_nai, "H2: aligned > naive"),
            ("nai<loc", a_loc, a_nai, "H0: naive  < local (interference)"),
        ]:
            t, p       = paired_ttest_one_sided(a, b)
            d          = cohen_d_paired(a, b)
            ci_lo, ci_hi = bootstrap_ci(a, b)
            w, pw      = wilcoxon_test(a, b)
            delta      = float(np.mean(a) - np.mean(b))
            stars      = sig_stars(p, bonf=True)

            all_results[cl][label] = {
                "delta": delta, "d": d, "t": t, "p": p,
                "ci": (ci_lo, ci_hi), "wilcoxon_p": pw
            }

            print(f"  {SHORT[cl]:<14} {delta:>+8.4f}  [{ci_lo:>+7.4f}, {ci_hi:>+7.4f}]"
                  f"  {d:>+6.2f}  {t:>+6.2f}  {p:>12.4f}  {stars:>3}  {hyp}")

        print()

    print(f"  *** p<{ALPHA_BONF/5:.4f}  ** p<{ALPHA_BONF:.4f} (Bonferroni)  "
          f"* p<{ALPHA_BONF*2:.4f}  ns not significant")
    return all_results


# ---------------------------------------------------------------------------
# 2. Effect size summary
# ---------------------------------------------------------------------------

def print_effect_summary(data):
    n = len(SEEDS)
    print(f"\n{'='*75}")
    print(f"  EFFECT SIZES — Cohen's d  (|d|>0.8 = large, >0.5 = medium, >0.2 = small)")
    print(f"{'='*75}")
    print(f"\n  {'Client':<14} {'ali vs loc':>12} {'ali vs naive':>14} {'naive vs loc':>14}")
    print("  " + "-"*58)
    for cl in CLIENTS:
        a_loc = np.array(data["local_only"][cl])
        a_nai = np.array(data["naive_fl"][cl])
        a_ali = np.array(data["aligned_fl"][cl])
        d1 = cohen_d_paired(a_ali, a_loc)
        d2 = cohen_d_paired(a_ali, a_nai)
        d3 = cohen_d_paired(a_loc, a_nai)
        def interp(d):
            a = abs(d)
            s = "↑" if d > 0 else "↓"
            if a > 0.8: size = "LARGE"
            elif a > 0.5: size = "medium"
            elif a > 0.2: size = "small"
            else: size = "neg."
            return f"{s}{a:.2f} ({size})"
        print(f"  {SHORT[cl]:<14} {interp(d1):>18} {interp(d2):>18} {interp(d3):>18}")


# ---------------------------------------------------------------------------
# 3. Per-seed breakdown
# ---------------------------------------------------------------------------

def print_per_seed(data):
    print(f"\n{'='*75}")
    print(f"  PER-SEED BREAKDOWN (normalised units)")
    print(f"{'='*75}")
    for cl in CLIENTS:
        print(f"\n  {SHORT[cl]}")
        print(f"  {'Seed':<6} {'local':>8} {'naive':>8} {'aligned':>10} "
              f"{'Δ ali-loc':>10} {'Δ ali-nai':>10}")
        print("  " + "-"*55)
        for i, seed in enumerate(SEEDS):
            loc = data["local_only"][cl][i]
            nai = data["naive_fl"][cl][i]
            ali = data["aligned_fl"][cl][i]
            print(f"  {seed:<6} {loc:>+8.3f} {nai:>+8.3f} {ali:>+10.3f} "
                  f"{ali-loc:>+10.3f} {ali-nai:>+10.3f}")
        # Summary row
        loc_a = np.array(data["local_only"][cl])
        nai_a = np.array(data["naive_fl"][cl])
        ali_a = np.array(data["aligned_fl"][cl])
        print(f"  {'mean':<6} {np.mean(loc_a):>+8.3f} {np.mean(nai_a):>+8.3f} "
              f"{np.mean(ali_a):>+10.3f} "
              f"{np.mean(ali_a-loc_a):>+10.3f} {np.mean(ali_a-nai_a):>+10.3f}")
        print(f"  {'std':<6} {np.std(loc_a):>8.3f} {np.std(nai_a):>8.3f} "
              f"{np.std(ali_a):>10.3f} "
              f"{np.std(ali_a-loc_a):>10.3f} {np.std(ali_a-nai_a):>10.3f}")


# ---------------------------------------------------------------------------
# 4. Learning curves
# ---------------------------------------------------------------------------

def plot_curves(curves):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)

    for ax, cl in zip(axes, CLIENTS):
        for cond in CONDITIONS:
            seed_curves = curves[cond][cl]
            if not seed_curves:
                continue
            all_rounds = sorted(set(r for s in seed_curves.values() for r, _ in s))
            means, stds = [], []
            for rnd in all_rounds:
                vals = [v for s in seed_curves.values() for r, v in s if r == rnd]
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            rounds = np.array(all_rounds)
            means  = np.array(means)
            stds   = np.array(stds)
            label  = {"local_only": "Local only", "naive_fl": "Naive FL",
                      "aligned_fl": "Aligned FL (ours)"}[cond]
            ax.plot(rounds, means, label=label, color=COLORS[cond],
                    ls=CLINES[cond], linewidth=2.0)
            ax.fill_between(rounds, means - stds, means + stds,
                            alpha=0.15, color=COLORS[cond])

        ax.set_title(SHORT[cl], fontsize=12, fontweight="bold")
        ax.set_xlabel("FL Round", fontsize=10)
        ax.set_ylabel("Normalised Reward", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Learning Curves — QE-SAC-FL  ({len(SEEDS)} seeds ± 1 std)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    out = "artifacts/qe_sac_fl/verification/learning_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Learning curves        → {out}")
    plt.close()


# ---------------------------------------------------------------------------
# 5. VQC gradient norm (barren plateau)
# ---------------------------------------------------------------------------

def plot_grad_norms(norms):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, cl in zip(axes, CLIENTS):
        for cond in ["naive_fl", "aligned_fl"]:
            seed_data = norms[cond][cl]
            if not seed_data:
                continue
            all_rounds = sorted(set(r for s in seed_data.values() for r, _ in s))
            means = []
            for rnd in all_rounds:
                vals = [v for s in seed_data.values()
                        for r, v in s if r == rnd and v > 1e-10]
                means.append(np.mean(vals) if vals else np.nan)
            rounds = np.array(all_rounds)
            means  = np.array(means)
            valid  = ~np.isnan(means)
            if valid.any():
                label = "Naive FL" if cond == "naive_fl" else "Aligned FL (ours)"
                ax.plot(rounds[valid], means[valid], label=label,
                        color=COLORS[cond], ls=CLINES[cond], linewidth=2.0)

        ax.set_title(SHORT[cl], fontsize=12, fontweight="bold")
        ax.set_xlabel("FL Round", fontsize=10)
        ax.set_ylabel("VQC ‖∇w‖", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    plt.suptitle("VQC Gradient Norm: Naive FL vs Aligned FL\n"
                 "(collapse → barren plateau from latent space incompatibility)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out = "artifacts/qe_sac_fl/verification/vqc_grad_norms.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  VQC grad norms         → {out}")
    plt.close()


# ---------------------------------------------------------------------------
# 6. Delta distribution plot (key result visual)
# ---------------------------------------------------------------------------

def plot_delta_distributions(data):
    """Box plot of per-seed Δ(aligned - local) for each client."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    positions = [1, 2, 3]
    deltas_ali_loc = []
    deltas_ali_nai = []

    for cl in CLIENTS:
        a_loc = np.array(data["local_only"][cl])
        a_nai = np.array(data["naive_fl"][cl])
        a_ali = np.array(data["aligned_fl"][cl])
        deltas_ali_loc.append(a_ali - a_loc)
        deltas_ali_nai.append(a_ali - a_nai)

    bp1 = ax.boxplot(deltas_ali_loc, positions=[p - 0.2 for p in positions],
                     widths=0.3, patch_artist=True,
                     boxprops=dict(facecolor="#2980b9", alpha=0.7),
                     medianprops=dict(color="white", linewidth=2))
    bp2 = ax.boxplot(deltas_ali_nai, positions=[p + 0.2 for p in positions],
                     widths=0.3, patch_artist=True,
                     boxprops=dict(facecolor="#e74c3c", alpha=0.7),
                     medianprops=dict(color="white", linewidth=2))

    ax.axhline(0, color="black", lw=1.2, ls="--", alpha=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels([SHORT[cl] for cl in CLIENTS], fontsize=11)
    ax.set_ylabel("Δ Normalised Reward (aligned − baseline)", fontsize=10)
    ax.set_title("Aligned FL improvement over baselines\n"
                 "(above 0 = aligned FL wins)", fontsize=12)
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]],
              ["Aligned vs Local", "Aligned vs Naive FL"], fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = "artifacts/qe_sac_fl/verification/delta_distributions.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Delta distributions    → {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*75}")
    print(f"  QE-SAC-FL RIGOROUS VERIFICATION  (seeds: {SEEDS})")
    print(f"{'='*75}")

    print("\n  Loading results...")
    data   = load_all()
    curves = load_curves()
    norms  = load_grad_norms()

    stat_results = print_stat_table(data)
    print_effect_summary(data)
    print_per_seed(data)

    print(f"\n{'='*75}")
    print("  Generating plots...")
    plot_curves(curves)
    plot_grad_norms(norms)
    plot_delta_distributions(data)

    # Save full summary
    summary = {}
    for cond in CONDITIONS:
        summary[cond] = {}
        for cl in CLIENTS:
            v = np.array(data[cond][cl])
            summary[cond][cl] = {
                "mean": float(np.mean(v)), "std": float(np.std(v)),
                "seeds": [float(x) for x in v],
            }
    summary["_meta"] = {
        "n_seeds": len(SEEDS), "seeds": SEEDS,
        "alpha": ALPHA, "alpha_bonferroni": ALPHA_BONF,
        "n_bootstrap": N_BOOTSTRAP,
    }
    out = "artifacts/qe_sac_fl/verification/summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary JSON           → {out}")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()
