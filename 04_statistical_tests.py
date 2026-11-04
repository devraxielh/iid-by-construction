"""04_statistical_tests.py — hypothesis tests on results/replicate_scores.csv (alpha = 0.05).

H1a  IID control: FedAvg is equivalent to Centralized (TOST, margin EQUIV_MARGIN, via paired
     bootstrap 90% CI) + Wilcoxon signed-rank (two-sided) with rank-biserial r.
H1b  The FedAvg - Centralized gap increases monotonically with h, per shift type
     (Page's L trend test; blocks = replicates) + Spearman rho(gap, h).
H2   At h = 1, each FL variant vs FedAvg (Wilcoxon).
Holm-Bonferroni is applied within each pre-declared family (H1a, H1b, H2, Friedman).
Omnibus: Friedman + Nemenyi over all unique scenarios (blocks), with critical difference.
McNemar is not applicable (count regression, no paired binary decisions).
Outputs: tables/table_04_statistical_significance.{csv,tex}, results/friedman_ranks.csv,
         figures/fig_06_critical_difference.{pdf,png}
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

U = __import__("importlib").import_module("00_config_and_utils")


def paired(rep, shift, h, a, b):
    d = rep[(rep["shift"] == shift) & (rep.level == h)].pivot(index="replicate", columns="method",
                                                             values="excess_deviance")
    return d[a].to_numpy(), d[b].to_numpy()


def wilcoxon_row(family, test, comp, scen, x, y):
    diff = x - y
    w = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
    lo, hi = U.bootstrap_ci(diff)
    return {"family": family, "test": test, "comparison": comp, "scenario": scen, "statistic": float(w.statistic),
            "p_value": float(w.pvalue), "effect_size": U.rank_biserial_wilcoxon(x, y), "effect_type": "rank-biserial r",
            "mean_diff": diff.mean(), "ci95_low": lo, "ci95_high": hi, "n_blocks": len(diff)}


def main() -> None:
    rep = pd.read_csv(U.RESULTS_DIR / "replicate_scores.csv")
    rows = []

    # ---- H1a: equivalence in the IID control (family of 3)
    tost = []
    oracle_dev = pd.read_parquet(U.RESULTS_DIR / "client_scores.parquet", columns=["shift", "deviance_oracle"])
    oracle_dev = float(oracle_dev.loc[oracle_dev["shift"] == "control", "deviance_oracle"].mean())
    for m in ["FedAvg", "FedProx", "SCAFFOLD"]:
        x, y = paired(rep, "control", 0.0, m, "Centralized")
        row = wilcoxon_row("H1a", "Wilcoxon + TOST", f"{m} vs Centralized", "IID control", x, y)
        lo90, hi90 = U.bootstrap_ci(x - y, alpha=0.10)
        row["equivalent"] = bool(lo90 > -U.EQUIV_MARGIN and hi90 < U.EQUIV_MARGIN)
        tost.append({"comparison": row["comparison"], "mean_diff": row["mean_diff"], "ci90_low": lo90,
                     "ci90_high": hi90, "margin": U.EQUIV_MARGIN, "equivalent": row["equivalent"],
                     "oracle_mean_deviance_per_obs": oracle_dev,
                     "margin_pct_of_oracle_deviance": 100 * U.EQUIV_MARGIN / oracle_dev})
        rows.append(row)
    lx, cy = paired(rep, "control", 0.0, "Local", "Centralized")
    for t in tost:
        t["local_minus_centralized"] = float((lx - cy).mean())
        t["margin_pct_of_local_gap"] = 100 * U.EQUIV_MARGIN / float((lx - cy).mean())
    pd.DataFrame(tost).to_csv(U.RESULTS_DIR / "h1a_tost.csv", index=False)

    # ---- H1b: monotone trend with h (family of 6: federated gap and total loss x 3 shift types)
    for shift in U.SHIFT_TYPES:
        d = rep[rep["shift"] == shift].pivot_table(index=["replicate", "level"], columns="method",
                                                   values="excess_deviance")
        for comp, mat in [("FedAvg - Centralized gap vs h", (d["FedAvg"] - d["Centralized"]).unstack("level")),
                          ("FedAvg excess deviance vs h", d["FedAvg"].unstack("level"))]:
            mat = mat[U.SHIFT_LEVELS]
            pt = stats.page_trend_test(mat.to_numpy())             # increasing in column order
            long = mat.stack()
            rho = stats.spearmanr(long.index.get_level_values("level"), long.to_numpy())
            end = (mat[1.0] - mat[0.0]).to_numpy()
            lo, hi = U.bootstrap_ci(end)
            rows.append({"family": "H1b", "test": "Page trend", "comparison": comp,
                         "scenario": U.SHIFT_LABEL[shift], "statistic": float(pt.statistic),
                         "p_value": float(pt.pvalue), "effect_size": float(rho.statistic),
                         "effect_type": "Spearman rho", "mean_diff": float(end.mean()),
                         "ci95_low": lo, "ci95_high": hi, "n_blocks": len(mat)})

    # ---- H2: variants vs FedAvg at h = 1 (family of 18)
    for shift in U.SHIFT_TYPES:
        for m in ["FedProx", "SCAFFOLD", "FedRep", "Ditto", "Local", "Centralized-FE"]:
            x, y = paired(rep, shift, 1.0, m, "FedAvg")
            rows.append(wilcoxon_row("H2", "Wilcoxon", f"{m} vs FedAvg", f"{U.SHIFT_LABEL[shift]} h=1", x, y))

    # ---- Omnibus Friedman + Nemenyi (family of 4)
    uniq = rep[~((rep.level == 0.0) & (rep["shift"] != "control"))]        # drop duplicated controls
    blocks = uniq.pivot_table(index=["shift", "level", "replicate"], columns="method",
                              values="excess_deviance")[U.METHODS]
    chi2, p, ranks, cd, pw = U.friedman_nemenyi(blocks)
    k = len(U.METHODS)
    rows.append({"family": "Friedman", "test": "Friedman", "comparison": f"{k} methods",
                 "scenario": "all unique scenarios", "statistic": chi2, "p_value": p,
                 "effect_size": chi2 / (len(blocks) * (k - 1)), "effect_type": "Kendall W", "n_blocks": len(blocks)})
    pd.DataFrame({"avg_rank": ranks, "critical_difference": cd}).to_csv(U.RESULTS_DIR / "friedman_ranks.csv")
    pw.to_csv(U.RESULTS_DIR / "nemenyi_pairwise_p.csv")
    per_shift = {}
    for shift in U.SHIFT_TYPES:
        b = rep[(rep["shift"] == shift) & (rep.level == 1.0)].pivot(index="replicate", columns="method",
                                                                  values="excess_deviance")[U.METHODS]
        c2, pp, rk, cdd, _ = U.friedman_nemenyi(b)
        per_shift[shift] = (rk, cdd)
        rows.append({"family": "Friedman", "test": "Friedman", "comparison": f"{k} methods",
                     "scenario": f"{U.SHIFT_LABEL[shift]} h=1", "statistic": c2, "p_value": pp,
                     "effect_size": c2 / (len(b) * (k - 1)), "effect_type": "Kendall W", "n_blocks": len(b)})

    tab = pd.DataFrame(rows)
    tab["p_holm"] = np.nan
    for fam, idx in tab.groupby("family").groups.items():                  # Holm within each family
        tab.loc[idx, "p_holm"] = U.holm(tab.loc[idx, "p_value"].to_numpy())
    tab["significant"] = tab["p_holm"] < 0.05

    def interp(r):
        if r["family"] == "H1a":
            return "equivalent (TOST)" if r["equivalent"] else "not equivalent"
        if r["family"] == "H1b":
            if not r["significant"]:
                return "no trend"
            return "increasing trend" if r["ci95_low"] > 0 else "weak trend (endpoint CI includes 0)"
        if r["family"] == "H2":
            if not r["significant"]:
                return "n.s."
            return "better than FedAvg" if r["mean_diff"] < 0 else "worse than FedAvg"
        return "methods differ" if r["significant"] else "n.s."
    tab["interpretation"] = tab.apply(interp, axis=1)
    tab.to_csv(U.RESULTS_DIR / "tests_detail.csv", index=False)               # incl. mean diffs and CIs
    keep = ["family", "test", "comparison", "scenario", "n_blocks", "statistic", "p_value", "p_holm",
            "effect_size", "effect_type", "interpretation"]
    out = tab[keep]
    assert not out.isna().any().any()
    U.save_table(out, "table_04_statistical_significance", df_tex=tex_view(out), column_format="lllrrrrl")
    cd_figure(ranks, cd, per_shift)
    print(out[["family", "comparison", "scenario", "p_value", "p_holm", "effect_size", "interpretation"]].to_string())


def tex_view(tab: pd.DataFrame) -> pd.DataFrame:
    def sci(v):
        if v < 1e-300:
            return r"$<10^{-300}$"
        m, e = f"{v:.2e}".split("e")
        return "$" + m + r"\times10^{" + str(int(e)) + "}$"
    t = pd.DataFrame({
        "Family": tab.family, "Comparison": tab.comparison, "Scenario": tab.scenario,
        "Statistic": tab.statistic.map(lambda v: f"{v:.1f}"),
        "$p$": tab.p_value.map(sci), r"$p_{\mathrm{Holm}}$": tab.p_holm.map(sci),
        "Effect": [f"{e:+.3f} ({t_.split()[0]})" for e, t_ in zip(tab.effect_size, tab.effect_type)],
        "Interpretation": tab.interpretation})
    return t


def cd_figure(ranks, cd, per_shift):
    """Nemenyi interval plot: average rank +/- CD/2; non-overlapping intervals differ (alpha = 0.05)."""
    fig, axes = plt.subplots(2, 2, figsize=(U.DOUBLE_COL, 4.6), sharex=True)
    panels = [("all", ranks, cd)] + [(s, *per_shift[s]) for s in U.SHIFT_TYPES]
    k = len(U.METHODS)
    for ax, (name, rk, c), lab in zip(axes.flat, panels, "abcd"):
        rk = rk.sort_values(ascending=False)                       # best (lowest rank) on top
        y = np.arange(len(rk))
        best = rk.min()
        for yi, (m, r) in zip(y, rk.items()):
            col, mk, _ = U.METHOD_STYLE[m]
            ax.errorbar(r, yi, xerr=c / 2, fmt=mk, color=col, ms=5, capsize=2.5, lw=1.2)
        ax.axvspan(best - c / 2, best + c / 2, color="#999999", alpha=0.15, lw=0)
        ax.set_yticks(y, rk.index, fontsize=8)
        ax.set_ylim(-1.3, len(rk) - 0.5)                           # room for the annotations
        ax.set_xlim(0.5, k + 0.5)
        ax.set_xticks(range(1, k + 1))
        ax.invert_xaxis()
        ax.text(0.98, 0.03, f"CD = {c:.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
        ax.text(0.02, 0.03, "all scenarios" if name == "all" else f"{U.SHIFT_LABEL[name]}, $h=1$",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=8, style="italic")
        U.add_panel_label(ax, lab, x=-0.30)
    for ax in axes[1]:
        ax.set_xlabel("Average Friedman rank (lower is better)")
    fig.tight_layout()
    U.save_figure(fig, "fig_06_critical_difference")


if __name__ == "__main__":
    main()
