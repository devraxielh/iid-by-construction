"""05_explainability_and_error.py — what the models learn, and where they fail.

For a Poisson GLM the explanation *is* the coefficient vector, and here the true client
coefficients are known, so we report coefficient recovery instead of post-hoc attributions:
  fig_04 (a) estimated vs true client PM2.5 log-RR (concept shift, h = 1)
         (b) slope RMSE vs h, concept shift      (c) slope RMSE vs h, prior shift
Error analysis on held-out test blocks:
  fig_05 (a) client-level excess deviance (fairness), concept shift h = 1
         (b) calibration by decile of predicted rate, prior shift h = 1
         (c) per-client mean Pearson residual of FedAvg vs FedRep, prior shift h = 1
Outputs: results/coef_recovery.csv, results/calibration.csv, results/fairness.csv
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

U = __import__("importlib").import_module("00_config_and_utils")
SHOW = ["Centralized", "Centralized-FE", "FedAvg", "SCAFFOLD", "FedRep", "Ditto", "Local"]


def slope_curves(ax, cs, shift, methods):
    d = cs[cs["shift"] == shift].groupby(["level", "replicate", "method"], as_index=False).slope_rmse.mean()
    rows = []
    for m in methods:
        col, mk, ls = U.METHOD_STYLE[m]
        mean, lo, hi = [], [], []
        for h in U.SHIFT_LEVELS:
            v = d.loc[(d.level == h) & (d.method == m), "slope_rmse"].to_numpy()
            a, b = U.bootstrap_ci(v)
            mean.append(v.mean()); lo.append(a); hi.append(b)
            rows.append({"shift": shift, "level": h, "method": m, "slope_rmse_mean": v.mean(), "ci95_low": a,
                         "ci95_high": b})
        ax.plot(U.SHIFT_LEVELS, mean, color=col, marker=mk, ls=ls, ms=4, lw=1.2, label=m)
        ax.fill_between(U.SHIFT_LEVELS, lo, hi, color=col, alpha=0.15, lw=0)
    ax.set_xlabel(f"{shift.capitalize()} shift level $h$")       # short label: avoids clipping (R3.7)
    ax.set_ylabel("Slope RMSE")
    return rows


def figure_04(cs):
    fig, axes = plt.subplots(1, 3, figsize=(U.DOUBLE_COL, 2.6))
    ax = axes[0]
    d = cs[(cs["shift"] == "concept") & (cs.level == 1.0)]
    lim = [d.pm25_slope_true.min() - 0.05, d.pm25_slope_true.max() + 0.05]
    for m in ["FedAvg", "Ditto", "Local"]:
        col, mk, _ = U.METHOD_STYLE[m]
        s = d[d.method == m]
        ax.scatter(s.pm25_slope_true, s.pm25_slope_hat, s=6, alpha=0.5, color=col, marker=mk, label=m, lw=0.3,
                   rasterized=True)
    ax.plot(lim, lim, color="black", lw=0.8, ls="--")
    ax.axhline(U.FOCAL_EFFECT, color="#999999", lw=0.6)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("True client PM$_{2.5}$ log-RR")
    ax.set_ylabel("Estimated PM$_{2.5}$ log-RR")
    ax.legend(fontsize=8, loc="upper left", markerscale=2)
    U.add_panel_label(ax, "a", x=-0.28)
    rows = slope_curves(axes[1], cs, "concept", SHOW)
    axes[1].set_yscale("log")
    U.add_panel_label(axes[1], "b", x=-0.28)
    rows += slope_curves(axes[2], cs, "prior", SHOW)
    U.add_panel_label(axes[2], "c", x=-0.28)
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.6, 1.14), fontsize=8, handlelength=2.2)
    fig.tight_layout(w_pad=2.0)
    U.save_figure(fig, "fig_04_explainability_shap_or_latent")
    rec = pd.DataFrame(rows)
    # PM2.5 slope bias and correlation with truth at concept h = 1
    stats_rows = []
    for m in U.METHODS:
        s = d[d.method == m]
        stats_rows.append({"shift": "concept", "level": 1.0, "method": m, "level_type": "pm25_recovery",
                           "corr_true_hat": float(np.corrcoef(s.pm25_slope_true, s.pm25_slope_hat)[0, 1])
                           if s.pm25_slope_hat.std() > 0 else np.nan,
                           "mean_abs_error": float((s.pm25_slope_hat - s.pm25_slope_true).abs().mean())})
    pd.concat([rec, pd.DataFrame(stats_rows)], ignore_index=True).to_csv(U.RESULTS_DIR / "coef_recovery.csv",
                                                                       index=False)


def figure_05(cs, pr):
    fig, axes = plt.subplots(1, 3, figsize=(U.DOUBLE_COL, 2.7), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    # (a) fairness: client-level excess deviance, concept h = 1
    ax = axes[0]
    d = cs[(cs["shift"] == "concept") & (cs.level == 1.0)]
    data = [d.loc[d.method == m, "excess_deviance"].clip(lower=1e-4).to_numpy() for m in U.METHODS]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=True,
                    flierprops=dict(marker=".", ms=2, alpha=0.4), medianprops=dict(color="black"))
    for patch, m in zip(bp["boxes"], U.METHODS):
        patch.set_facecolor(U.METHOD_STYLE[m][0]); patch.set_alpha(0.7)
    ax.set_yscale("log")
    ax.set_xticks(range(1, len(U.METHODS) + 1), U.METHODS, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Excess deviance (per obs.)")      # shorter label: no overlap with "(a)"
    U.add_panel_label(ax, "a", x=-0.25)
    fair = d.groupby(["replicate", "method"]).excess_deviance.agg(["mean", "max", "std"]).groupby("method").mean()
    fair.to_csv(U.RESULTS_DIR / "fairness.csv")

    # (b) calibration by decile of predicted rate, prior shift h = 1 (all replicates)
    ax = axes[1]
    p = pr[(pr["shift"] == "prior") & (pr.level == 1.0)].copy()
    cal_rows = []
    for m in ["FedAvg", "FedRep", "Ditto", "Local"]:
        s = p[p.method == m].copy()
        s["bin"] = pd.qcut(s.mu_hat, 10, labels=False, duplicates="drop")
        g = s.groupby("bin").agg(pred=("mu_hat", "mean"), obs=("y", "mean"), n=("y", "size"))
        col, mk, ls = U.METHOD_STYLE[m]
        ax.plot(g.pred, g.obs, color=col, marker=mk, ls=ls, ms=4, lw=1.1, label=m)
        cal_rows.append(g.assign(method=m).reset_index())
    allg = pd.concat(cal_rows)
    lim = [min(allg.pred.min(), allg.obs.min()) * 0.9, max(allg.pred.max(), allg.obs.max()) * 1.05]
    ax.plot(lim, lim, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Mean predicted rate (decile)")
    ax.set_ylabel("Mean observed count")
    ax.legend(fontsize=8, loc="upper left")
    U.add_panel_label(ax, "b", x=-0.25)
    pd.concat(cal_rows).to_csv(U.RESULTS_DIR / "calibration.csv", index=False)

    # (c) mean Pearson residual by client baseline rank (offsets are permuted across replicates,
    #     so clients are ordered by their true mean rate within each replicate), prior shift h = 1
    ax = axes[2]
    p["pearson"] = (p.y - p.mu_hat) / np.sqrt(p.mu_hat)
    width = 0.38
    x = np.arange(len(U.REGIONS))
    base_rank = (p[p.method == "FedAvg"].groupby(["replicate", "client"]).lambda_true.mean()
                 .groupby("replicate").rank(method="first").astype(int).rename("rank"))
    for k, m in enumerate(["FedAvg", "FedRep"]):
        g = p[p.method == m].groupby(["replicate", "client"]).pearson.mean().to_frame().join(base_rank).reset_index()
        means = [g.loc[g["rank"] == q + 1, "pearson"].mean() for q in x]
        cis = [U.bootstrap_ci(g.loc[g["rank"] == q + 1, "pearson"].to_numpy()) for q in x]
        err = np.array([[mu - a, b - mu] for mu, (a, b) in zip(means, cis)]).T
        ax.bar(x + (k - 0.5) * width, means, width, yerr=err, color=U.METHOD_STYLE[m][0], alpha=0.85,
               capsize=2, label=m, hatch=None if m == "FedAvg" else "//", edgecolor="black", lw=0.4)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(x, ["1\n(lowest)", "2", "3", "4", "5\n(highest)"], fontsize=8)
    ax.set_xlabel("Client baseline-rate rank")
    ax.set_ylabel("Mean Pearson residual")
    ax.legend(fontsize=8, loc="upper left")
    U.add_panel_label(ax, "c", x=-0.25)
    fig.tight_layout()
    U.save_figure(fig, "fig_05_error_calibration_or_case_study")


def main() -> None:
    cs = pd.read_parquet(U.RESULTS_DIR / "client_scores.parquet")
    pr = pd.read_parquet(U.RESULTS_DIR / "test_predictions.parquet")
    figure_04(cs)
    figure_05(cs, pr)
    print(pd.read_csv(U.RESULTS_DIR / "fairness.csv").round(4).to_string())
    print(pd.read_csv(U.RESULTS_DIR / "coef_recovery.csv").query("level_type == 'pm25_recovery'")
          [["method", "corr_true_hat", "mean_abs_error"]].round(4).to_string())


if __name__ == "__main__":
    main()
