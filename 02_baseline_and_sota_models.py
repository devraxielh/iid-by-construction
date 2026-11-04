"""02_baseline_and_sota_models.py — main simulation grid: 8 methods x (control + 3 shifts x 4 levels) x R.

Primary analysis: every method tuned with the same 5-point budget on an inner chronological
validation split (review R2.1). Supplementary: the a-priori configuration (no tuning).

Primary metric (fixed a priori): client-macro-averaged excess test Poisson deviance per
observation over the oracle (true rate), ED = mean_k [ D(y, mu_hat) - D(y, lambda_true) ].
Outputs:
  results/client_scores.parquet     one row per (scenario, method, client)
  results/replicate_scores.csv      client-macro aggregates per (scenario, method)
  results/null_scores.csv           real null target (respiratory_admissions, true Poisson(10))
  results/test_predictions.parquet  predictions for h in {0, 1} (used in 05)
  results/selected_hparams.csv      hyper-parameter chosen per (scenario, method)
  results/replicate_scores_apriori.csv  same grid with the a-priori configuration
  tables/table_S1_apriori_configuration.{csv,tex}
  tables/table_02_main_results_cv.{csv,tex}
  figures/fig_02_main_performance_comparison.{pdf,png}
"""
from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

U = __import__("importlib").import_module("00_config_and_utils")
F = U.load_module("fl_core")
P01 = U.load_module("01_data_splitting_and_prep")


def run_one(arr, shift, level, r, keep_pred):
    sc = F.make_scenario(arr, shift, level, r)
    fits, scaler, chosen = F.fit_all(sc, tune=True)
    rows = F.evaluate(sc, fits, scaler)
    preds = F.predictions(sc, fits, scaler) if keep_pred else []
    fits0, scaler0, _ = F.fit_all(sc, tune=False)
    rows0 = F.evaluate(sc, fits0, scaler0)
    hp = [{"shift": shift, "level": level, "replicate": r, "method": m, "hp_name": F.HP_NAME[m], "hp": v}
          for m, v in chosen.items()]
    return rows, preds, rows0, hp


def null_scenario(base: pd.DataFrame):
    sc = F.Scenario("null_real", 0.0, -1)
    for reg in U.REGIONS:
        tr = base[(base.client == reg) & (base.split == "train")]
        te = base[(base.client == reg) & (base.split == "test")]
        sc.clients.append(F.ClientData(
            reg, tr[U.FEATURES].to_numpy(float), te[U.FEATURES].to_numpy(float),
            np.full(len(tr), U.NULL_TRUE_LAMBDA), np.full(len(te), U.NULL_TRUE_LAMBDA),
            tr[U.NULL_TARGET].to_numpy(float), te[U.NULL_TARGET].to_numpy(float),
            np.zeros(len(U.FEATURES)), float(np.log(U.NULL_TRUE_LAMBDA))))
    return sc


def aggregate(cs: pd.DataFrame) -> pd.DataFrame:
    g = cs.groupby(["shift", "level", "replicate", "method"])
    return g.agg(excess_deviance=("excess_deviance", "mean"),
                 worst_client_ed=("excess_deviance", "max"),
                 client_sd_ed=("excess_deviance", "std"),
                 rmse_rate=("rmse_rate", "mean"),
                 slope_rmse=("slope_rmse", "mean")).reset_index()


def main() -> None:
    t0 = time.time()
    base = pd.read_parquet(U.RESULTS_DIR / "base_clients.parquet")
    arr = P01.base_arrays(base)
    jobs = [("control", 0.0, r) for r in range(U.N_REPLICATES)]
    jobs += [(s, h, r) for s in U.SHIFT_TYPES for h in U.SHIFT_LEVELS if h > 0 for r in range(U.N_REPLICATES)]
    # make_scenario treats any non-matching shift name as "no shift", so "control" == h = 0
    out = Parallel(n_jobs=-1)(delayed(run_one)(arr, s, h, r, h in (0.0, 1.0)) for s, h, r in jobs)
    preds = pd.DataFrame([p for o in out for p in o[1]],
                         columns=["shift", "level", "replicate", "method", "client", "y", "lambda_true", "mu_hat"])
    preds.to_parquet(U.RESULTS_DIR / "test_predictions.parquet", index=False)
    pd.DataFrame([h for o in out for h in o[3]]).to_csv(U.RESULTS_DIR / "selected_hparams.csv", index=False)

    def expand(rows):
        cs = pd.DataFrame(rows)
        ctrl = cs[cs["shift"] == "control"]   # the IID control is the h = 0 point of every curve (CRN)
        return pd.concat([cs] + [ctrl.assign(shift=s) for s in U.SHIFT_TYPES], ignore_index=True)

    cs = expand([row for o in out for row in o[0]])
    cs.to_parquet(U.RESULTS_DIR / "client_scores.parquet", index=False)
    rep = aggregate(cs)
    rep.to_csv(U.RESULTS_DIR / "replicate_scores.csv", index=False)
    rep0 = aggregate(expand([row for o in out for row in o[2]]))
    rep0.to_csv(U.RESULTS_DIR / "replicate_scores_apriori.csv", index=False)

    sc = null_scenario(base)
    fits, scaler, chosen = F.fit_all(sc, tune=True)
    nul = pd.DataFrame(F.evaluate(sc, fits, scaler))
    nul.to_csv(U.RESULTS_DIR / "null_scores.csv", index=False)
    fits0, scaler0, _ = F.fit_all(sc, tune=False)
    nul0 = pd.DataFrame(F.evaluate(sc, fits0, scaler0))

    make_table(rep, nul, "table_02_main_results_cv")
    make_table(rep0, nul0, "table_S1_apriori_configuration")
    make_figure(rep)
    print(f"02 done: {len(jobs)} scenarios, {len(cs)} client rows, {time.time() - t0:.1f}s")


def make_table(rep: pd.DataFrame, nul: pd.DataFrame, name: str) -> None:
    cols = [("null_real", 0.0, "Real null target$^{a}$"), ("control", 0.0, "IID control ($h=0$)"),
            ("covariate", 1.0, "Covariate ($h=1$)"), ("prior", 1.0, "Prior ($h=1$)"),
            ("concept", 1.0, "Concept ($h=1$)")]
    long = []
    means, sds = pd.DataFrame(index=U.METHODS), pd.DataFrame(index=U.METHODS)
    for shift, h, lab in cols:
        for m in U.METHODS:
            if shift == "null_real":
                v = nul.loc[nul.method == m, "excess_deviance"].to_numpy()       # 5 clients
            else:
                v = rep.loc[(rep["shift"] == shift) & (rep.level == h) & (rep.method == m), "excess_deviance"].to_numpy()
            lo, hi = U.bootstrap_ci(v)
            long.append({"scenario": lab.replace("$^{a}$", ""), "shift": shift, "level": h, "method": m,
                         "n_units": len(v), "unit": "client" if shift == "null_real" else "replicate",
                         "mean": v.mean(), "sd": v.std(ddof=1), "ci95_low": lo, "ci95_high": hi,
                         "median": np.median(v)})
            means.loc[m, lab], sds.loc[m, lab] = v.mean(), v.std(ddof=1)
    tex = U.rank_format(means, sds, lower_is_better=True)
    tex.columns = [c + " $\\downarrow$" for c in tex.columns]
    tex = tex.reset_index().rename(columns={"index": "Method"})
    U.save_table(pd.DataFrame(long), name, df_tex=tex, column_format="lccccc")


def make_figure(rep: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(U.DOUBLE_COL, 5.6))
    for ax, shift, lab in zip(axes.flat[:3], U.SHIFT_TYPES, "abc"):
        d = rep[rep["shift"] == shift]
        for m in U.METHODS:
            col, mk, ls = U.METHOD_STYLE[m]
            mean, lo, hi = [], [], []
            for h in U.SHIFT_LEVELS:
                v = d.loc[(d.level == h) & (d.method == m), "excess_deviance"].to_numpy()
                a, b = U.bootstrap_ci(v)
                mean.append(v.mean()); lo.append(a); hi.append(b)
            ax.plot(U.SHIFT_LEVELS, mean, color=col, marker=mk, ls=ls, ms=4, lw=1.2, label=m)
            ax.fill_between(U.SHIFT_LEVELS, lo, hi, color=col, alpha=0.15, lw=0)
        ax.set_yscale("log")
        ax.set_xlabel(f"{U.SHIFT_LABEL[shift]} level $h$")
        ax.set_ylabel("Excess deviance (per obs.)")
        U.add_panel_label(ax, lab)
    # (d) federated - centralized gap for FedAvg, by shift type (H1)
    ax = axes[1, 1]
    styles = {"covariate": ("#56B4E9", "o", "-"), "prior": ("#E69F00", "s", "--"), "concept": ("#D55E00", "^", "-.")}
    for shift in U.SHIFT_TYPES:
        d = rep[rep["shift"] == shift].pivot_table(index=["level", "replicate"], columns="method",
                                                   values="excess_deviance")
        gap = (d["FedAvg"] - d["Centralized"]).reset_index(name="gap")
        mean, lo, hi = [], [], []
        for h in U.SHIFT_LEVELS:
            v = gap.loc[gap.level == h, "gap"].to_numpy()
            a, b = U.bootstrap_ci(v)
            mean.append(v.mean()); lo.append(a); hi.append(b)
        col, mk, ls = styles[shift]
        ax.plot(U.SHIFT_LEVELS, mean, color=col, marker=mk, ls=ls, ms=4, lw=1.2, label=U.SHIFT_LABEL[shift])
        ax.fill_between(U.SHIFT_LEVELS, lo, hi, color=col, alpha=0.2, lw=0)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel("Heterogeneity level $h$")
    ax.set_ylabel("FedAvg $-$ Centralized (per obs.)")
    ax.legend(loc="upper left", fontsize=8)
    U.add_panel_label(ax, "d")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.06), fontsize=8,
               handlelength=2.2, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    U.save_figure(fig, "fig_02_main_performance_comparison")


if __name__ == "__main__":
    main()
