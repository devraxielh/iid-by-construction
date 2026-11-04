"""03_ablation_and_sensitivity.py — component ablation (vs FedAvg) and sensitivity analyses.

(1) Ablation (from results/replicate_scores.csv, paired by replicate): effect of adding one
    FL component to FedAvg: proximal term (FedProx), drift correction (SCAFFOLD), local
    intercept head (FedRep), proximal personal models (Ditto), and no federation (Local).
(2) Sensitivity: local steps E (client drift), Ditto lambda (personalisation strength).
(3) Privacy: record-level DP-GD inside FedAvg for epsilon in {1, 3, 8, inf} (RDP accounting).
Outputs: results/sens_local_steps.csv, results/sens_ditto_lambda.csv, results/sens_dp.csv,
         tables/table_03_ablation_study.{csv,tex}, figures/fig_03_ablation_and_sensitivity.{pdf,png}
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

E_GRID = [1, 2, 5, 10, 20]
LAMBDA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
EPS_GRID = [1.0, 3.0, 8.0, np.inf]
DP_ROUNDS, DP_LOCAL_STEPS, DP_CLIP, DP_DELTA = 50, 1, 5.0, 1e-5
SCEN_SENS = [("control", 0.0), ("covariate", 1.0), ("prior", 1.0), ("concept", 1.0)]


def macro(rows) -> pd.DataFrame:
    d = pd.DataFrame(rows)
    return d.groupby(["shift", "level", "replicate", "method"], as_index=False).excess_deviance.mean()


def job_E(arr, shift, h, r, E):
    sc = F.make_scenario(arr, shift, h, r)
    fits, scl, _ = F.fit_all(sc, methods=["Centralized", "FedAvg", "FedProx", "SCAFFOLD"], E=E)
    return macro(F.evaluate(sc, fits, scl)).assign(E=E)


def job_lambda(arr, shift, h, r, lam):
    sc = F.make_scenario(arr, shift, h, r)
    fits, scl, _ = F.fit_all(sc, methods=["Ditto"], ditto_lam=lam)
    return macro(F.evaluate(sc, fits, scl)).assign(lam=lam)


def job_dp(arr, shift, h, r, eps):
    sc = F.make_scenario(arr, shift, h, r)
    Xp = np.vstack([c.X_tr for c in sc.clients])
    mean, sd = Xp.mean(axis=0), Xp.std(axis=0)
    As = [F.design(c.X_tr, mean, sd) for c in sc.clients]
    ys = [c.y_tr for c in sc.clients]
    lr = F.safe_lr(As, [F.newton(A, y) for A, y in zip(As, ys)])
    steps = DP_ROUNDS * DP_LOCAL_STEPS
    if np.isinf(eps):
        dp, sigma = None, 0.0
    else:
        sigma = F.sigma_for_epsilon(eps, steps, DP_DELTA)
        dp = {"clip": DP_CLIP, "sigma": sigma}
    rng = np.random.default_rng(U.SEED * 7 + r)
    w = F.fedavg(As, ys, rounds=DP_ROUNDS, E=DP_LOCAL_STEPS, lr=lr, dp=dp, rng=rng)
    fits = {"FedAvg-DP": [w] * len(As)}
    return macro(F.evaluate(sc, fits, (mean, sd))).assign(eps=eps, sigma=sigma)


def ablation_table(rep: pd.DataFrame) -> pd.DataFrame:
    """Paired effect of one FL component relative to FedAvg (all methods tuned, see 02).
    Best (most negative) per column in bold, second underlined; dagger = 95% CI excludes 0."""
    comps = [("FedProx", r"+ proximal term ($\mu$ tuned)"), ("SCAFFOLD", "+ drift correction (control variates)"),
             ("FedRep", "+ local intercept head"), ("Ditto", r"+ proximal personal models ($\lambda$ tuned)"),
             ("Local", "no federation (local only)"), ("Centralized", "pooled data, global model"),
             ("Centralized-FE", "pooled data, client fixed effects")]
    rows = []
    for m, lab in comps:
        for shift, h in SCEN_SENS:
            d = rep[(rep["shift"] == shift) & (rep.level == h)].pivot(index="replicate", columns="method",
                                                                     values="excess_deviance")
            delta = (d[m] - d["FedAvg"]).to_numpy()
            lo, hi = U.bootstrap_ci(delta)
            rows.append({"variant": m, "description": lab, "shift": shift, "level": h, "delta_mean": delta.mean(),
                         "delta_sd": delta.std(ddof=1), "ci95_low": lo, "ci95_high": hi,
                         "ci_excludes_zero": bool(hi < 0 or lo > 0),
                         "relative_change_pct": 100 * delta.mean() / d["FedAvg"].mean()})
    tab = pd.DataFrame(rows)
    heads = ["IID control", "Covariate $h=1$", "Prior $h=1$", "Concept $h=1$"]
    means = tab.pivot(index="description", columns="shift", values="delta_mean")[[s for s, _ in SCEN_SENS]]
    means = means.loc[[lab for _, lab in comps]]
    ranked = U.rank_format(means, None, lower_is_better=True, fmt="{:.6g}")
    tex = pd.DataFrame(index=means.index, columns=means.columns, dtype=object)
    for _, r in tab.iterrows():
        body = (f"{U.sci_tex(r.delta_mean)} [{U.sci_tex(r.ci95_low)}, {U.sci_tex(r.ci95_high)}]"
                + (r"$^{\dagger}$" if r.ci_excludes_zero else ""))
        wrap = ranked.loc[r.description, r["shift"]]
        tex.loc[r.description, r["shift"]] = (r"\textbf{" + body + "}" if wrap.startswith(r"\textbf")
                                              else r"\underline{" + body + "}" if wrap.startswith(r"\underline")
                                              else body)
    tex.columns = [c + r" $\Delta\downarrow$" for c in heads]
    tex = tex.reset_index().rename(columns={"description": "Variant (vs. FedAvg)"})
    U.save_table(tab, "table_03_ablation_study", df_tex=tex, column_format="lcccc")
    return tab


def ci_line(ax, x, groups, color, marker, ls, label):
    mean, lo, hi = [], [], []
    for v in groups:
        a, b = U.bootstrap_ci(v)
        mean.append(np.mean(v)); lo.append(a); hi.append(b)
    ax.plot(x, mean, color=color, marker=marker, ls=ls, ms=4, lw=1.2, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0)


def figure_03(sE, sL, sD, rep):
    fig, axes = plt.subplots(1, 3, figsize=(U.DOUBLE_COL, 2.6))
    # (a) client drift vs local steps: (method - Centralized), covariate h = 1
    ax = axes[0]
    d = sE[(sE["shift"] == "covariate")]
    piv = d.pivot_table(index=["E", "replicate"], columns="method", values="excess_deviance")
    for m in ["FedAvg", "FedProx", "SCAFFOLD"]:
        col, mk, ls = U.METHOD_STYLE[m]
        ci_line(ax, E_GRID, [(piv.loc[e, m] - piv.loc[e, "Centralized"]).to_numpy() for e in E_GRID], col, mk, ls, m)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xscale("log", base=2)
    ax.set_xticks(E_GRID, [str(e) for e in E_GRID])
    ax.set_xlabel("Local steps per round $E$")
    ax.set_ylabel("Gap to Centralized (per obs.)")
    ax.legend(fontsize=8, loc="upper left")
    U.add_panel_label(ax, "a", x=-0.25)
    # (b) Ditto lambda
    ax = axes[1]
    sty = {"control": ("#000000", "o", "-"), "covariate": ("#56B4E9", "s", "--"),
           "prior": ("#E69F00", "^", "-."), "concept": ("#D55E00", "D", ":")}
    for shift, h in SCEN_SENS:
        d = sL[(sL["shift"] == shift)]
        col, mk, ls = sty[shift]
        ci_line(ax, LAMBDA_GRID, [d.loc[d.lam == l, "excess_deviance"].to_numpy() for l in LAMBDA_GRID],
                col, mk, ls, "IID" if shift == "control" else shift.capitalize())   # h = 1 stated in caption
    ax.set_xscale("log")
    ax.set_xticks(LAMBDA_GRID, ["0.01", "0.1", "1", "10", "100"])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_yscale("log")
    ax.set_xlabel("Ditto coupling $\\lambda$")
    ax.set_ylabel("Excess deviance (per obs.)")
    ax.legend(fontsize=8, loc="upper left", handlelength=1.5, borderaxespad=0.2)   # empty region above small-lambda points
    U.add_panel_label(ax, "b", x=-0.25)
    # (c) DP
    ax = axes[2]
    xs = np.arange(len(EPS_GRID))
    for shift, col, mk, ls in [("control", "#000000", "o", "-"), ("covariate", "#56B4E9", "s", "--")]:
        d = sD[sD["shift"] == shift]
        ci_line(ax, xs, [d.loc[d.eps == e, "excess_deviance"].to_numpy() for e in EPS_GRID], col, mk, ls,
                "IID control" if shift == "control" else "Covariate $h=1$")
    ax.set_xticks(xs, ["1", "3", "8", "$\\infty$"])
    ax.set_yscale("log")
    ax.set_xlabel("Privacy budget $\\varepsilon$ ($\\delta=10^{-5}$)")
    ax.set_ylabel("Excess deviance (DP-FedAvg)")
    ax.legend(fontsize=8, loc="upper right")
    U.add_panel_label(ax, "c", x=-0.25)
    fig.tight_layout()
    U.save_figure(fig, "fig_03_ablation_and_sensitivity")


def main() -> None:
    t0 = time.time()
    arr = P01.base_arrays(pd.read_parquet(U.RESULTS_DIR / "base_clients.parquet"))
    R = range(U.N_REPLICATES)
    par = Parallel(n_jobs=-1)
    sE = pd.concat(par(delayed(job_E)(arr, s, h, r, E) for s, h in [("covariate", 1.0), ("concept", 1.0)]
                       for E in E_GRID for r in R), ignore_index=True)
    sE.to_csv(U.RESULTS_DIR / "sens_local_steps.csv", index=False)
    sL = pd.concat(par(delayed(job_lambda)(arr, s, h, r, l) for s, h in SCEN_SENS for l in LAMBDA_GRID for r in R),
                   ignore_index=True)
    sL.to_csv(U.RESULTS_DIR / "sens_ditto_lambda.csv", index=False)
    sD = pd.concat(par(delayed(job_dp)(arr, s, h, r, e) for s, h in [("control", 0.0), ("covariate", 1.0)]
                       for e in EPS_GRID for r in R), ignore_index=True)
    sD.to_csv(U.RESULTS_DIR / "sens_dp.csv", index=False)
    rep = pd.read_csv(U.RESULTS_DIR / "replicate_scores.csv")
    ablation_table(rep)
    figure_03(sE, sL, sD, rep)
    print(f"03 done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
