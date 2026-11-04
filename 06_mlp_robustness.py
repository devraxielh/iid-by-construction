"""06_mlp_robustness.py — robustness of the main conclusions for a non-convex model (review R1.2).

One-hidden-layer MLP (10 -> 16 tanh -> 1, Poisson log link), full-batch GD, identical fixed
hyper-parameters and identical initialisation for every training regime (no regime is tuned):
  Centralized     pooled GD, all parameters shared
  Centralized-FE  pooled GD, client-specific output bias
  Local           per-client GD, nothing shared
  FedAvg          T rounds x E local steps, all parameters shared
  SCAFFOLD        FedAvg + control variates (option II)
  FedRep          shared body (W1, b1), local head (w2, b2); head then body updated each round
Scenarios: IID control and the three shift types at h = 1, R = MLP_REPLICATES replicates.
Outputs: results/mlp_scores.csv, results/mlp_tests.csv,
         tables/table_05_mlp_robustness.{csv,tex}, figures/fig_07_mlp_robustness.{pdf,png}
"""
from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

U = __import__("importlib").import_module("00_config_and_utils")
F = U.load_module("fl_core")
P01 = U.load_module("01_data_splitting_and_prep")

HIDDEN = 16
ROUNDS, LOCAL_STEPS, LR, L2 = 200, 5, 0.1, 1e-4
MLP_REPLICATES = 30
MLP_METHODS = ["Centralized", "Centralized-FE", "Local", "FedAvg", "SCAFFOLD", "FedRep"]
SCEN = [("control", 0.0), ("covariate", 1.0), ("prior", 1.0), ("concept", 1.0)]
P = len(U.FEATURES)
SIZES = [("W1", (P, HIDDEN)), ("b1", (HIDDEN,)), ("w2", (HIDDEN,)), ("b2", (1,))]
N_PAR = sum(int(np.prod(s)) for _, s in SIZES)


def _slices():
    out, i = {}, 0
    for name, shape in SIZES:
        n = int(np.prod(shape))
        out[name] = (slice(i, i + n), shape)
        i += n
    return out


SL = _slices()


def mask(names):
    m = np.zeros(N_PAR, bool)
    for n in names:
        m[SL[n][0]] = True
    return m


def unpack(th):
    return {n: th[s].reshape(sh) for n, (s, sh) in SL.items()}


def init(seed, log_rate):
    rng = np.random.default_rng(seed)
    th = np.zeros(N_PAR)
    th[SL["W1"][0]] = rng.normal(0, 1 / np.sqrt(P), P * HIDDEN)
    th[SL["w2"][0]] = rng.normal(0, 0.1 / np.sqrt(HIDDEN), HIDDEN)
    th[SL["b2"][0]] = log_rate
    return th


def forward(th, X):
    p = unpack(th)
    Hh = np.tanh(X @ p["W1"] + p["b1"])
    return Hh, Hh @ p["w2"] + p["b2"][0]


def grad(th, X, y, scale):
    """Gradient of the mean Poisson NLL divided by `scale` (pooled mean count) + L2 on weights."""
    p = unpack(th)
    Hh, eta = forward(th, X)
    r = (np.exp(np.clip(eta, -20, 20)) - y) / (len(y) * scale)
    dZ = np.outer(r, p["w2"]) * (1 - Hh ** 2)
    g = np.concatenate([(X.T @ dZ + L2 * p["W1"]).ravel(), dZ.sum(axis=0), Hh.T @ r + L2 * p["w2"], [r.sum()]])
    return g


def train(Xs, ys, scheme, local, th0, scale, alternate=False):
    """scheme: 'central' (exact pooled GD on shared coords) | 'fedavg' | 'scaffold'.
    local: boolean mask of client-specific coordinates. Returns per-client parameter vectors."""
    K = len(Xs)
    pk = np.array([len(y) for y in ys], float)
    pk /= pk.sum()
    th = [th0.copy() for _ in range(K)]
    sh = ~local
    if scheme == "central":
        for _ in range(ROUNDS * LOCAL_STEPS):
            gs = [grad(t, X, y, scale) for t, X, y in zip(th, Xs, ys)]
            gsh = np.sum([p * g for p, g in zip(pk, gs)], axis=0)
            for k in range(K):
                th[k] = th[k] - LR * np.where(local, gs[k], gsh)   # local coords: own gradient
        return th
    glob = th0.copy()
    c, ck = np.zeros(N_PAR), [np.zeros(N_PAR) for _ in range(K)]
    for _ in range(ROUNDS):
        new, dc = [], np.zeros(N_PAR)
        for k in range(K):
            t = np.where(sh, glob, th[k])
            if alternate:                                        # FedRep: head first, then body
                for _ in range(LOCAL_STEPS):
                    t = t - LR * np.where(local, grad(t, Xs[k], ys[k], scale), 0.0)
                for _ in range(LOCAL_STEPS):
                    t = t - LR * np.where(sh, grad(t, Xs[k], ys[k], scale), 0.0)
            else:
                for _ in range(LOCAL_STEPS):
                    g = grad(t, Xs[k], ys[k], scale)
                    if scheme == "scaffold":
                        g = g - ck[k] + c
                    t = t - LR * g
            if scheme == "scaffold":
                c_new = ck[k] - c + (glob - t) / (LOCAL_STEPS * LR)
                dc += pk[k] * (c_new - ck[k])
                ck[k] = c_new
            th[k] = t
            new.append(t)
        glob = np.where(sh, np.sum([p * t for p, t in zip(pk, new)], axis=0), glob)
        c = c + dc
    return [np.where(sh, glob, t) for t in th]


def run_one(arr, shift, level, r):
    sc = F.make_scenario(arr, shift, level, r)
    Xp = np.vstack([c.X_tr for c in sc.clients])
    mean, sd = Xp.mean(axis=0), Xp.std(axis=0)
    Xs = [(c.X_tr - mean) / sd for c in sc.clients]
    ys = [c.y_tr for c in sc.clients]
    scale = float(np.concatenate(ys).mean())
    th0 = init(U.SEED * 100 + r, np.log(scale))
    none, all_ = np.zeros(N_PAR, bool), np.ones(N_PAR, bool)
    fits = {"Centralized": train(Xs, ys, "central", none, th0, scale),
            "Centralized-FE": train(Xs, ys, "central", mask(["b2"]), th0, scale),
            "Local": train(Xs, ys, "central", all_, th0, scale),
            "FedAvg": train(Xs, ys, "fedavg", none, th0, scale),
            "SCAFFOLD": train(Xs, ys, "scaffold", none, th0, scale),
            "FedRep": train(Xs, ys, "fedavg", mask(["w2", "b2"]), th0, scale, alternate=True)}
    rows = []
    for m, ths in fits.items():
        for c, th in zip(sc.clients, ths):
            mu = np.exp(np.clip(forward(th, (c.X_te - mean) / sd)[1], -20, 20))
            ed = F.mean_deviance(c.y_te, mu) - F.mean_deviance(c.y_te, c.lam_te)
            rows.append({"shift": shift, "level": level, "replicate": r, "method": m, "client": c.name,
                         "excess_deviance": ed})
    return rows


def main() -> None:
    t0 = time.time()
    arr = P01.base_arrays(pd.read_parquet(U.RESULTS_DIR / "base_clients.parquet"))
    out = Parallel(n_jobs=-1)(delayed(run_one)(arr, s, h, r) for s, h in SCEN for r in range(MLP_REPLICATES))
    cs = pd.DataFrame([row for rows in out for row in rows])
    rep = cs.groupby(["shift", "level", "replicate", "method"], as_index=False).excess_deviance.mean()
    rep.to_csv(U.RESULTS_DIR / "mlp_scores.csv", index=False)

    labels = {"control": "IID control", "covariate": "Covariate $h=1$", "prior": "Prior $h=1$",
              "concept": "Concept $h=1$"}
    long, means, sds = [], pd.DataFrame(index=MLP_METHODS), pd.DataFrame(index=MLP_METHODS)
    for s, h in SCEN:
        for m in MLP_METHODS:
            v = rep.loc[(rep["shift"] == s) & (rep.method == m), "excess_deviance"].to_numpy()
            lo, hi = U.bootstrap_ci(v)
            long.append({"scenario": s, "level": h, "method": m, "n_replicates": len(v), "mean": v.mean(),
                         "sd": v.std(ddof=1), "ci95_low": lo, "ci95_high": hi})
            means.loc[m, labels[s]], sds.loc[m, labels[s]] = v.mean(), v.std(ddof=1)
    tex = U.rank_format(means, sds)
    tex.columns = [c + r" $\downarrow$" for c in tex.columns]
    U.save_table(pd.DataFrame(long), "table_05_mlp_robustness",
                 df_tex=tex.reset_index().rename(columns={"index": "Method (MLP)"}), column_format="lcccc")

    # Tests (one family, Holm): federated penalty and personalization vs FedAvg in each scenario
    tests = []
    for s, h in SCEN:
        piv = rep[rep["shift"] == s].pivot(index="replicate", columns="method", values="excess_deviance")
        for a, b in [("FedAvg", "Centralized"), ("SCAFFOLD", "Centralized"), ("FedRep", "Centralized-FE"),
                     ("FedRep", "FedAvg"), ("Local", "FedAvg")]:
            d = (piv[a] - piv[b]).to_numpy()
            w = stats.wilcoxon(piv[a], piv[b])
            lo, hi = U.bootstrap_ci(d)
            tests.append({"scenario": s, "comparison": f"{a} vs {b}", "mean_diff": d.mean(), "ci95_low": lo,
                          "ci95_high": hi, "statistic": float(w.statistic), "p_value": float(w.pvalue),
                          "rank_biserial": U.rank_biserial_wilcoxon(piv[a], piv[b]),
                          "relative_to_b_pct": 100 * d.mean() / piv[b].mean()})
    tt = pd.DataFrame(tests)
    tt["p_holm"] = U.holm(tt.p_value.to_numpy())
    tt.to_csv(U.RESULTS_DIR / "mlp_tests.csv", index=False)

    fig, ax = plt.subplots(figsize=(U.DOUBLE_COL, 2.8))
    x = np.arange(len(SCEN))
    w = 0.13
    for j, m in enumerate(MLP_METHODS):
        col, mk, _ = U.METHOD_STYLE[m]
        g = pd.DataFrame(long)
        g = g[g.method == m].set_index("scenario").loc[[s for s, _ in SCEN]]
        ax.errorbar(x + (j - 2.5) * w, g["mean"], yerr=[g["mean"] - g.ci95_low, g.ci95_high - g["mean"]],
                    fmt=mk, color=col, ms=5, capsize=2, lw=1.0, label=m)
    ax.set_yscale("log")
    ax.set_xticks(x, ["IID control", "Covariate shift $h=1$", "Prior shift $h=1$", "Concept shift $h=1$"])
    ax.set_ylabel("MLP excess deviance (per obs.)")
    ax.legend(ncol=6, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.01),   # outside the axes (R3.7)
              handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout()
    U.save_figure(fig, "fig_07_mlp_robustness")
    print(f"06 done in {time.time() - t0:.1f}s")
    print(means.round(4).to_string())
    print(tt[["scenario", "comparison", "mean_diff", "p_holm", "relative_to_b_pct"]].round(4).to_string())


if __name__ == "__main__":
    main()
