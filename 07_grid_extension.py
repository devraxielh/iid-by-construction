"""07_grid_extension.py — are the conclusions sensitive to the width of the tuning grids? (review R2.7)

Re-runs the tuned analysis for the IID control and the three shift types at h = 1 (same 50
replicates, common random numbers) with extended grids:
  L2      : 1e-4 ... 1e2   (7 values; primary grid stops at 1)
  mu, lam : 1e-3 ... 1e4   (8 values; primary grid 1e-2 ... 1e2)
and compares each method's excess deviance with the primary analysis (paired by replicate).
Outputs: results/grid_extension_scores.csv, results/grid_extension_hparams.csv,
         tables/table_S2_grid_extension.{csv,tex}
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

U = __import__("importlib").import_module("00_config_and_utils")
F = U.load_module("fl_core")
P01 = U.load_module("01_data_splitting_and_prep")

EXT = {"l2": [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
       "mu": [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4],
       "lambda": [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4]}
SCEN = [("control", 0.0), ("covariate", 1.0), ("prior", 1.0), ("concept", 1.0)]
LABEL = {"control": "IID control", "covariate": "Covariate $h=1$", "prior": "Prior $h=1$",
         "concept": "Concept $h=1$"}


def run_one(arr, shift, level, r):
    sc = F.make_scenario(arr, shift, level, r)
    fits, scaler, chosen = F.fit_all(sc, tune=True, grids=EXT)
    ev = pd.DataFrame(F.evaluate(sc, fits, scaler))
    ev = ev.groupby(["shift", "level", "replicate", "method"], as_index=False).excess_deviance.mean()
    hp = [{"shift": shift, "level": level, "replicate": r, "method": m, "hp_name": F.HP_NAME[m], "hp": v,
           "at_upper_bound": v == max(EXT[F.HP_NAME[m]]), "at_lower_bound": v == min(EXT[F.HP_NAME[m]])}
          for m, v in chosen.items()]
    return ev, hp


def main() -> None:
    t0 = time.time()
    arr = P01.base_arrays(pd.read_parquet(U.RESULTS_DIR / "base_clients.parquet"))
    jobs = [("control" if s == "control" else s, h, r) for s, h in SCEN for r in range(U.N_REPLICATES)]
    out = Parallel(n_jobs=-1)(delayed(run_one)(arr, s, h, r) for s, h, r in jobs)
    ext = pd.concat([o[0] for o in out], ignore_index=True)
    ext.to_csv(U.RESULTS_DIR / "grid_extension_scores.csv", index=False)
    hp = pd.DataFrame([h for o in out for h in o[1]])
    hp.to_csv(U.RESULTS_DIR / "grid_extension_hparams.csv", index=False)

    prim = pd.read_csv(U.RESULTS_DIR / "replicate_scores.csv")
    rows = []
    means, sds = pd.DataFrame(index=U.METHODS), pd.DataFrame(index=U.METHODS)
    for s, h in SCEN:
        for m in U.METHODS:
            e = ext[(ext["shift"] == s) & (ext.method == m)].set_index("replicate").excess_deviance
            p = prim[(prim["shift"] == s) & (prim.level == h) & (prim.method == m)].set_index("replicate").excess_deviance
            d = (e - p.loc[e.index]).to_numpy()
            lo, hi = U.bootstrap_ci(d)
            hs = hp[(hp["shift"] == s) & (hp.method == m)]
            rows.append({"scenario": s, "level": h, "method": m, "primary_mean": p.mean(), "extended_mean": e.mean(),
                         "extended_sd": e.std(ddof=1), "delta_mean": d.mean(), "delta_ci95_low": lo,
                         "delta_ci95_high": hi, "relative_change_pct": 100 * d.mean() / p.mean(),
                         "share_upper_bound": hs.at_upper_bound.mean(), "share_lower_bound": hs.at_lower_bound.mean()})
            means.loc[m, LABEL[s]], sds.loc[m, LABEL[s]] = e.mean(), e.std(ddof=1)
    tab = pd.DataFrame(rows)
    tex = U.rank_format(means, sds)
    tex.columns = [c + r" $\downarrow$" for c in tex.columns]
    U.save_table(tab, "table_S2_grid_extension", df_tex=tex.reset_index().rename(columns={"index": "Method"}),
                 column_format="lcccc")
    # does the winner per scenario change?
    for s, _ in SCEN:
        g = tab[tab.scenario == s]
        print(f"{s:10s} best primary: {g.loc[g.primary_mean.idxmin(), 'method']:<15s} "
              f"best extended: {g.loc[g.extended_mean.idxmin(), 'method']}")
    print(tab[["scenario", "method", "primary_mean", "extended_mean", "relative_change_pct",
               "share_upper_bound"]].round(4).to_string())
    print(f"07 done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
