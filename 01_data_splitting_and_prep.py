"""01_data_splitting_and_prep.py — clients, train/test partition, IID audit, cohort table, fig_01.

Unit of analysis: client = region (5 clients). Partition: calendar blocks 0-2 train,
blocks 3-4 local test (temporal hold-out; no shuffling). Outputs:
  results/base_clients.parquet   raw covariates + real null target with client/split labels
  results/design.json            split description
  tables/table_01_cohort_summary.{csv,tex}
  figures/fig_01_data_distribution_or_manifold.{pdf,png}
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

U = __import__("importlib").import_module("00_config_and_utils")
F = U.load_module("fl_core")


def load_clients() -> pd.DataFrame:
    df = pd.read_csv(U.DATA_PATH)
    onehot = {r: f"region_{r.lower()}" for r in U.REGIONS if f"region_{r.lower()}" in df.columns}
    region = np.full(len(df), "West", dtype=object)          # West is the dropped reference level
    for r, col in onehot.items():
        region[df[col].to_numpy() == 1] = r
    out = df[["row_id", "calendar_block"] + U.FEATURES + [U.NULL_TARGET]].copy()
    out.insert(1, "client", region)
    out["split"] = np.where(out.calendar_block.isin(U.TRAIN_BLOCKS), "train", "test")
    return out


def base_arrays(base: pd.DataFrame) -> dict:
    return {r: (base.loc[(base.client == r) & (base.split == "train"), U.FEATURES].to_numpy(float),
                base.loc[(base.client == r) & (base.split == "test"), U.FEATURES].to_numpy(float))
            for r in U.REGIONS}


def cohort_table(base: pd.DataFrame) -> None:
    """Rows = descriptors, columns = clients + all + the test of between-client homogeneity.
    Counts: chi-square uniformity; means: Kruskal-Wallis; SD: Levene (median); rate: chi-square."""
    tr = base[base.split == "train"]
    rows = []

    def add(label, fn, test_name, p):
        r = {"Descriptor": label}
        for reg in U.REGIONS:
            r[reg] = fn(base[base.client == reg])
        r["All"] = fn(base)
        r["Test"] = test_name
        r["p_value"] = float(p)
        rows.append(r)

    for split in ("train", "test"):
        cnt = base[base.split == split].client.value_counts().reindex(U.REGIONS).to_numpy()
        add(f"n ({split})", lambda d, s=split: float((d.split == s).sum()), "chi2 uniformity",
            stats.chisquare(cnt).pvalue)
    groups = lambda f: [base.loc[base.client == r, f] for r in U.REGIONS]
    add("Resp. admissions, mean", lambda d: d[U.NULL_TARGET].mean(), "Kruskal-Wallis",
        stats.kruskal(*groups(U.NULL_TARGET)).pvalue)
    add("Resp. admissions, SD", lambda d: d[U.NULL_TARGET].std(), "Levene",
        stats.levene(*groups(U.NULL_TARGET), center="median").pvalue)
    for f, lab in [("pm2_5", "PM2.5, mean"), ("temperature", "Temperature, mean"),
                   ("mobility_index", "Mobility index, mean")]:
        add(lab, lambda d, f=f: d[f].mean(), "Kruskal-Wallis", stats.kruskal(*groups(f)).pvalue)
    ct = pd.crosstab(base.client, base.lockdown_status)
    add("Lockdown rate", lambda d: d.lockdown_status.mean(), "chi2 independence", stats.chi2_contingency(ct)[1])
    tab = pd.DataFrame(rows)
    assert not tab.isna().any().any()

    tex = tab.copy()
    for c in U.REGIONS + ["All"]:
        tex[c] = [f"{int(v)}" if lab.startswith("n (") else f"{v:.3f}" for v, lab in zip(tab[c], tab.Descriptor)]
    tex["p_value"] = tab.p_value.map(lambda v: f"{v:.3f}")
    tex = tex.rename(columns={"p_value": "$p$"})
    tex["Descriptor"] = tex.Descriptor.str.replace("PM2.5", "PM$_{2.5}$", regex=False)
    tex["Test"] = tex.Test.str.replace("chi2", r"$\chi^2$", regex=False)
    U.save_table(tab, "table_01_cohort_summary", df_tex=tex, column_format="lrrrrrrlr")
    # full IID audit over all covariates (Holm across the 11 tests)
    tests = {f: stats.kruskal(*groups(f)).pvalue for f in U.FEATURES + [U.NULL_TARGET]}
    audit = pd.DataFrame({"kruskal_p": tests})
    audit["p_holm"] = U.holm(audit.kruskal_p.to_numpy())
    audit.to_csv(U.RESULTS_DIR / "iid_audit_kruskal.csv")


def heterogeneity_magnitudes(base: pd.DataFrame) -> dict:
    """What h = 1 implies in interpretable units (review R1.4), over all replicates."""
    arr = base_arrays(base)
    ratio, flips, shift_sd, slope_sd = [], [], [], []
    k = U.FEATURES.index(U.FOCAL_EXPOSURE)
    for r in range(U.N_REPLICATES):
        pr = F.make_scenario(arr, "prior", 1.0, r)
        b0 = np.array([c.b0_true for c in pr.clients])
        ratio.append(float(np.exp(b0.max() - b0.min())))
        co = F.make_scenario(arr, "concept", 1.0, r)
        sl = np.array([c.beta_true[k] for c in co.clients])
        flips.append(float(np.mean(np.sign(sl) != np.sign(U.FOCAL_EFFECT))))
        slope_sd.append(float(np.std(sl, ddof=1)))
        cv = F.make_scenario(arr, "covariate", 1.0, r)
        z = [F.ref_standardise(c.X_tr).mean(axis=0) for c in cv.clients]
        shift_sd.append(float(np.mean([np.linalg.norm(m) for m in z])))
    ref = [np.linalg.norm(F.ref_standardise(a[0]).mean(axis=0)) for a in arr.values()]
    out = {"prior_max_min_baseline_rate_ratio": float(np.mean(ratio)),
           "concept_share_clients_pm25_sign_flipped": float(np.mean(flips)),
           "concept_between_client_sd_pm25_slope": float(np.mean(slope_sd)),
           "covariate_client_mean_shift_norm_sd_units": float(np.mean(shift_sd)),
           "covariate_client_mean_shift_norm_h0_sd_units": float(np.mean(ref))}
    (U.RESULTS_DIR / "heterogeneity_magnitudes.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def figure_01(base: pd.DataFrame) -> None:
    arr = base_arrays(base)
    fig, axes = plt.subplots(2, 2, figsize=(U.DOUBLE_COL, 5.4))
    colors = [U.OKABE_ITO[i] for i in (0, 1, 2, 4, 5)]
    markers = ["o", "s", "^", "D", "v"]
    from sklearn.decomposition import PCA

    # (a)/(b) PCA of training covariates at h = 0 vs covariate shift h = 1 (replicate 0)
    for ax, level, lab in [(axes[0, 0], 0.0, "a"), (axes[0, 1], 1.0, "b")]:
        sc = F.make_scenario(arr, "covariate", level, 0)
        Z = [F.ref_standardise(c.X_tr) for c in sc.clients]
        pca = PCA(2, random_state=U.SEED).fit(F.ref_standardise(np.vstack([a[0] for a in arr.values()])))
        for j, (c, z) in enumerate(zip(sc.clients, Z)):
            P = pca.transform(z)
            ax.scatter(P[:, 0], P[:, 1], s=3, alpha=0.35, color=colors[j], marker=markers[j], rasterized=True,
                       label=c.name)
            m = P.mean(axis=0)
            ax.scatter(*m, s=60, color=colors[j], marker=markers[j], edgecolor="black", linewidth=0.8, zorder=5)
        ax.set_xlabel("PC1 (reference-standardised covariates)")
        ax.set_ylabel("PC2")
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4.5, 4.5)
        U.add_panel_label(ax, lab)
    axes[0, 1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), markerscale=3, ncol=1, fontsize=8)  # outside axes

    # (c) outcome distribution per client under prior shift (h = 0 vs h = 1)
    ax = axes[1, 0]
    sc0 = F.make_scenario(arr, "prior", 0.0, 0)
    sc1 = F.make_scenario(arr, "prior", 1.0, 0)
    pos = np.arange(5)
    ax.boxplot([c.y_tr for c in sc0.clients], positions=pos - 0.18, widths=0.3, patch_artist=True,
               boxprops=dict(facecolor="white"), medianprops=dict(color="black"), showfliers=False)
    bp = ax.boxplot([c.y_tr for c in sc1.clients], positions=pos + 0.18, widths=0.3, patch_artist=True,
                    medianprops=dict(color="black"), showfliers=False)
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.8)
    ax.set_xticks(pos, U.REGIONS, rotation=20)
    ax.set_ylabel("Simulated admissions per day")
    ax.plot([], [], "s", color="white", markeredgecolor="black", label="$h=0$")
    ax.plot([], [], "s", color="#999999", label="$h=1$ (prior shift)")
    ax.legend(loc="upper left", fontsize=8)
    U.add_panel_label(ax, "c")

    # (d) true client PM2.5 slope vs concept-shift level (all replicates)
    ax = axes[1, 1]
    for j, reg in enumerate(U.REGIONS):
        vals = []
        for h in U.SHIFT_LEVELS:
            sl = [F.make_scenario(arr, "concept", h, r).clients[j].beta_true[U.FEATURES.index(U.FOCAL_EXPOSURE)]
                  for r in range(10)]
            vals.append(sl)
        vals = np.array(vals)
        ax.plot(U.SHIFT_LEVELS, vals, color=colors[j], alpha=0.25, lw=0.7)
    ax.axhline(U.FOCAL_EFFECT, color="black", ls="--", lw=1, label="Global effect")
    ax.axhline(0, color="#999999", lw=0.6)
    ax.set_xlabel("Heterogeneity level $h$")
    ax.set_ylabel("True client PM$_{2.5}$ log-RR per SD")
    ax.legend(loc="lower left", fontsize=8)
    U.add_panel_label(ax, "d")
    fig.tight_layout()
    U.save_figure(fig, "fig_01_data_distribution_or_manifold")


def main() -> None:
    base = load_clients()
    assert set(base.client) == set(U.REGIONS)
    base.to_parquet(U.RESULTS_DIR / "base_clients.parquet", index=False)
    design = {"clients": U.REGIONS, "train_blocks": list(U.TRAIN_BLOCKS), "test_blocks": list(U.TEST_BLOCKS),
              "features": U.FEATURES, "n_train_per_client": base[base.split == "train"].client.value_counts().to_dict(),
              "n_test_per_client": base[base.split == "test"].client.value_counts().to_dict(),
              "scaler": "fit on pooled training covariates of each scenario (federated moments); never on test"}
    (U.RESULTS_DIR / "design.json").write_text(json.dumps(design, indent=2), encoding="utf-8")
    cohort_table(base)
    mags = heterogeneity_magnitudes(base)
    figure_01(base)
    print("heterogeneity at h = 1:", mags)
    print("01 done:", design["n_train_per_client"], design["n_test_per_client"])


if __name__ == "__main__":
    main()
