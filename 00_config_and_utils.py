from __future__ import annotations

import os
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ----------------------------------------------------------------------------
# Reproducibilidad
# ----------------------------------------------------------------------------
SEED = 42
N_BOOTSTRAP = 2000

# ----------------------------------------------------------------------------
# Experimental design constants (fixed a priori, see manifest)
# ----------------------------------------------------------------------------
N_REPLICATES = 50                      # simulation replicates per (shift type, level)
SHIFT_TYPES = ["covariate", "prior", "concept"]
SHIFT_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
TRAIN_BLOCKS = (0, 1, 2)               # calendar blocks used for training
TEST_BLOCKS = (3, 4)                   # held-out calendar blocks (local test per client)
REGIONS = ["Central", "East", "North", "South", "West"]

# Covariates and the exact generator moments recovered in 01_bitacora (Sec. 2.5).
# The DGP standardises with these population moments, so it never uses sample data.
FEATURES = ["pm2_5", "pm10", "no2", "o3", "temperature", "humidity",
            "wind_speed", "mobility_index", "lockdown_status", "mask_usage_rate"]
GEN_MOMENTS = {
    "pm2_5": (60.0, 15.0), "pm10": (80.0, 20.0), "no2": (40.0, 10.0), "o3": (30.0, 8.0),
    "temperature": (25.0, 5.0), "humidity": (55.0, 70.0 / 12 ** 0.5),
    "wind_speed": (7.5, 15.0 / 12 ** 0.5), "mobility_index": (80.0, 10.0),
    "lockdown_status": (0.15, (0.15 * 0.85) ** 0.5), "mask_usage_rate": (75.0, 50.0 / 12 ** 0.5),
}
FOCAL_EXPOSURE = "pm2_5"
NULL_TARGET = "respiratory_admissions"  # real (null) target, true generator Poisson(10)
NULL_TRUE_LAMBDA = 10.0

# DGP magnitudes (log scale)
BASE_LOG_RATE = float(__import__("math").log(10.0))
FOCAL_EFFECT = 0.08                    # global log-RR per SD of PM2.5
OTHER_EFFECT_SD = 0.05                 # other slopes ~ N(0, 0.05^2)
PRIOR_SHIFT_SCALE = 0.40               # client intercept offsets at h = 1
CONCEPT_SHIFT_SD = 0.15                # client slope deviations at h = 1
COVARIATE_TILT = 1.0                   # exponential tilt strength at h = 1

# Federated optimisation defaults (fixed a priori; varied only in 03)
FL_ROUNDS = 100
LOCAL_STEPS = 5
LR = 0.05
L2 = 1e-4
FEDPROX_MU = 1.0
DITTO_LAMBDA = 1.0
EQUIV_MARGIN = 0.01                    # H1a equivalence margin (excess deviance per obs)

# Symmetric tuning (review R2.1): same 5-point budget for every method's hyper-parameter,
# selected on an inner chronological validation split of each client's training rows.
VAL_FRACTION = 0.25
L2_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
MU_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
LAMBDA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]

METHODS = ["Centralized", "Centralized-FE", "Local", "FedAvg", "FedProx", "SCAFFOLD", "FedRep", "Ditto"]
METHOD_STYLE = {  # colour (Okabe-Ito), marker, linestyle
    "Centralized": ("#000000", "o", "-"), "Centralized-FE": ("#D55E00", "*", (0, (3, 1, 1, 1))),
    "Local": ("#999999", "x", ":"),
    "FedAvg": ("#E69F00", "s", "-"), "FedProx": ("#56B4E9", "^", "--"),
    "SCAFFOLD": ("#009E73", "v", "-."), "FedRep": ("#CC79A7", "D", "--"),
    "Ditto": ("#0072B2", "P", "-"),
}
SHIFT_LABEL = {"covariate": "Covariate shift", "prior": "Prior (baseline) shift",
               "concept": "Concept shift"}


def set_global_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


set_global_seed()

# ----------------------------------------------------------------------------
# Rutas
# ----------------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPTS_DIR.parent
DATA_PATH = PROJECT_DIR / "dataset_ml_ready.csv"
DICT_PATH = PROJECT_DIR / "diccionario_datos.csv"
RESULTS_DIR = SCRIPTS_DIR / "results"
TABLES_DIR = SCRIPTS_DIR / "tables"
FIGURES_DIR = SCRIPTS_DIR / "figures"
for _d in (RESULTS_DIR, TABLES_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Estilo gráfico Q1 (sin títulos)
# ----------------------------------------------------------------------------
OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
             "#0072B2", "#D55E00", "#CC79A7", "#000000"]
SINGLE_COL = 3.5   # pulgadas
DOUBLE_COL = 7.0


def set_publication_style(family: str = "serif") -> None:
    sns.set_theme(context="paper", style="ticks", palette=OKABE_ITO)
    plt.rcParams.update({
        "font.family": family,
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "stix" if family == "serif" else "dejavusans",
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "figure.dpi": 150,
    })


set_publication_style()


def add_panel_label(ax, label: str, x: float = -0.12, y: float = 1.05) -> None:
    """Etiqueta de subfigura (a), (b)… en la esquina superior izquierda."""
    ax.text(x, y, f"({label})", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="left")


def _strip_titles(fig) -> None:
    if fig._suptitle is not None:
        fig._suptitle.set_text("")
    for ax in fig.get_axes():
        ax.set_title("")
        ax.set_title("", loc="left")
        ax.set_title("", loc="right")


def save_figure(fig, name: str, formats=("pdf", "png"), dpi: int = 600) -> list[Path]:
    """Guarda en vectorial + PNG 600 DPI, eliminando cualquier título residual."""
    _strip_titles(fig)
    for ax in fig.get_axes():
        sns.despine(ax=ax)
    paths = []
    for fmt in formats:
        p = FIGURES_DIR / f"{name}.{fmt}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


# ----------------------------------------------------------------------------
# Tablas
# ----------------------------------------------------------------------------
def _latex_escape(s) -> str:
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("#", r"\#"), ("_", r"\_"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def save_table(df: pd.DataFrame, name: str, df_tex: pd.DataFrame | None = None,
               index: bool = False, column_format: str | None = None) -> None:
    """CSV con valores completos + .tex booktabs (df_tex permite una versión formateada)."""
    df.to_csv(TABLES_DIR / f"{name}.csv", index=index)
    tex_df = (df_tex if df_tex is not None else df).copy()
    tex_df.columns = [c if ("$" in str(c) or "\\" in str(c)) else _latex_escape(c) for c in tex_df.columns]
    for c in tex_df.select_dtypes(include="object").columns:
        tex_df[c] = tex_df[c].map(lambda v: v if ("\\" in str(v) or "$" in str(v)) else _latex_escape(v))
    tex = tex_df.to_latex(index=index, escape=False, column_format=column_format,
                          float_format="%.3f")
    (TABLES_DIR / f"{name}.tex").write_text(tex, encoding="utf-8")


def fmt_mean_sd(values, decimals: int = 3) -> str:
    v = np.asarray(values, dtype=float)
    return f"{v.mean():.{decimals}f} $\\pm$ {v.std(ddof=1):.{decimals}f}"


# ----------------------------------------------------------------------------
# Estadística
# ----------------------------------------------------------------------------
def bootstrap_ci(values, stat=np.mean, n_boot: int = N_BOOTSTRAP,
                 alpha: float = 0.05, seed: int = SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boots = stat(v[idx], axis=1)
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def rank_biserial_wilcoxon(x, y) -> float:
    """Tamaño del efecto r de rango-biserial para Wilcoxon pareado."""
    from scipy.stats import rankdata

    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    r = rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def cliffs_delta(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    gt = (x[:, None] > y[None, :]).sum()
    lt = (x[:, None] < y[None, :]).sum()
    return float((gt - lt) / (x.size * y.size))


def holm(pvals) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def friedman_nemenyi(scores: pd.DataFrame, alpha: float = 0.05):
    """scores: rows = blocks, cols = methods (lower is better).

    Returns (chi2, p, avg_ranks, critical_difference, pairwise_p DataFrame).
    Nemenyi p-values use the studentized range distribution (no scikit-posthocs).
    """
    from scipy.stats import friedmanchisquare, rankdata, studentized_range

    X = scores.to_numpy(float)
    n, k = X.shape
    chi2, p = friedmanchisquare(*X.T)
    ranks = np.apply_along_axis(rankdata, 1, X)
    avg = pd.Series(ranks.mean(axis=0), index=scores.columns)
    se = np.sqrt(k * (k + 1) / (6.0 * n))
    q_crit = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = q_crit * se
    pw = pd.DataFrame(np.ones((k, k)), index=scores.columns, columns=scores.columns)
    for i in range(k):
        for j in range(i + 1, k):
            q = abs(avg.iloc[i] - avg.iloc[j]) / se * np.sqrt(2)
            pw.iloc[i, j] = pw.iloc[j, i] = float(studentized_range.sf(q, k, np.inf))
    return float(chi2), float(p), avg, float(cd), pw


def load_module(name: str):
    """Import a sibling script whose file name starts with a digit."""
    import importlib
    import sys

    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    return importlib.import_module(name)


def rank_format(means: pd.DataFrame, sds: pd.DataFrame | None = None, lower_is_better: bool = True,
                fmt: str = "{:.3f}", sd_fmt: str = "{:.3f}", tol: float = 1e-9) -> pd.DataFrame:
    r"""Format a (rows = methods, cols = scenarios) table of means (+/- SD): best per column in
    \textbf{}, second-best in \underline{} according to the metric direction (ties share a rank)."""
    out = pd.DataFrame(index=means.index, columns=means.columns, dtype=object)
    for c in means.columns:
        v = means[c].astype(float)
        keys = np.round(v if lower_is_better else -v, 12)
        order = np.sort(np.unique(keys))
        first = order[0]
        second = order[1] if len(order) > 1 else np.nan
        for i in means.index:
            cell = fmt.format(v[i]) + ("" if sds is None else " $\\pm$ " + sd_fmt.format(sds.loc[i, c]))
            if abs(keys[i] - first) < tol:
                cell = "\\textbf{" + cell + "}"
            elif not np.isnan(second) and abs(keys[i] - second) < tol:
                cell = "\\underline{" + cell + "}"
            out.loc[i, c] = cell
    return out


def sci_tex(v: float, digits: int = 2) -> str:
    r"""Signed scientific notation for LaTeX, e.g. -2.4e-05 -> $-2.40\times10^{-5}$."""
    if v == 0 or not np.isfinite(v):
        return "$0$" if v == 0 else "--"
    m, e = f"{v:.{digits}e}".split("e")
    e = int(e)
    return f"${float(m) * 10 ** e:.{digits + 1}f}$" if -2 <= e <= 1 else "$" + m + "\\times10^{" + str(e) + "}$"
