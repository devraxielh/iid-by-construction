"""run_all_experiments.py — clean run of 01 -> 05, exit-criteria checks, and a reproducibility re-run.

Usage:  python scripts/run_all_experiments.py [--log=path]   (full run + second run for reproducibility)
        python scripts/run_all_experiments.py --single   (one run, checks only)
"""
from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG: list[str] = []
STEPS = ["01_data_splitting_and_prep.py", "02_baseline_and_sota_models.py", "03_ablation_and_sensitivity.py",
         "04_statistical_tests.py", "05_explainability_and_error.py", "06_mlp_robustness.py",
         "07_grid_extension.py"]
TABLES = ["table_01_cohort_summary", "table_02_main_results_cv", "table_03_ablation_study",
          "table_04_statistical_significance", "table_05_mlp_robustness", "table_S1_apriori_configuration",
          "table_S2_grid_extension"]
FIGURES = ["fig_01_data_distribution_or_manifold", "fig_02_main_performance_comparison",
           "fig_03_ablation_and_sensitivity", "fig_04_explainability_shap_or_latent",
           "fig_05_error_calibration_or_case_study", "fig_06_critical_difference", "fig_07_mlp_robustness"]


def clean() -> None:
    for d in ("results", "tables", "figures"):
        shutil.rmtree(HERE / d, ignore_errors=True)


def run_pipeline() -> float:
    t0 = time.time()
    for s in STEPS:
        t = time.time()
        r = subprocess.run([sys.executable, str(HERE / s)], cwd=HERE, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-3000:], r.stderr[-3000:])
            raise SystemExit(f"FAILED: {s}")
        print(f"  {s:<36s} {time.time() - t:6.1f}s")
        LOG.append(f"===== {s} ({time.time() - t:.1f}s) =====\n{r.stdout}\n{r.stderr}")
    return time.time() - t0


def digest() -> dict:
    out = {}
    for p in sorted((HERE / "tables").glob("*.csv")) + sorted((HERE / "results").glob("*.csv")):
        out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def checks() -> dict:
    from PIL import Image

    res = {}
    res["tables_exist"] = all((HERE / "tables" / f"{t}.{e}").exists() for t in TABLES for e in ("csv", "tex"))
    res["figures_exist"] = all((HERE / "figures" / f"{f}.{e}").exists() for f in FIGURES for e in ("pdf", "png"))
    dpis = {}
    for f in FIGURES:
        d = Image.open(HERE / "figures" / f"{f}.png").info.get("dpi", (0, 0))
        dpis[f] = round(float(d[0]))
    res["png_dpi_ge_600"] = all(v >= 599 for v in dpis.values())
    res["dpi"] = dpis
    pat = re.compile(r"(set_title|plt\.title|suptitle)\(\s*(?!\"\"|'')[^)]")
    offenders = []
    for p in HERE.glob("*.py"):
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if pat.search(line) and "def " not in line and "_suptitle" not in line:
                offenders.append(f"{p.name}:{i}: {line.strip()}")
    res["no_titles"] = not offenders
    res["title_offenders"] = offenders
    # Anti-leakage: scaler / models are only fitted inside fl_core.fit_all on training covariates
    core = (HERE / "fl_core.py").read_text(encoding="utf-8")
    res["scaler_fit_on_train_only"] = "Xpool = np.vstack([c.X_tr for c in sc.clients])" in core and "X_te" not in \
        core.split("def fit_all")[1].split("def local_smoothness")[0]
    return res


def main() -> None:
    single = "--single" in sys.argv
    print("Run 1 (clean)")
    clean()
    t1 = run_pipeline()
    d1 = digest()
    res = checks()
    if not single:
        print("Run 2 (clean, reproducibility)")
        clean()
        t2 = run_pipeline()
        d2 = digest()
        res["reproducible"] = d1 == d2
        res["non_identical_files"] = [k for k in d1 if d1.get(k) != d2.get(k)]
        print(f"Runtime: run1 {t1:.1f}s, run2 {t2:.1f}s")
    print("\nExit criteria")
    ok = True
    for k, v in res.items():
        if isinstance(v, bool):
            ok &= v
            print(f"  [{'x' if v else ' '}] {k}")
        elif v:
            print(f"      {k}: {v}")
    log_path = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--log=")), None)
    if log_path:
        Path(log_path).write_text("\n".join(LOG) + "\n\nEXIT CRITERIA\n" +
                                  "\n".join(f"{k}: {v}" for k, v in res.items()), encoding="utf-8")
    (HERE / "results" / "exit_criteria.txt").write_text("\n".join(f"{k}: {v}" for k, v in res.items()),
                                                         encoding="utf-8")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
