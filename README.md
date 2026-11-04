# IID by Construction

Code, results and manuscript sources for the paper

> **IID by Construction: Using a Verified-Null Multi-Region Health Dataset to Isolate the Cost of Non-IID Heterogeneity in Federated Hospital-Demand Models**
> Rodrigo Garcia, Adrian Doria, Jessica Polo — Universidad del Sinú

## Overview

Federated learning lets hospitals train shared demand models without pooling records, but its losses under heterogeneous clients are rarely separated from task difficulty. This repository builds an **oracle-referenced testbed** on the public *Air Quality, Weather, and Respiratory Health* dataset, whose random generator we reconstructed exactly (all 24 numeric columns are independent draws with seed 42). Because its covariates are independent across the five regional clients, the IID case is a **verified control**.

On the real covariates, a semi-synthetic Poisson outcome is generated with **covariate, prior (baseline) or concept shift** of graded intensity $h \in \{0, 0.25, 0.5, 0.75, 1\}$, and eight training regimes of a Poisson GLM are compared:

| Regime | Shared parameters |
|---|---|
| Centralized | all |
| Centralized-FE | slopes (client fixed-effect intercepts) |
| Local | none |
| FedAvg, FedProx, SCAFFOLD | all |
| FedRep | slopes (local intercept) |
| Ditto | personal models coupled to the global model by $\lambda$ |

Every regime receives the same tuning budget (5-point grid, inner chronological validation), each condition is replicated 50 times (650 scenarios), and models are scored by **excess Poisson deviance over the oracle** on temporally held-out calendar blocks.

## Main findings

| Finding | Value | Source |
|---|---|---|
| FedAvg vs. centralized in the IID control | paired difference $8.1\times10^{-5}$, 90% CI within $\pm0.01$ (TOST equivalent) | `scripts/results/h1a_tost.csv` |
| Excess deviance of FedAvg, prior / concept shift ($h=1$) | 1.285 / 2.077 | `table_02` |
| Share of that loss caused by federation | 0.55% / 0.69% | `table_03` |
| Prior-shift loss removed by client intercepts (Centralized-FE / FedRep) | 99.35% / 99.32% | `table_03` |
| Concept-shift loss removed by Local / Ditto | 98.48% / 98.45% | `table_03` |
| Cost of local training when personalization is not needed (IID / covariate) | +346.2% / +241.7% | `table_03` |
| Best average Friedman rank over all scenarios | Ditto (3.03) | `scripts/results/friedman_ranks.csv` |
| Federated MLP: FedAvg vs. centralized, all scenarios | −0.23% to +0.66% | `table_05` |

In short: the cost of federation is small; the cost of a **single global model** can be large, and which parameters to personalize depends on the heterogeneity mechanism.

## Repository structure

```
.
├── refinery_loop.py              # audit/cleaning loop of the raw file
├── preprocess_pipeline.py        # stateless cleaning + fold-level (stateful) pipeline
├── eda_stats.py                  # exploratory and statistical analysis
├── outputs/
│   ├── reconstruct_generator.py  # exact reconstruction of the data generator (seed 42)
│   └── *.json                    # audit, EDA and reconstruction results
├── dataset_ml_ready.csv/.parquet # cleaned dataset used by the experiments
├── diccionario_datos.csv         # data dictionary
├── 01–04_*.md                    # cleaning log, EDA report, paper catalog, findings report (Spanish)
├── scripts/
│   ├── 00_config_and_utils.py    # seeds, paths, plotting style, table/figure export, statistics helpers
│   ├── fl_core.py                # DGP, Newton solver, federated regimes, tuning
│   ├── 01_data_splitting_and_prep.py      # clients, temporal split, IID audit  -> table_01, fig_01
│   ├── 02_baseline_and_sota_models.py     # 650 scenarios x 8 regimes           -> table_02, fig_02
│   ├── 03_ablation_and_sensitivity.py     # ablation, local steps, Ditto lambda, DP -> table_03, table_S1, fig_03
│   ├── 04_statistical_tests.py            # Wilcoxon, TOST, Page, Friedman–Nemenyi -> table_04, fig_06
│   ├── 05_explainability_and_error.py     # coefficient recovery, calibration    -> fig_04, fig_05
│   ├── 06_mlp_robustness.py               # federated MLP                        -> table_05, fig_07
│   ├── 07_grid_extension.py               # extended tuning grids                -> table_S2
│   ├── run_all_experiments.py             # full pipeline, exit checks, reproducibility re-run
│   ├── requirements.txt
│   ├── experiment_manifest_and_captions.md
│   ├── results/   tables/   figures/      # generated artifacts (.csv/.parquet, .csv/.tex, .pdf/.png 600 DPI)
│   └── review/                            # simulated peer-review rounds, audit script, changelog
└── paper/                        # MDPI LaTeX manuscript (main.tex + section files + references.bib)
```

## Data

The raw data are the public Kaggle dataset
[*Air Quality, Weather, and Respiratory Health*](https://www.kaggle.com/datasets/khushikyad001/air-quality-weather-and-respiratory-health)
(3000 daily records, 26 columns, five regions). The experiments read the cleaned file `dataset_ml_ready.csv` from the repository root. To rebuild it from the raw file:

```bash
python -I preprocess_pipeline.py path/to/air_quality_health_dataset.csv .
```

Optional audit, EDA and generator reconstruction:

```bash
python -I refinery_loop.py path/to/air_quality_health_dataset.csv
python -I eda_stats.py path/to/air_quality_health_dataset.csv
python -I outputs/reconstruct_generator.py path/to/air_quality_health_dataset.csv outputs/generator_reconstruction.json
```

## Reproducing the experiments

Requirements: Python 3.13 and the pinned packages in `scripts/requirements.txt` (NumPy, SciPy, scikit-learn, pandas, Matplotlib, seaborn, joblib, Pillow, PyArrow). The federated solvers, the Newton solver and the neural network are written directly in NumPy; no federated-learning framework is needed.

```bash
pip install -r scripts/requirements.txt
python scripts/run_all_experiments.py
```

`run_all_experiments.py` deletes `results/`, `tables/` and `figures/`, runs scripts 01→07, checks the exit criteria (all tables and figures present, 600 DPI, no figure titles) and then runs the whole pipeline a second time, comparing SHA-256 hashes of every result file. Both runs produce identical results. One run takes about five minutes on a desktop CPU (scenarios run in parallel with joblib). Use `--single` for one run only, and `--log=path` to save the console log.

All randomness derives from the global seed 42; replicate `r` uses seed `42·1000 + r`, shared across shift types and levels (common random numbers).

## Building the paper

The manuscript uses the MDPI template (`paper/Definitions/`). From `paper/`:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

`resultsanddiscussion.tex` reads tables and figures from `../scripts/` (macro `\resultsdir`), so compile inside `paper/` with `scripts/` next to it.

## Citation

If you use this code or testbed, please cite the paper (reference to be completed upon publication) and the original dataset:

```bibtex
@misc{khushikyad001airquality,
  author       = {{khushikyad001}},
  title        = {Air Quality, Weather, and Respiratory Health},
  howpublished = {Kaggle dataset},
  url          = {https://www.kaggle.com/datasets/khushikyad001/air-quality-weather-and-respiratory-health}
}
```

## License

<!-- TODO: choose a license for the code (e.g. MIT) and check the Kaggle dataset license before redistributing dataset_ml_ready.csv. -->
License to be defined.

## Contact

Rodrigo Garcia — rodrigogarcia@unisinu.edu.co
