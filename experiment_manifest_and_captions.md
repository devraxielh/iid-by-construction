# Experiment manifest — *IID by Construction: Using a Verified-Null Multi-Region Health Dataset to Isolate the Cost of Non-IID Heterogeneity in Federated Hospital-Demand Models*

**Versión:** final tras 3 rondas de revisión simulada (`review/`). Todas las cifras se copiaron de `tables/*.csv` y `results/*`, regenerados con `python scripts/run_all_experiments.py --log=scripts/review/run_round_03.log`. Dos ejecuciones limpias dieron resultados idénticos bit a bit.

---

## 0. Posicionamiento y aporte (R1.3, R1.5)

**Literatura de referencia y qué aporta este trabajo:**

| Línea | Qué hacen | Qué falta y aporta este trabajo |
|---|---|---|
| Benchmarks Non-IID (NIID-Bench, Li et al., ICDE 2022; LEAF; FedLab) | Particiones heterogéneas de datasets de imagen/texto, evaluadas por *accuracy* | No hay **oráculo**: se desconoce el riesgo mínimo alcanzable, así que no se puede separar "dificultad del problema" de "coste de federar". |
| FLamby (NeurIPS D&B 2022) | Datos médicos multi-sitio reales | La heterogeneidad es real pero **no controlable ni descomponible** por mecanismo. |
| Teoría de FedAvg bajo heterogeneidad (Li et al., 2020; Karimireddy et al., SCAFFOLD; Wang et al., FedNova) | Cotas de convergencia y sesgo del punto fijo | Este trabajo cuantifica empíricamente, frente a un oráculo, cuánto pesa la deriva en comparación con la mala especificación del modelo global. |

**Aporte (afirmación → evidencia):**
1. *Un banco de pruebas referenciado a un oráculo sobre una base de nulo verificado:* el generador se reconstruyó al bit (archivo 01, §2.5), la heterogeneidad se inyecta por mecanismo (covariables, tasa basal, concepto) y con magnitud graduada → §1, Tabla 1, Fig. 1.
2. *Descomposición del coste de la heterogeneidad:* casi todo el coste se debe a usar un modelo global; la federación en sí aporta una fracción mínima → Tabla 2, Tabla 3 (`Centralized` / `Centralized-FE`), Fig. 2d, Tabla 5.
3. *La regla de decisión depende del mecanismo:* personalizar parámetros sobra cuando p(y|x) es común y es imprescindible cuando difiere → Tabla 4 (H2), Fig. 6.

**Alcance (R1.5).** El espacio de comparación es **régimen de entrenamiento FL × clase de modelo**:
- 8 regímenes: global centralizado, centralizado con efectos fijos, local, FedAvg, FedProx, SCAFFOLD, FedRep y Ditto.
- 2 clases de modelo: GLM de Poisson y MLP.

Los árboles y el *boosting* federados (SecureBoost, agregación de histogramas) siguen un protocolo distinto y no permiten una comparación equivalente. Quedan fuera de alcance.

---

## 1. Diseño

### 1.1 Hipótesis
Base IID: el Kruskal–Wallis entre las 5 regiones no es significativo para ninguna covariable tras Holm (`results/iid_audit_kruskal.csv`); Tabla 1: p entre 0,216 y 0,908.

| Tipo (h ∈ {0; 0,25; 0,5; 0,75; 1}) | Mecanismo | Magnitud en h = 1 (`results/heterogeneity_magnitudes.json`) |
|---|---|---|
| Covariate shift | inclinación exponencial w ∝ exp(h·z·u_k) | desplazamiento medio de la media del cliente de 1,02 DE (0,14 con h = 0) |
| Prior (baseline) shift | b_k = log 10 + 0,40·h·o_k | razón máx./mín. de tasa basal entre clientes: 2,66 |
| Concept shift | β_k = β + 0,15·h·D_k | DE entre clientes de la pendiente de PM₂.₅: 0,151; el 30,8 % de los clientes tiene el signo del efecto invertido |

- **H1a:** con IID, FedAvg ≡ Centralized (TOST, margen ±0,01).
- **H1b:** la brecha FedAvg − Centralized crece con h.
- **H2:** la corrección de deriva y el término proximal ayudan sobre todo bajo *covariate shift*; la personalización solo bajo *prior* y *concept shift*.

### 1.2 Datos, partición y métrica
- 5 clientes (regiones). Entrenamiento: bloques de calendario 0–2 (422–465 filas por cliente). Test local: bloques 3–4 (147–175 filas). Sin barajar.
- Y semisintética ~ Poisson(λ), con log λ = b_k + z·β_k sobre 10 covariables reales estandarizadas con los momentos poblacionales del generador. Control adicional con el **target nulo real** (oráculo Poisson(10)).
- 50 réplicas por condición, con números aleatorios comunes: h = 0 es el mismo control IID para los tres tipos.
- **Métrica primaria (a priori):** exceso de deviance de Poisson por observación sobre el oráculo, promediado por cliente (ED).
- **Secundarias:** peor cliente, DE entre clientes, RMSE de pendientes, calibración y residuo de Pearson por cliente.

### 1.3 Métodos y tuning (R2.1, R2.7)
- Todos los GLM llevan penalización L2 sobre las pendientes. **Tuning simétrico:** cada método recibe una grilla de 5 valores para su hiperparámetro:
  - L2 ∈ {10⁻⁴, …, 1} para Centralized, Centralized-FE, Local, FedAvg, SCAFFOLD y FedRep.
  - μ ∈ {10⁻², …, 10²} para FedProx.
  - λ ∈ {10⁻², …, 10²} para Ditto.
- La selección se hace por deviance de validación ponderada sobre un **split interno cronológico** (el último 25 % del train de cada cliente) y después se reajusta en el train completo. El test nunca interviene.
- FL: T = 100 rondas y E = 5 pasos locales. El paso común es η = min(0,05; 0,9 / máx_k(L_k + L2_max)).
- Hiperparámetros elegidos con más frecuencia (`results/selected_hparams.csv`):
  - Ditto: λ = 100 en IID y *covariate shift* (≈ modelo global), 0,1 bajo *prior* y 0,01 bajo concepto (≈ local).
  - FedProx: μ = 100 en IID y *covariate shift*, 10 bajo *prior* y concepto.
- **Grillas ampliadas (Tabla S2, `07_grid_extension.py`):** L2 hasta 10², μ y λ de 10⁻³ a 10⁴, 50 réplicas pareadas.
  - Ya ningún método elige el límite superior, salvo Ditto con λ = 10⁴ (≈ modelo global) en el 50–52 % de las réplicas bajo IID y *covariate shift*.
  - Bajo *prior* y concepto, el ED de los métodos globales cambia ≤ 0,78 % (*prior*) y ≤ 0,51 % (concepto); el de los personalizados, ≤ 5,8 % (el máximo, FedRep bajo *prior*: 0,0087 → 0,0082, −5,8 %; Local +3,6 %).
  - Bajo IID y *covariate shift*, la grilla amplia **empeora** a varios métodos (+8 % a +52 %), por la varianza de seleccionar entre más candidatos con una validación interna pequeña.
  - Ninguna conclusión cambia: los ganadores por escenario solo se intercambian dentro de empates estadísticos. **Se mantiene la grilla de 5 valores como análisis primario.**
- **Configuración a priori (sin tuning; Tabla S1):** L2 = 10⁻⁴, μ = 1, λ = 1.

### 1.4 Pruebas (α = 0,05; Holm dentro de cada familia declarada)
- **H1a** (3 pruebas): Wilcoxon + TOST, con IC bootstrap pareado al 90 %. Margen 0,01 = 0,97 % de la deviance media del oráculo (1,027 por obs.) = 55 % de la brecha Local − Centralized bajo IID (0,018) (`results/h1a_tost.csv`).
- **H1b** (6 pruebas): Page L + ρ de Spearman. Se llama "tendencia creciente" solo si p_Holm < 0,05 y además el IC del contraste h = 1 frente a h = 0 excluye 0.
- **H2** (18 pruebas): Wilcoxon frente a FedAvg en h = 1, con r de rango-biserial.
- **Friedman** (4 pruebas) + Nemenyi (CD).
- **MLP** (20 pruebas, una sola familia; `results/mlp_tests.csv`).
- IC al 95 %: bootstrap con 2000 remuestreos sobre las réplicas.

### 1.5 Desviaciones respecto al diseño original
1. Sin MLP en el análisis principal: el GLM es convexo y coincide con la forma funcional del DGP. Se añadió un **estudio de robustez con MLP** (Tabla 5, Fig. 7; 10→16 tanh→1, T = 200 rondas, E = 5, η = 0,1, hiperparámetros **fijos e iguales** para todos los regímenes, 30 réplicas).
2. H2 se reformuló respecto a la ficha del catálogo, porque esta contradecía la motivación teórica de FedProx y SCAFFOLD.
3. Métrica: exceso sobre el oráculo.
4. Implementación: paso común seguro; paso proximal implícito en Ditto y FedProx; filas remuestreadas reordenadas cronológicamente.
5. Ablación: el efecto de **añadir un componente** a FedAvg (Tabla 3).
6. McNemar no aplica. Nemenyi se implementó con `scipy.stats.studentized_range`.
7. Módulo adicional `fl_core.py`. Pasos adicionales `06` y `07`. Figuras adicionales `fig_06` y `fig_07`. Tablas `table_05`, `S1` y `S2`.

### 1.6 Cambios de resultados entre rondas de revisión (R1.7, R2.8)
- **Ditto.** En la ronda 1, con λ = 1 fijo, parecía malo en todos los escenarios (por ejemplo, ED = 0,022 bajo IID). Con tuning es el método con **mejor rango medio** (3,03; ED = 0,005 bajo IID). La conclusión de la ronda 1 sobre Ditto era un artefacto de no ajustar el hiperparámetro.
- **Deriva bajo IID (H1a).** En la ronda 1, FedAvg − Centralized = +4,35 × 10⁻⁴ (p = 3,2 × 10⁻³); ahora es +8,1 × 10⁻⁵ (p = 0,92). Dos causas:
  - el paso común se calcula ahora con el mayor L2 de la grilla (η menor → menos deriva por paso local);
  - el orden cronológico de las filas remuestreadas cambia qué extracción de Poisson recibe cada fila.

  Ambas versiones son equivalentes por TOST. La deriva sigue siendo visible de forma directa en la sensibilidad a E (Fig. 3a).

---

## 2. Entorno y reproducibilidad
- Python 3.13.9 (Anaconda). Versiones exactas en `requirements.txt` (numpy 2.3.5, scipy 1.16.3, scikit-learn 1.7.2, pandas 2.3.3, matplotlib 3.10.6, seaborn 0.13.2, joblib 1.5.2, pillow 12.0.0, pyarrow 26.0.0). `SEED = 42`.
- `python scripts/run_all_experiments.py [--log=ruta]` borra `results/`, `tables/` y `figures/`, ejecuta 01 → 07 **dos veces** y compara los SHA-256 de todos los CSV.
- Tiempo: 295,9 s y 303,1 s (12 núcleos).
- Exit criteria: todos ✅ (`results/exit_criteria.txt`, `review/audit_round_03.json`). 7 tablas, 7 figuras a 600 DPI con su PDF, sin títulos, fuentes ≥ 8 pt, sin NaN, tablas en booktabs, mejor valor en negrita, segundo subrayado y ↓ en las tablas comparativas. Reproducible bit a bit.

---

## 3. Hallazgos (cifras de las tablas regeneradas)

**Tesis central (respaldada):** el coste de la heterogeneidad viene de usar *un único modelo global*, no de *federar*. Con *prior shift* en h = 1, el ED es:

| Método | ED (h = 1, *prior shift*) |
|---|---|
| Centralized | 1,278 ± 0,069 |
| FedAvg | 1,285 ± 0,068 |
| Centralized-FE (efectos fijos, sin federar) | **0,008 ± 0,007** |
| FedRep (federado) | 0,009 ± 0,007 |

La personalización del intercepto rinde lo mismo con o sin federación. Bajo *concept shift*, los modelos con pendientes comunes (Centralized 2,062; Centralized-FE 2,053; FedRep 2,041) fallan igual que FedAvg (2,077), y solo Local (0,032 ± 0,017) y Ditto (0,032 ± 0,014) lo evitan.

1. **H1a — equivalencia bajo IID: se acepta.**
   - FedAvg − Centralized = 8,1 × 10⁻⁵, IC 90 % [−2,6 × 10⁻⁴; 4,5 × 10⁻⁴] (Wilcoxon p_Holm = 0,92, r = −0,02).
   - FedProx −4,7 × 10⁻⁴ y SCAFFOLD 2,2 × 10⁻¹⁷: también equivalentes.
   - Target nulo real: Centralized y SCAFFOLD 0,014 ± 0,012, FedAvg 0,014 ± 0,012, Local 0,035 ± 0,030.
2. **H1b — la brecha federada crece con h solo bajo *prior* y *concept shift*, y es pequeña.**
   - *Prior*: Page L = 2343, p_Holm = 0,017, ρ = 0,188, Δ(h1−h0) = +0,0070 [0,0030; 0,0107].
   - *Concepto*: L = 2341, p_Holm = 0,017, ρ = 0,175, Δ = +0,0143 [0,0029; 0,0255].
   - *Covariate shift*: sin tendencia (p_Holm = 0,129).
   - En cambio, el ED total de FedAvg crece hasta 1,285 y 2,077 (L = 2750, p_Holm = 6,3 × 10⁻⁴⁵, ρ = 0,980 y 0,974). La brecha federada es el 0,5 % y el 0,7 % de esa pérdida.
3. **H2 — se confirma, con matices** (14 de 18 comparaciones significativas tras Holm).
   - *Covariate shift* h = 1: solo FedProx mejora a FedAvg (−1,5 × 10⁻³ [−2,6 × 10⁻³; −7,0 × 10⁻⁴], p_Holm = 0,018, r = −0,49). SCAFFOLD (p_Holm = 0,067), FedRep, Ditto y Centralized-FE no difieren. Local empeora (+0,018, p_Holm = 1,1 × 10⁻¹³, r = +1,00).
   - *Prior shift*: FedRep −1,277, Centralized-FE −1,277, Local −1,260 y Ditto −1,257 frente a FedAvg (todos p_Holm = 3,2 × 10⁻¹⁴, r = −1,00). FedProx −9,0 × 10⁻³ y SCAFFOLD −7,0 × 10⁻³ también son significativos, pero el efecto es trivial.
   - *Concept shift*: Local −2,045 y Ditto −2,045 (p_Holm = 3,2 × 10⁻¹⁴, r = −1,00). FedRep −0,036 y Centralized-FE −0,024 mejoran a FedAvg solo de forma marginal.
   - Friedman: global χ² = 465,6, p = 2,0 × 10⁻⁹⁶, W = 0,102; por tipo, W = 0,36 / 0,86 / 0,63. Rangos medios: Ditto 3,03 · FedRep 3,97 · Centralized-FE 4,11 · Local 4,46 · FedProx 4,76 · SCAFFOLD 5,14 · Centralized 5,15 · FedAvg 5,39 (CD = 0,41). El orden se invierte según el mecanismo de heterogeneidad (Fig. 6).
4. **Robustez con MLP (Tabla 5, Fig. 7).**
   - La penalización federada sigue siendo despreciable: FedAvg frente a Centralized varía −0,07 % (IID), +0,66 % (covariables), −0,23 % (*prior*) y +0,61 % (concepto).
   - Bajo *prior shift*, Centralized-FE (0,022 ± 0,012) rinde mejor que FedRep con cabeza local (0,053 ± 0,020; +139 %, p_Holm < 10⁻⁴). **Personalizar más parámetros de los que exige el mecanismo cuesta precisión** con unas 440 filas por cliente (R1.6).
   - Bajo concepto, FedRep (0,155) y Local (0,141) rinden un 93 % mejor que los modelos globales (≈ 2,15).
   - El MLP sin ajustar tiene un ED basal mayor que el GLM (≈ 0,020 frente a 0,005 bajo IID).
5. **Sensibilidad** (configuración a priori).
   - (a) Deriva según E (brecha FedAvg − Centralized): bajo *covariate shift* sube de 0 (E = 1) a 1,36 × 10⁻³ (E = 20); bajo concepto, a 0,075. SCAFFOLD = 0 para todo E.
   - (b) Ditto λ: 100 es óptimo bajo IID (0,0046); λ ≤ 0,1 bajo *prior* y concepto (0,027–0,030).
   - (c) DP-FedAvg (δ = 10⁻⁵; σ = 34,7 / 12,0 / 4,9): ED de 0,039 / 0,013 / 0,0098 frente a 0,0045 sin DP (IID). ε = 1 multiplica la pérdida por unas 9.
6. **Coeficientes y errores.**
   - Bajo concepto h = 1, los modelos con pendientes comunes no recuperan las pendientes de cada cliente (correlación ≈ 0, error absoluto medio 0,114); Ditto y Local sí (r = 0,992, error 0,014).
   - Bajo *prior* h = 1, el residuo de Pearson medio de FedAvg sube de forma monótona con la tasa basal del cliente (−1,31; −0,78; −0,10; +0,79; +1,91), mientras que FedRep queda en |·| ≤ 0,02. La calibración agregada lo oculta en parte (error medio por decil: FedAvg 0,35; FedRep 0,059).
   - Bajo *prior*, Ditto estima las pendientes peor (RMSE 0,0146) que FedRep (0,0072) y Centralized-FE (0,0070), porque su λ bajo lo acerca al ajuste local (Fig. 4c).

**Decisiones:** se acepta H1a (equivalencia). H1b se confirma solo para *prior* y concepto, con un efecto pequeño frente al coste total. H2 se confirma para la personalización; la corrección de deriva no aporta una mejora relevante.

**Limitaciones y amenazas a la validez (R2.5):**
- (i) 5 clientes *cross-silo*, unas 440 filas cada uno, participación completa y sin desbalance de tamaño.
- (ii) Las covariables son ruido independiente: no reproducen la autocorrelación ni la correlación cruzada reales de la contaminación.
- (iii) La heterogeneidad es sintética, de magnitudes fijadas a priori (§1.1) y sin mezclar tipos.
- (iv) La inferencia es condicional al diseño: todas las réplicas reutilizan el mismo conjunto de covariables y los test locales son pequeños (147–175 filas).
- (v) El MLP no se ajustó.
- (vi) DP solo a nivel de registro, sin submuestreo.
- (vii) Sin validación externa en redes hospitalarias reales (siguiente paso propuesto: FLamby, o datos reales de varias ciudades).
- (viii) El tuning por validación interna no es inocuo con validaciones pequeñas (unas 110 filas por cliente): frente a la configuración a priori (Tabla S1), **empeora** a los métodos globales bajo IID (FedAvg +10,9 %, Centralized +12,8 %) y bajo *covariate shift* (+18,0 % a +20,4 %), porque elige L2 = 1 y sobrerregulariza. Las conclusiones cualitativas no cambian.

---

## 4. Captions LaTeX

Convenciones de las tablas (R3.8): **negrita** = mejor media de la columna; subrayado = segunda; ↓ = menor es mejor; † = IC 95 % que excluye 0.

```latex
\begin{table}[t]\centering\small
\caption{Client cohort (temporal split by calendar block) and IID audit. Each row reports a descriptor
per client and the $p$-value of a between-client homogeneity test ($\chi^2$ uniformity for counts,
Kruskal--Wallis for means, Levene for the SD, $\chi^2$ independence for the lockdown rate).
Respiratory admissions are the real (null) target.}
\label{tab:cohort}
\input{tables/table_01_cohort_summary.tex}
\end{table}

\begin{table*}[t]\centering\small
\caption{Excess test Poisson deviance per observation over the oracle (lower is better, $\downarrow$),
client-macro mean $\pm$ SD over 50 simulation replicates. All methods are tuned with the same 5-point
budget on an inner chronological validation split. $^{a}$Real null target: mean $\pm$ SD over the
5 clients (single realisation). Bold: best; underlined: second best (ties share the mark).
95\% bootstrap CIs are given in the accompanying CSV.}
\label{tab:main}
\input{tables/table_02_main_results_cv.tex}
\end{table*}

\begin{table*}[t]\centering\small
\caption{Component ablation relative to FedAvg: paired difference $\Delta$ in excess deviance
(mean [95\% bootstrap CI], 50 replicates; negative values indicate improvement). Bold: most negative
mean per column; underlined: second; $^{\dagger}$: CI excludes zero. The pooled rows (global model,
client fixed effects) separate the cost of federation from the cost of a global model.}
\label{tab:ablation}
\input{tables/table_03_ablation_study.tex}
\end{table*}

\begin{table*}[t]\centering\scriptsize
\caption{Hypothesis tests with Holm--Bonferroni correction within each pre-declared family.
H1a: Wilcoxon signed-rank and TOST equivalence (margin $\pm0.01$, paired bootstrap 90\% CI).
H1b: Page's L trend test over $h$ (blocks = replicates) with Spearman $\rho$; ``increasing'' requires
$p_{\mathrm{Holm}}<0.05$ and an $h=1$ vs.\ $h=0$ CI excluding zero. H2: Wilcoxon signed-rank versus
FedAvg at $h=1$ with rank-biserial $r$. Friedman: Kendall's $W$.}
\label{tab:tests}
\input{tables/table_04_statistical_significance.tex}
\end{table*}

\begin{table}[t]\centering\small
\caption{Robustness with a non-convex model: excess deviance ($\downarrow$, mean $\pm$ SD over 30
replicates) of a federated one-hidden-layer MLP (16 tanh units, Poisson output) trained with identical,
untuned hyper-parameters under every regime. Bold: best; underlined: second.}
\label{tab:mlp}
\input{tables/table_05_mlp_robustness.tex}
\end{table}

\begin{table}[t]\centering\small
\caption{Supplementary S1: a-priori configuration without tuning (L2 $=10^{-4}$, $\mu=1$,
$\lambda=1$). Same layout and conventions as Table~\ref{tab:main}.}
\label{tab:apriori}
\input{tables/table_S1_apriori_configuration.tex}
\end{table}

\begin{table}[t]\centering\small
\caption{Supplementary S2: excess deviance ($\downarrow$) after re-tuning with extended grids
(L2 up to $10^{2}$; $\mu,\lambda\in[10^{-3},10^{4}]$), same 50 paired replicates. Paired differences
to Table~\ref{tab:main} and the share of boundary selections are given in the CSV.}
\label{tab:grid}
\input{tables/table_S2_grid_extension.tex}
\end{table}
```

```latex
\begin{figure*}[t]\centering
\includegraphics[width=\textwidth]{figures/fig_01_data_distribution_or_manifold.pdf}
\caption{Client data under the semi-synthetic protocol. (a,b)~PCA of reference-standardised training
covariates of the five clients in the IID control and under covariate shift ($h=1$); large markers are
client centroids. (c)~Simulated daily admissions per client, control (white) vs.\ prior shift ($h=1$).
(d)~True client PM$_{2.5}$ log rate ratio per SD versus the concept-shift level (10 replicates; dashed:
global effect 0.08).}
\label{fig:data}
\end{figure*}

\begin{figure*}[t]\centering
\includegraphics[width=\textwidth]{figures/fig_02_main_performance_comparison.pdf}
\caption{Excess test Poisson deviance per observation (log scale) versus the heterogeneity level $h$
for (a)~covariate, (b)~prior and (c)~concept shift; eight tuned training regimes. (d)~Federated penalty
(FedAvg minus Centralized). Lines: mean of 50 replicates; bands: 95\% bootstrap CI. Centralized and
SCAFFOLD coincide; Centralized-FE and FedRep nearly coincide.}
\label{fig:main_performance}
\end{figure*}

\begin{figure*}[t]\centering
\includegraphics[width=\textwidth]{figures/fig_03_ablation_and_sensitivity.pdf}
\caption{Sensitivity analyses under the a-priori configuration (50 replicates; mean, 95\% bootstrap CI).
(a)~Client drift: gap to Centralized versus local steps $E$ (covariate shift, $h=1$). (b)~Ditto coupling
$\lambda$ in the IID control and under each shift type at $h=1$. (c)~Record-level DP-FedAvg (DP-GD, clip 5, 50 rounds, $E=1$, RDP accounting,
$\delta=10^{-5}$) versus $\varepsilon$.}
\label{fig:sensitivity}
\end{figure*}

\begin{figure*}[t]\centering
\includegraphics[width=\textwidth]{figures/fig_04_explainability_shap_or_latent.pdf}
\caption{Coefficient recovery against the known data-generating process. (a)~Estimated versus true
client PM$_{2.5}$ log rate ratio under concept shift ($h=1$; 250 client-replicates; dashed: identity).
(b,c)~RMSE of the ten slope estimates versus $h$ under concept and prior shift (mean, 95\% bootstrap CI).}
\label{fig:coefficients}
\end{figure*}

\begin{figure*}[t]\centering
\includegraphics[width=\textwidth]{figures/fig_05_error_calibration_or_case_study.pdf}
\caption{Error analysis on local test sets. (a)~Client-level excess deviance under concept shift
($h=1$; 250 client-replicates per method; log scale). (b)~Calibration by decile of predicted rate under
prior shift ($h=1$, pooled over clients and replicates). (c)~Mean Pearson residual with clients ordered
by their true baseline rate within each replicate (prior shift, $h=1$; 95\% bootstrap CI).}
\label{fig:errors}
\end{figure*}

\begin{figure}[t]\centering
\includegraphics[width=\columnwidth]{figures/fig_06_critical_difference.pdf}
\caption{Average Friedman ranks (lower is better) with Nemenyi intervals ($\pm$CD/2, $\alpha=0.05$);
non-overlapping intervals differ significantly. (a)~All 650 unique scenarios; (b--d)~covariate, prior
and concept shift at $h=1$ (50 replicates). Shaded: interval of the best method.}
\label{fig:cd}
\end{figure}

\begin{figure*}[t]\centering
\includegraphics[width=\textwidth]{figures/fig_07_mlp_robustness.pdf}
\caption{Federated MLP robustness study: excess deviance (log scale; mean and 95\% bootstrap CI over
30 replicates) of six training regimes in the IID control and at $h=1$ for each shift type.}
\label{fig:mlp}
\end{figure*}
```

---

## 5. Draft Results paragraph (English)

In the IID control, federated averaging was statistically equivalent to centralized training (excess deviance 0.0052 ± 0.0070 vs. 0.0051 ± 0.0062; paired difference 8.1 × 10⁻⁵, 90% CI [−2.6 × 10⁻⁴, 4.5 × 10⁻⁴], well within the ±0.01 margin; Tables~\ref{tab:main} and \ref{tab:tests}). Client drift was visible only when local computation was increased (Fig.~\ref{fig:sensitivity}a) and was removed by SCAFFOLD. Injected heterogeneity was costly, but the cost was almost entirely attributable to fitting a single global model rather than to federation (Fig.~\ref{fig:main_performance}). Under prior shift at h = 1, the excess deviance reached 1.285 ± 0.068 for FedAvg and 1.278 ± 0.069 for centralized training on pooled data. A pooled model with client fixed effects reduced it to 0.008 ± 0.007, matching federated partial personalization (FedRep, 0.009 ± 0.007). The federated penalty grew with h only weakly (Page's L, p_Holm = 0.017, ρ ≤ 0.19) and amounted to less than 1% of the total loss. Under concept shift, every model with shared slopes failed (2.04–2.08), and only local and Ditto models (0.032) recovered the client-specific PM₂.₅ effects (r = 0.99; Fig.~\ref{fig:coefficients}). With hyper-parameters tuned on an inner validation split, Ditto adapted its coupling to the mechanism (λ = 100 under IID, λ ≤ 0.1 under prior or concept shift) and obtained the best average rank (3.03; Friedman χ² = 465.6, p = 2.0 × 10⁻⁹⁶; Fig.~\ref{fig:cd}). Drift-correcting methods brought no practically relevant gain under covariate shift (SCAFFOLD p_Holm = 0.067; FedProx −1.5 × 10⁻³). These conclusions held for a federated MLP, for which the federated penalty stayed below 0.7% in every scenario (Table~\ref{tab:mlp}). The MLP also showed that personalizing more parameters than the shift requires is harmful: a local output head was 139% worse than client fixed effects under prior shift. Extending the tuning grids changed no conclusion (Table~\ref{tab:grid}). Pooled calibration masked a systematic client-level bias of global models under prior shift (mean Pearson residual from −1.31 to +1.91 across clients; Fig.~\ref{fig:errors}). Finally, record-level differential privacy at ε = 1 increased the excess deviance roughly ninefold at about 440 records per client.
