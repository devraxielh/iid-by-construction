# 04 · Reporte de hallazgos experimentales

**Paper:** *IID by Construction: Using a Verified-Null Multi-Region Health Dataset to Isolate the Cost of Non-IID Heterogeneity in Federated Hospital-Demand Models*
**Fecha:** 2026-11-03 · **Estado del protocolo:** *Accept* tras 3 rondas de revisión simulada (`scripts/review/CERTIFICATE_OF_PUBLISHABILITY.md`)

**Fuentes y cómo se obtuvieron las cifras:**
- Las cifras directas se copiaron de `scripts/tables/*.csv` y `scripts/results/*`.
- Toda cifra derivada (Δ, % relativo, ratios de DE, rankings) la calculó un script de verificación ejecutado en el scratchpad (`verify_results.py`), no a mano.
- **Métrica primaria:** exceso de deviance de Poisson en test por observación, sobre el oráculo (ED; **menor es mejor**). Por eso un **% relativo negativo indica mejora**.
- **Validación:** *hold-out* temporal (bloques de calendario 0–2 para entrenar y 3–4 para test local), 5 clientes, 50 réplicas de simulación por condición con números aleatorios comunes, e IC 95 % por bootstrap (2000 remuestreos).

> **Encuadre.** No hay un único "método propuesto". El aporte es un **banco de pruebas con oráculo** y una **descomposición del coste de la heterogeneidad**. Los contrastes centrales son: (i) FedAvg frente a Centralized, que mide la penalización por federar; (ii) los métodos personalizados frente a los globales, que mide el coste de usar un modelo global; (iii) Centralized-FE frente a FedRep, que mide si la personalización depende de federar.

---

## 1. Resumen ejecutivo y contribuciones empíricas clave

- **H1. Federar no cuesta nada medible con datos IID.** FedAvg obtiene ED = 0,0052 ± 0,0070 y Centralized 0,0051 ± 0,0062 (Δ = +0,0001, +1,6 % relativo). La diferencia pareada es 8,1 × 10⁻⁵, con IC 90 % [−2,6 × 10⁻⁴; 4,5 × 10⁻⁴] dentro del margen de equivalencia ±0,01 (TOST). Wilcoxon p_Holm = 0,92, r = −0,02 (fuente: `tables/table_02_main_results_cv.csv`, `results/h1a_tost.csv`, `tables/table_04_statistical_significance.csv`).
- **H2. El coste de la heterogeneidad viene del modelo global, no de la federación.** Con *prior shift* en h = 1, FedAvg llega a 1,285 ± 0,068 y Centralized a 1,278 ± 0,069. La brecha federada (+0,0070) es solo el **0,55 %** de la pérdida de FedAvg; bajo concepto, el **0,69 %**. Un modelo **centralizado** con efectos fijos por cliente baja la pérdida a 0,0083 ± 0,0075 (−99,35 % frente a Centralized) y FedRep, **federado**, a 0,0087 ± 0,0074 (+5,2 % frente a Centralized-FE). La ganancia de personalizar es la misma con o sin federación (fuente: `table_02`, `table_03`).
- **H3. La personalización solo ayuda cuando p(y|x) difiere entre clientes.**
  - *Concept shift* h = 1: Ditto (0,0323 ± 0,0142) y Local (0,0316 ± 0,0168) reducen el ED de FedAvg en −98,4 % y −98,5 % (p_Holm = 3,2 × 10⁻¹⁴, r = −1,00).
  - IID y *covariate shift*: Local empeora un +346 % y un +242 % (p_Holm = 1,1 × 10⁻¹³).
  - El orden de los métodos se invierte según el mecanismo: Friedman χ² = 465,6, p = 2,0 × 10⁻⁹⁶, W de Kendall = 0,102 (fuente: `table_04`, `results/friedman_ranks.csv`).
- **H4. Ditto con tuning es el régimen más robusto.** Tiene el mejor rango medio (3,03; CD = 0,41) porque la validación interna elige el extremo adecuado según el mecanismo: λ = 100 bajo IID y covariables, λ = 0,01–0,1 bajo *prior* y concepto (fuente: `friedman_ranks.csv`, `selected_hparams.csv`). Con λ fijo en 1 a priori era mucho peor (−74,8 % de ED gracias al tuning bajo IID; `table_S1`).
- **H5. Las conclusiones se mantienen con un modelo no convexo.** En un MLP federado, la penalización FedAvg − Centralized es −0,07 %, +0,66 %, −0,23 % y +0,61 % en los cuatro escenarios. Pero personalizar de más cuesta: con *prior shift*, FedRep con cabeza local (0,0533) es un +139 % peor que Centralized-FE (0,0223) (fuente: `table_05_mlp_robustness.csv`, `results/mlp_tests.csv`).

**Síntesis para el Abstract:**
```latex
Using an oracle-referenced testbed built on a verified-null multi-region health dataset, we show that
federated averaging is statistically equivalent to centralized training under IID data (paired
difference $8.1\times10^{-5}$ in excess deviance, 90\% CI within $\pm0.01$), and that the large losses
caused by prior and concept shift (excess deviance up to 1.29 and 2.08) stem almost entirely from fitting a
single global model rather than from federation, which accounts for less than 1\% of the loss. Client
fixed effects recover 99.4\% of the loss under prior shift whether trained centrally or federatively,
whereas only fully personalized models (Ditto, local) recover it under concept shift ($-98.4\%$,
$p_{\mathrm{Holm}}<10^{-13}$); the same personalization is harmful when the conditional distribution is shared.
```

---

## 2. Análisis de las tablas (`scripts/tables/`)

### Tabla 1 — `table_01_cohort_summary` (cohorte y auditoría IID)
- **Propósito:** describir la partición temporal por cliente y comprobar que la base es IID entre regiones.
- **Cifras:**
  - Entrenamiento: entre 422 (West) y 465 (East) filas por cliente, 2190 en total.
  - Test: entre 147 (Central) y 175 (South), 810 en total.
  - Ningún descriptor difiere entre clientes: n de entrenamiento χ² p = 0,616 y de test p = 0,560; admisiones media Kruskal–Wallis p = 0,216 y DE Levene p = 0,297; PM₂.₅ p = 0,908; temperatura p = 0,510; movilidad p = 0,613; tasa de confinamiento χ² p = 0,733.
  - La auditoría completa de 11 covariables, con Holm, no encuentra diferencias (`results/iid_audit_kruskal.csv`).
- **Lectura:** el control IID está garantizado por construcción y verificado estadísticamente. Toda heterogeneidad posterior es inyectada y, por tanto, atribuible.

```latex
Table~\ref{tab:cohort} summarises the five regional clients. Training sizes range from 422 to 465 daily
records and local test sizes from 147 to 175. None of the descriptors differs between clients
(all $p\geq0.216$), confirming that the base data are IID across clients before heterogeneity is injected.
```

### Tabla 2 — `table_02_main_results_cv` (resultado principal, con tuning)
- **Propósito:** comparar el ED de los 8 regímenes en el target nulo real, el control IID y los tres tipos de *shift* en h = 1.
- **Primero y segundo puesto por columna:**

| Escenario | 1.º | 2.º | Peor |
|---|---|---|---|
| Target nulo real | Centralized = SCAFFOLD 0,0137 ± 0,0116 (empate exacto) | FedAvg 0,0140 ± 0,0123 | Local 0,0350 |
| IID (h = 0) | FedProx 0,0046 ± 0,0061 | Centralized = SCAFFOLD 0,0051 ± 0,0062 | Local 0,0232 |
| Covariables (h = 1) | FedProx 0,0060 ± 0,0055 | Centralized = SCAFFOLD 0,0069 ± 0,0064 | Local 0,0259 |
| *Prior* (h = 1) | Centralized-FE 0,0083 ± 0,0075 | FedRep 0,0087 ± 0,0074 | FedAvg 1,2853 |
| Concepto (h = 1) | Local 0,0316 ± 0,0168 | Ditto 0,0323 ± 0,0142 | FedAvg 2,0768 |

- **Estabilidad (DE entre réplicas):**
  - La menor DE coincide casi siempre con el ganador: FedProx (0,0061 IID; 0,0055 covariables), FedRep (0,0074 *prior*) y Ditto (0,0142 concepto).
  - Bajo concepto, Ditto es más estable que Local (ratio DE_Local / DE_Ditto = 1,18) con una media prácticamente igual (+2,1 %). Es el mejor compromiso entre sesgo y varianza.
  - Bajo *prior*, la DE de los modelos globales es 9,2 veces la de FedRep; bajo concepto, 37,4 veces la de Ditto.
- **Compromisos (*trade-offs*):**
  - La personalización completa (Local) cuesta mucho cuando no hace falta: +346 % bajo IID y +242 % bajo covariables, frente a FedAvg.
  - La personalización parcial (FedRep, Centralized-FE) cuesta poco cuando sobra (+44 % bajo IID, unos 0,002 en términos absolutos), pero solo corrige el *prior shift* (−99,3 %), no el de concepto (−1,7 %).
  - SCAFFOLD reproduce a Centralized hasta precisión de máquina en todos los escenarios (Δ = 0,0000).
- **Advertencia de interpretación:** FedProx gana bajo IID y covariables, pero su ventaja frente a FedAvg no es significativa bajo IID (IC 95 % [−0,00126; 0,00012]). Bajo covariables es significativa pero absolutamente pequeña (−0,0015, −20,3 %; `table_03`).

```latex
As shown in Table~\ref{tab:main}, all global-model regimes are within $5\times10^{-4}$ of each other in the
IID control and under covariate shift, with FedProx marginally best (0.0046 and 0.0060), whereas fully
local training is 3.4--4.5 times worse. Under prior shift, every regime with a shared intercept fails
(excess deviance 1.28--1.29), while client fixed effects, whether centralized (0.0083) or federated
through FedRep (0.0087), remove more than 99\% of this loss. Under concept shift, only fully personalized
models succeed (local 0.0316, Ditto 0.0323), and Ditto achieves this with lower replicate-to-replicate
variability (SD 0.0142 vs.\ 0.0168).
```

### Tabla 3 — `table_03_ablation_study` (efecto de añadir un componente a FedAvg)
- **Propósito:** medir el efecto pareado, por réplica, de cada componente de FL frente a FedAvg (Δ ED; † = IC 95 % que excluye 0).
- **Cifras clave:**
  - **Término proximal (FedProx):** −0,00154 [−0,00256; −0,00070] bajo covariables (−20,3 %), −0,0090 bajo *prior* (−0,70 %) y −0,0115 bajo concepto (−0,55 %). Bajo IID, n.s.
  - **Corrección de deriva (SCAFFOLD)** = **datos agrupados con modelo global (Centralized)**, idénticos en todas las columnas: −0,00067 bajo covariables (−8,9 %) y −0,0144 bajo concepto (−0,69 %). Es exactamente la "penalización por federar".
  - **Cabeza local de intercepto (FedRep):** −1,2765 bajo *prior* (−99,32 %), pero +0,0023 bajo IID (+43,9 %, †).
  - **Efectos fijos centralizados (Centralized-FE):** −1,2770 bajo *prior* (−99,35 %), de nuevo +0,0022 bajo IID (+41,5 %, †).
  - **Modelos personales proximales (Ditto con tuning):** −2,0445 bajo concepto (−98,45 %) y −1,2566 bajo *prior* (−97,76 %); **no empeora bajo IID** (+0,27 %, n.s.).
  - **Sin federación (Local):** −2,0452 bajo concepto (−98,48 %), pero +0,0180 bajo IID (+346 %).
- **Componente crítico:** depende del mecanismo. Para el *prior shift*, un intercepto propio por cliente; para el concepto, pendientes propias. **La corrección de deriva nunca es el componente crítico:** su efecto máximo es el 0,7 % de la pérdida de FedAvg.

```latex
The ablation in Table~\ref{tab:ablation} decomposes the loss of FedAvg. Pooling the data while keeping a
global model (equivalently, removing client drift with SCAFFOLD) changes the excess deviance by at most
0.0144 ($-0.69\%$), which is the entire federated penalty. By contrast, a client-specific intercept
removes 99.3\% of the loss under prior shift, and client-specific slopes (Ditto, local) remove 98.4\%
under concept shift. Both forms of personalization carry a small but significant cost in the IID control
($+0.0022$ to $+0.0180$), except Ditto, whose tuned coupling makes it indistinguishable from FedAvg there
($+0.27\%$, CI including zero).
```

### Tabla 4 — `table_04_statistical_significance` (pruebas de hipótesis)
- **Propósito:** contrastar las hipótesis declaradas, con Holm dentro de cada familia.
- **Cifras:**
  - **H1a**, 0/3 significativas: es el resultado esperado, porque se trata de una equivalencia. Las tres comparaciones son equivalentes por TOST.
  - **H1b**, 4/6 significativas. Brecha federada: *prior* L = 2343, p_Holm = 0,017, ρ = 0,188, Δ(h1−h0) = +0,0070 [0,0030; 0,0107]; concepto L = 2341, p_Holm = 0,017, ρ = 0,175, +0,0143 [0,0029; 0,0255]; covariables n.s. (p_Holm = 0,129). Pérdida total de FedAvg: *prior* y concepto p_Holm = 6,3 × 10⁻⁴⁵ (ρ = 0,980 y 0,974); covariables n.s. (p_Holm = 0,054).
  - **H2**, 14/18 significativas. Bajo covariables solo FedProx mejora a FedAvg (p_Holm = 0,018, r = −0,49) y Local empeora (r = +1,00); SCAFFOLD n.s. (p_Holm = 0,067). Bajo *prior* y concepto las 12 comparaciones son significativas.
  - **Friedman**, 4/4: global W = 0,102; por tipo, W = 0,359 (covariables), 0,862 (*prior*), 0,631 (concepto).
- **Veredicto:** la evidencia para H1b es significativa pero de efecto pequeño (ρ < 0,19 frente a ρ ≈ 0,98 de la pérdida total). Debe presentarse así: "detectable pero despreciable frente al coste total".

```latex
Table~\ref{tab:tests} reports all hypothesis tests with Holm correction within families. Federated
averaging is formally equivalent to centralized training in the IID control. The federated penalty
increases with the heterogeneity level under prior and concept shift ($p_{\mathrm{Holm}}=0.017$), but with
a small monotone association ($\rho\leq0.19$) compared with the total loss ($\rho\geq0.97$,
$p_{\mathrm{Holm}}=6.3\times10^{-45}$). Under covariate shift neither quantity shows a trend.
```

### Tabla 5 — `table_05_mlp_robustness` (MLP, 30 réplicas, hiperparámetros fijos)
- **Primero y segundo puesto:**
  - IID: FedAvg 0,0199 ± 0,0116 y Centralized 0,0199 ± 0,0115 (empate).
  - Covariables: Centralized 0,0197 ± 0,0088 y SCAFFOLD 0,0197 ± 0,0088.
  - *Prior*: Centralized-FE 0,0223 ± 0,0117 y FedRep 0,0533 ± 0,0202.
  - Concepto: Local 0,1411 ± 0,0350 y FedRep 0,1546 ± 0,0418.
- **Comparación MLP frente a GLM (mismo régimen):** bajo IID y covariables, el MLP sin ajustar tiene un ED entre 2,5 veces mayor (Local, covariables: +147,8 %) y 6,6 veces mayor (FedRep, IID: +557,8 %); Centralized bajo IID: +289 %. Es exceso de parámetros frente a un DGP log-lineal. Bajo *prior* y concepto, los regímenes globales cambian poco (+1,8 % a +4,2 %), porque domina el sesgo de especificación. La excepción es FedRep bajo concepto, que mejora (−92,4 %): su cabeza local con 17 parámetros sí captura pendientes propias.
- **Conclusión:** el patrón cualitativo del GLM se reproduce. Penalización federada < 0,7 %; la personalización completa resuelve el concepto. La personalización excesiva (cabeza local completa) es peor que el mínimo necesario (intercepto) bajo *prior shift*.

```latex
Table~\ref{tab:mlp} replicates the analysis with a federated one-hidden-layer MLP. The federated penalty
remains negligible ($|\Delta|\leq0.7\%$ in every scenario), and personalization is again decisive under
concept shift (local 0.141, FedRep 0.155 vs.\ 2.16 for FedAvg). However, a fully local output head is
139\% worse than client fixed effects under prior shift (0.053 vs.\ 0.022), showing that personalizing
more parameters than the shift requires is costly with roughly 440 records per client.
```

### Tabla S1 — `table_S1_apriori_configuration` (sin tuning)
- En Ditto, el tuning reduce el ED entre un 20,5 % (*prior*) y un 74,8 % (IID: 0,0207 con λ = 1 fijo frente a 0,0052 con tuning).
- **Hallazgo relevante:** el tuning **empeora** a los métodos globales bajo IID (+10,9 % FedAvg, +12,8 % Centralized) y bajo covariables (+18,0 % a +20,4 %) frente a la configuración a priori con L2 = 10⁻⁴. La validación interna (el último 25 % del entrenamiento, unas 110 filas por cliente) elige L2 = 1 en la mayoría de réplicas con covariables desplazadas, y sobrerregulariza.
- Las conclusiones cualitativas no cambian, pero el tuning por validación interna no es inocuo con tamaños muestrales pequeños.

### Tabla S2 — `table_S2_grid_extension` (grillas ampliadas)
- Bajo *prior* y concepto, el cambio máximo de ED es 0,78 % / 0,51 % en los métodos globales y 5,79 % / 0,84 % en los personalizados.
- Bajo IID y covariables, la grilla ancha empeora hasta un +35,4 % (FedProx, IID) y un +52,5 % (FedProx, covariables): con más candidatos, más varianza de selección.
- Ningún cambio altera la separación entre métodos globales y personalizados.

---

## 3. Análisis de las figuras (`scripts/figures/`)

### Fig. 1 — `fig_01_data_distribution_or_manifold`
- **Patrones:**
  - (a) Bajo IID los centroides de los 5 clientes se superponen en el origen del PCA.
  - (b) Con *covariate shift* h = 1 se separan hasta ≈ 0,6–0,8 unidades de PC; el desplazamiento real es de 1,02 DE en el espacio de 10 dimensiones (`heterogeneity_magnitudes.json`) y el PCA de 2 componentes solo proyecta una parte.
  - (c) Con *prior shift*, las medianas de admisiones pasan de unas 10 en todos los clientes a entre ~6 (Central) y ~16 (South); razón máx./mín. de tasa basal 2,66.
  - (d) Las pendientes verdaderas de PM₂.₅ se abren en abanico desde 0,08; con h = 1 cubren aproximadamente [−0,21; 0,34] y el 30,8 % de los clientes tiene el signo invertido.
- **Conexión con las tablas:** cuantifica la magnitud de los tres mecanismos que explican el orden de la Tabla 2.
```latex
\caption{Client data under the semi-synthetic heterogeneity protocol. (a,b)~PCA projection of the
reference-standardised training covariates of the five regional clients in the IID control and under
covariate shift ($h=1$); large markers are client centroids. (c)~Simulated daily respiratory admissions per
client in the control (white) and under prior shift ($h=1$, coloured), spanning a 2.66-fold range of
baseline rates. (d)~True client-specific PM$_{2.5}$ log rate ratio per SD versus the concept-shift level
(10 replicates; dashed: global effect, 0.08); at $h=1$, 30.8\% of clients have a reversed effect sign.}
```

### Fig. 2 — `fig_02_main_performance_comparison`
- **Patrones:**
  - (a) Bajo covariables, todas las curvas son planas y cubren menos de una década (≈ 0,004–0,009), salvo Local (≈ 0,025); las bandas de los métodos globales se solapan por completo.
  - (b, c) Bajo *prior* y concepto, los métodos globales crecen unas 2,5 décadas en escala logarítmica entre h = 0 y h = 1, con el punto de inflexión ya en h = 0,25 (ED ≈ 0,09 y 0,12). FedRep y Centralized-FE quedan planos en ≈ 0,008 bajo *prior*. Ditto y Local quedan en ≈ 0,025–0,032 bajo *prior* y concepto.
  - (d) La brecha FedAvg − Centralized está en torno a 0 hasta h = 0,75. En h = 1 sube a ~0,007 (*prior*) y ~0,014 (concepto), con bandas amplias.
- **Conexión con las tablas:** visualiza H1b y la Tabla 2. La brecha federada de (d) es tres órdenes de magnitud menor que la escala de (b, c).
```latex
\caption{Excess test Poisson deviance per observation (log scale) versus the heterogeneity level $h$ for
(a)~covariate, (b)~prior and (c)~concept shift, for eight tuned training regimes. (d)~Federated penalty
(FedAvg minus Centralized). Lines: mean of 50 replicates; bands: 95\% bootstrap CI. Centralized and SCAFFOLD
coincide; Centralized-FE and FedRep nearly coincide. Global models degrade by about 2.5 orders of
magnitude under prior and concept shift, while the federated penalty stays below 0.015.}
```

### Fig. 3 — `fig_03_ablation_and_sensitivity`
- **Patrones:**
  - (a) La brecha de FedAvg frente a Centralized crece casi linealmente con E: de 0 (E = 1) a 1,36 × 10⁻³ (E = 20). FedProx la reduce algo (1,13 × 10⁻³) y SCAFFOLD la mantiene exactamente en 0.
  - (b) Hay un cruce en λ ≈ 1: hacia λ = 100, IID y covariables bajan (0,0046 / 0,0063) mientras *prior* y concepto suben hasta 1,04 / 1,69.
  - (c) DP: ε = 1 → 0,039, ε = 3 → 0,013, ε = 8 → 0,0098, sin DP → 0,0045 (IID). El descenso es casi log-lineal entre ε = 1 y 3.
- **Conexión con las tablas:** (a) muestra que la deriva existe pero es pequeña (explica H1a); (b) explica por qué el λ ajustado convierte a Ditto en el régimen más robusto (Tabla 2 frente a S1).
```latex
\caption{Sensitivity analyses under the a-priori configuration (50 replicates; mean and 95\% bootstrap CI).
(a)~Client drift: gap to the centralized optimum versus local steps $E$ (covariate shift, $h=1$); SCAFFOLD
removes it exactly. (b)~Ditto coupling $\lambda$ in the IID control and under each shift type at $h=1$: strong coupling is optimal when the conditional
distribution is shared and weak coupling when it differs. (c)~Record-level DP-FedAvg (DP-GD, clipping norm 5,
50 rounds, $E=1$, RDP accounting, $\delta=10^{-5}$) versus the privacy budget $\varepsilon$.}
```

### Fig. 4 — `fig_04_explainability_shap_or_latent`
- **Patrones:**
  - (a) Bajo concepto h = 1, las estimaciones de Ditto y Local caen sobre la diagonal identidad (r = 0,992, error absoluto medio 0,014), mientras que FedAvg forma una banda horizontal en ≈ 0,08: estima el efecto medio y no el de cada cliente (error 0,114).
  - (b) El RMSE de las pendientes de los modelos globales crece de ~0,007 a 0,13 con h; Ditto y Local se mantienen en ~0,015.
  - (c) Bajo *prior*, el RMSE de las pendientes de Centralized y SCAFFOLD sube de 0,007 a 0,012; FedRep y Centralized-FE quedan en ~0,007, y Ditto y Local en ~0,015 (varianza local).
- **Conexión con las tablas:** muestra el mecanismo que la Tabla 2 solo cuantifica. Bajo concepto el error es de **pendientes**; bajo *prior*, de **intercepto** (las pendientes globales casi no se degradan).
```latex
\caption{Coefficient recovery against the known data-generating process. (a)~Estimated versus true
client-specific PM$_{2.5}$ log rate ratio under concept shift ($h=1$; 250 client-replicates; dashed line:
identity; grey line: global effect). (b,c)~Root-mean-square error of the ten slope estimates versus $h$
under concept and prior shift (mean and 95\% bootstrap CI over 50 replicates).}
```

### Fig. 5 — `fig_05_error_calibration_or_case_study`
- **Patrones:**
  - (a) Bajo concepto h = 1, los 6 métodos globales o semipersonalizados tienen ED por cliente entre 0,2 y 6 (peor cliente de FedAvg: 3,65 de media), mientras que Local y Ditto llegan como máximo a ≈ 0,1 (peor cliente medio: 0,071; `fairness.csv`). La equidad entre clientes sigue el mismo orden que la precisión.
  - (b) La calibración agregada por deciles parece buena para todos. FedAvg se desvía ligeramente por encima de la diagonal en predicciones de 8–13 (error medio por decil 0,349, frente a 0,059 de FedRep y 0,065 de Local).
  - (c) El residuo de Pearson de FedAvg crece de forma monótona con la tasa basal del cliente (−1,31; −0,78; −0,10; +0,79; +1,91), mientras que FedRep queda en |·| ≤ 0,02.
- **Conexión:** la calibración agregada **oculta** el sesgo sistemático por cliente que explica el ED de 1,285 de la Tabla 2.
```latex
\caption{Error analysis on the local test sets. (a)~Client-level excess deviance under concept shift
($h=1$; 250 client-replicates per method; log scale). (b)~Calibration under prior shift ($h=1$): mean observed
count versus mean predicted rate by decile, pooled over clients and replicates. (c)~Mean Pearson residual
per client, with clients ordered by their true baseline rate within each replicate (prior shift, $h=1$;
95\% bootstrap CI). Pooled calibration conceals the systematic client-level bias of the global model.}
```

### Fig. 6 — `fig_06_critical_difference`
- **Patrones:**
  - (a) En el global, Ditto (3,03) aparece aislado como mejor; su intervalo no se solapa con ningún otro (CD = 0,41).
  - (b) Bajo covariables, Centralized, FedProx, SCAFFOLD y Ditto forman un grupo indistinguible (CD = 1,48) y Local queda último.
  - (c) Bajo *prior*, Centralized-FE y FedRep encabezan, seguidos de Local y Ditto.
  - (d) Bajo concepto, Ditto y Local encabezan y FedAvg queda último.
- **Conexión:** visualiza la Tabla 4 (Friedman/Nemenyi): **no hay un ganador universal**, salvo Ditto con tuning en el agregado.
```latex
\caption{Average Friedman ranks (lower is better) with Nemenyi intervals ($\pm$CD/2, $\alpha=0.05$); methods
with non-overlapping intervals differ significantly. (a)~All 650 unique scenarios (CD = 0.41); (b--d)~covariate,
prior and concept shift at $h=1$ (50 replicates, CD = 1.48). Shading marks the interval of the best method.}
```

### Fig. 7 — `fig_07_mlp_robustness`
- **Patrones:** bajo IID y covariables, los cuatro regímenes globales del MLP se alinean en ≈ 0,02, y Local y FedRep en ≈ 0,05–0,06. Bajo *prior*, Centralized-FE (≈ 0,022) queda claramente por debajo de FedRep (≈ 0,053). Bajo concepto, los globales quedan en ≈ 2,15 y Local y FedRep en ≈ 0,14–0,15.
- **Conexión:** Tabla 5.
```latex
\caption{Federated MLP robustness study: excess deviance (log scale; mean and 95\% bootstrap CI over 30
replicates) of six training regimes in the IID control and at $h=1$ for each shift type. All regimes share
the same untuned hyper-parameters and initialisation.}
```

---

## 4. Ablación, sensibilidad y significancia estadística

- **Ablación (Tabla 3):** el componente decisivo depende del mecanismo: intercepto propio para el *prior shift* (−99,3 %) y pendientes propias para el concepto (−98,4 %). Los componentes contra la deriva aportan como máximo −0,0154 (FedProx bajo covariables, −20,3 % relativo pero 0,0015 absoluto).
- **Sensibilidad:** la deriva crece con E (Fig. 3a). El λ de Ditto define un compromiso nítido (Fig. 3b). DP con ε = 1 multiplica el ED por unas 8,7 (0,0392 frente a 0,0045; `sens_dp.csv`). Ampliar la grilla no cambia conclusiones (S2). El tuning interno sobrerregulariza a los globales con covariables desplazadas (S1).
- **Veredicto estadístico:**
  - **Equivalencia demostrada:** FedAvg ≡ Centralized ≡ SCAFFOLD ≡ FedProx bajo IID (TOST).
  - **Superioridad demostrada:** las 12 comparaciones frente a FedAvg bajo *prior* y concepto son significativas (p_Holm ≤ 3,6 × 10⁻²). La personalización (FedRep, Ditto, Local, Centralized-FE) llega a p_Holm = 3,2 × 10⁻¹⁴ y r = −1,00 en 6 de ellas. Las mejoras de FedProx y SCAFFOLD son significativas pero ≤ 0,7 % de la pérdida.
  - **Inferioridad demostrada:** Local frente a FedAvg bajo covariables (r = +1,00).
  - **Empate estadístico:** SCAFFOLD, FedRep, Ditto y Centralized-FE frente a FedAvg bajo covariables.
  - Toda la inferencia aplica Holm por familia; el post-hoc de Friedman usa Nemenyi.

## 5. Interpretabilidad, mecanismos y anatomía del error

- **Mecanismo:** como el DGP es conocido, la "explicación" del modelo es la recuperación de coeficientes (Fig. 4), que es más rigurosa que SHAP aquí.
  - Bajo concepto, los modelos globales estiman el efecto **promedio** de PM₂.₅ (≈ 0,08) para todos los clientes, aunque en el 30,8 % de ellos el efecto real tiene signo opuesto. Es un riesgo de **conclusión ecológica errónea**: un análisis federado global diría "PM₂.₅ aumenta los ingresos" en regiones donde, por construcción, no lo hace.
  - Bajo *prior*, las pendientes globales son casi correctas (RMSE 0,0084 en FedAvg) y el error es de intercepto.
- **Anatomía del error:**
  - Los errores de los modelos globales se concentran en los clientes extremos: residuo +1,91 (sesgo de subpredicción) en el de mayor tasa basal y −1,31 (sobrepredicción) en el de menor. Esto es un **problema de equidad**: el modelo global falla precisamente en las regiones de mayor demanda hospitalaria.
  - Bajo concepto, el peor cliente de FedAvg tiene un ED medio de 3,65, frente a 0,071 en Ditto.
  - Local paga varianza: con unas 440 filas, su ED bajo IID (0,0232) es 4,5 veces el de FedAvg.
- **Sin SHAP:** no se calculó SHAP ni importancia por permutación. Es deliberado: con un GLM de coeficientes verdaderos conocidos, la recuperación de coeficientes es la medida directa. No disponible en los artefactos, y no es necesario.

## 6. Insumos para la *Discussion*

**Por qué funciona cada régimen (hipótesis respaldadas por la evidencia):**
- Con un modelo bien especificado y participación completa, FedAvg converge cerca del óptimo agrupado. La deriva por pasos locales escala con E y es de segundo orden (Fig. 3a).
- La pérdida bajo heterogeneidad es **sesgo de especificación**: la familia de modelos globales no contiene la función verdadera de cada cliente, y ningún algoritmo de optimización federada lo corrige, ni siquiera agrupando los datos.
- La personalización corrige el sesgo a cambio de varianza. Ditto es robusto porque elige el compromiso con datos de validación.

**Limitaciones (anticipar a un revisor Q1):**
1. 5 clientes *cross-silo* con unas 440 filas, participación completa y sin desbalance de tamaño.
2. Covariables de nulo verificado, sin la estructura espacio-temporal real; no hay validez externa en redes hospitalarias reales.
3. Heterogeneidad sintética de magnitud fijada a priori, sin mezclas de tipos.
4. Inferencia condicional al conjunto de covariables; tests locales pequeños (147–175 filas).
5. El tuning por validación interna sobrerregulariza a los métodos globales con *covariate shift* (S1) y depende del ancho de la grilla bajo IID (S2).
6. MLP sin ajustar; DP solo a nivel de registro.
7. Lenguaje: los efectos son de un DGP controlado. Hablar de "atribuible al mecanismo inyectado", no de causalidad en datos reales.

**Auditoría de calidad de archivos:**
- Las 7 tablas `.tex` compilan con `pdflatex` (MiKTeX) en un documento mínimo con `booktabs`, `amsmath` y `graphicx`: 0 errores y 2 avisos *Overfull hbox* de 0,12 pt y 0,11 pt, atribuibles al `\resizebox` del documento de prueba.
- Todos los valores de cada `.csv` aparecen en su `.tex`.
- Las 7 figuras están en `.pdf` y `.png` a 600 DPI y sin títulos internos (`review/audit_round_03.json`).

---

## Nota técnica final (defectos e inconsistencias)

| # | Archivo | Defecto detectado | Estado |
|---|---|---|---|
| 1 | `figures/fig_05_error_calibration_or_case_study` | La etiqueta "(a)" se solapaba con el rótulo del eje y | **Corregido**: el rótulo se acortó a "Excess deviance (per obs.)" y la figura se regeneró |
| 2 | `figures/fig_03_ablation_and_sensitivity` (b) | La leyenda cubría la curva de concepto en λ = 10 | **Corregido**: leyenda arriba a la izquierda con etiquetas cortas (h = 1 indicado en el caption) |
| 3 | `figures/fig_01_data_distribution_or_manifold` (b) | La leyenda cubría puntos dispersos | **Corregido**: leyenda fuera de los ejes |
| 4 | `scripts/experiment_manifest_and_captions.md` §1.3 | "≤ 0,84 %" para los globales (el 0,84 % era de FedRep) | **Corregido**: ≤ 0,78 % (*prior*) y ≤ 0,51 % (concepto) |
| 5 | `review/CERTIFICATE_OF_PUBLISHABILITY.md`, manifiesto | No declaraban que el tuning empeora a los métodos globales bajo IID y covariables (+11 % a +20 % frente a S1) | **Corregido**: limitación 7 del certificado y limitación (viii) del manifiesto |
| 6 | — | SHAP / importancia por permutación: no disponible en los artefactos experimentales | Se mantiene a propósito (se usa la recuperación de coeficientes frente al DGP conocido); declararlo en Métodos |

Verificación posterior a las correcciones: solo se regeneraron las figs. 1, 3 y 5 (pasos deterministas 01, 03 y 05). Las 7 tablas `.csv`/`.tex` son idénticas byte a byte a las de la ronda 3, y el auditor no encuentra títulos ni fuentes < 8 pt, todas las figuras están a 600 DPI y no hay patrones de fuga (`review/audit_round_04.json`). Ninguna cifra de este reporte cambia.
