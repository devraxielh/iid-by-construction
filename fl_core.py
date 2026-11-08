"""fl_core.py — semi-synthetic data-generating process and federated Poisson-GLM solvers.

Shared library imported by scripts 02, 03 and 05 (they exchange *results* only via
files in results/; this module holds code, not state).

All solvers optimise the same objective,
    F(w) = sum_k (n_k / n) * f_k(w),  f_k(w) = mean Poisson NLL on client k + L2/2 ||w_slopes||^2,
on covariates standardised with a scaler fitted on the pooled *training* data only
(equivalent to a federated aggregation of first/second moments).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

U = __import__("importlib").import_module("00_config_and_utils")


# ---------------------------------------------------------------------------
# Data-generating process
# ---------------------------------------------------------------------------
@dataclass
class ClientData:
    name: str
    X_tr: np.ndarray            # raw covariates (train)
    X_te: np.ndarray
    lam_tr: np.ndarray          # true Poisson rate
    lam_te: np.ndarray
    y_tr: np.ndarray
    y_te: np.ndarray
    beta_true: np.ndarray       # true client slopes on the generator-reference scale
    b0_true: float              # true client intercept


@dataclass
class Scenario:
    shift: str
    level: float
    replicate: int
    clients: list[ClientData] = field(default_factory=list)


def ref_standardise(X: np.ndarray) -> np.ndarray:
    mu = np.array([U.GEN_MOMENTS[f][0] for f in U.FEATURES])
    sd = np.array([U.GEN_MOMENTS[f][1] for f in U.FEATURES])
    return (X - mu) / sd


def replicate_parameters(r: int, k: int, p: int) -> dict:
    """Per-replicate random structure, shared by every shift type and level
    (common random numbers: h = 0 is the identical IID control for all types)."""
    rng = np.random.default_rng(U.SEED * 1000 + r)
    beta = rng.normal(0.0, U.OTHER_EFFECT_SD, p)
    beta[U.FEATURES.index(U.FOCAL_EXPOSURE)] = U.FOCAL_EFFECT
    offsets = np.linspace(-1.0, 1.0, k) * np.sqrt(3.0 / 2.0)    # zero-mean; values 0, +-0.61, +-1.22 (variance 0.75 for k = 5)
    offsets = rng.permutation(offsets)
    D = rng.normal(0.0, 1.0, (k, p))
    D -= D.mean(axis=0, keepdims=True)                            # average slope = beta
    dirs = rng.normal(0.0, 1.0, (k, p))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return {"beta": beta, "offsets": offsets, "D": D, "dirs": dirs,
            "resample_seed": int(rng.integers(2**31)), "y_seed": int(rng.integers(2**31))}


def make_scenario(base, shift: str, level: float, r: int) -> Scenario:
    """base: dict region -> (X_train_pool, X_test_pool) raw covariate arrays."""
    k, p = len(U.REGIONS), len(U.FEATURES)
    par = replicate_parameters(r, k, p)
    rs = np.random.default_rng(par["resample_seed"])
    ry = np.random.default_rng(par["y_seed"])
    sc = Scenario(shift, level, r)
    for j, reg in enumerate(U.REGIONS):
        Xtr_pool, Xte_pool = base[reg]
        tilt = level * U.COVARIATE_TILT if shift == "covariate" else 0.0
        out = []
        for pool in (Xtr_pool, Xte_pool):
            s = ref_standardise(pool) @ par["dirs"][j]
            w = np.exp(tilt * s)
            w /= w.sum()
            # sorted indices keep the chronological order of the pool (inner validation split)
            out.append(pool[np.sort(rs.choice(len(pool), size=len(pool), replace=True, p=w))])
        Xtr, Xte = out
        b0 = U.BASE_LOG_RATE + (level * U.PRIOR_SHIFT_SCALE * par["offsets"][j] if shift == "prior" else 0.0)
        beta_k = par["beta"] + (level * U.CONCEPT_SHIFT_SD * par["D"][j] if shift == "concept" else 0.0)
        lam_tr = np.exp(b0 + ref_standardise(Xtr) @ beta_k)
        lam_te = np.exp(b0 + ref_standardise(Xte) @ beta_k)
        sc.clients.append(ClientData(reg, Xtr, Xte, lam_tr, lam_te,
                                     ry.poisson(lam_tr).astype(float), ry.poisson(lam_te).astype(float),
                                     beta_k, b0))
    return sc


# ---------------------------------------------------------------------------
# Poisson GLM primitives
# ---------------------------------------------------------------------------
def design(X: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), (X - mean) / sd])


def grad(w, A, y, l2=U.L2):
    eta = np.clip(A @ w, -20, 20)
    g = A.T @ (np.exp(eta) - y) / len(y)
    g[1:] += l2 * w[1:]
    return g


def newton(A, y, w0=None, l2=U.L2, iters=50, tol=1e-10, n_unpen=1):
    """Penalised Poisson MLE; the first `n_unpen` columns (intercepts) are not penalised."""
    w = np.zeros(A.shape[1]) if w0 is None else w0.copy()
    if w0 is None:
        w[:n_unpen] = np.log(max(y.mean(), 1e-3))
    R = np.eye(A.shape[1]) * l2
    R[:n_unpen, :n_unpen] = 0.0
    n = len(y)
    for _ in range(iters):
        mu = np.exp(np.clip(A @ w, -20, 20))
        g = A.T @ (mu - y) / n + R @ w
        H = (A * mu[:, None]).T @ A / n + R
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < tol:
            break
    return w


def centralized_fe(As, ys, l2=U.L2):
    """Pooled GLM with client fixed effects (one unpenalised intercept per client, shared slopes):
    the centralized analogue of FedRep. Returns per-client coefficient vectors [b_k, slopes]."""
    K = len(As)
    blocks = []
    for j, A in enumerate(As):
        D = np.zeros((len(A), K))
        D[:, j] = 1.0
        blocks.append(np.column_stack([D, A[:, 1:]]))
    w = newton(np.vstack(blocks), np.concatenate(ys), l2=l2, n_unpen=K)
    return [np.concatenate([[w[j]], w[K:]]) for j in range(K)]


def mean_deviance(y, mu):
    mu = np.clip(mu, 1e-12, None)
    ylogy = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
    return float(np.mean(2.0 * (ylogy - (y - mu))))


# ---------------------------------------------------------------------------
# Federated solvers (full participation, full-batch local GD)
# ---------------------------------------------------------------------------
def _init(As, ys):
    w = np.zeros(As[0].shape[1])
    n = np.array([len(y) for y in ys], float)
    w[0] = np.log(sum(y.sum() for y in ys) / n.sum())
    return w, n / n.sum()


def fedavg(As, ys, rounds=U.FL_ROUNDS, E=U.LOCAL_STEPS, lr=U.LR, mu_prox=0.0, dp=None, rng=None, l2=U.L2):
    """FedAvg; mu_prox > 0 gives FedProx (proximal term applied as an implicit step, stable for
    any mu). dp = dict(clip, sigma) enables record-level DP-GD inside each client."""
    w, pk = _init(As, ys)
    for _ in range(rounds):
        new = []
        for A, y in zip(As, ys):
            wk = w.copy()
            for _ in range(E):
                g = grad(wk, A, y, l2) if dp is None else _dp_grad(wk, A, y, dp, rng, l2)
                wk = (wk - lr * g + lr * mu_prox * w) / (1.0 + lr * mu_prox)
            new.append(wk)
        w = np.sum([p * wk for p, wk in zip(pk, new)], axis=0)
    return w


def _dp_grad(w, A, y, dp, rng, l2=U.L2):
    eta = np.clip(A @ w, -20, 20)
    G = A * (np.exp(eta) - y)[:, None]                       # per-example gradients
    norms = np.linalg.norm(G, axis=1, keepdims=True)
    G = G / np.maximum(1.0, norms / dp["clip"])
    g = (G.sum(axis=0) + rng.normal(0.0, dp["sigma"] * dp["clip"], G.shape[1])) / len(y)
    g[1:] += l2 * w[1:]
    return g


def scaffold(As, ys, rounds=U.FL_ROUNDS, E=U.LOCAL_STEPS, lr=U.LR, l2=U.L2):
    w, pk = _init(As, ys)
    c = np.zeros_like(w)
    ck = [np.zeros_like(w) for _ in As]
    for _ in range(rounds):
        dw, dc = np.zeros_like(w), np.zeros_like(w)
        for j, (A, y) in enumerate(zip(As, ys)):
            wk = w.copy()
            for _ in range(E):
                wk -= lr * (grad(wk, A, y, l2) - ck[j] + c)
            c_new = ck[j] - c + (w - wk) / (E * lr)              # option II
            dw += pk[j] * (wk - w)
            dc += pk[j] * (c_new - ck[j])
            ck[j] = c_new
        w = w + dw
        c = c + dc
    return w


def ditto(As, ys, lam=U.DITTO_LAMBDA, rounds=U.FL_ROUNDS, E=U.LOCAL_STEPS, lr=U.LR, l2=U.L2):
    """Returns (global w, list of personal v_k)."""
    w, pk = _init(As, ys)
    vs = [w.copy() for _ in As]
    for _ in range(rounds):
        new = []
        for j, (A, y) in enumerate(zip(As, ys)):
            wk = w.copy()
            for _ in range(E):
                wk -= lr * grad(wk, A, y, l2)
                # proximal-gradient step: explicit on f_k, implicit on (lam/2)||v - w||^2
                vs[j] = (vs[j] - lr * grad(vs[j], A, y, l2) + lr * lam * w) / (1.0 + lr * lam)
            new.append(wk)
        w = np.sum([p * wk for p, wk in zip(pk, new)], axis=0)
    return w, vs


def fedrep(As, ys, rounds=U.FL_ROUNDS, E=U.LOCAL_STEPS, lr=U.LR, l2=U.L2):
    """Partial personalisation: shared slopes (body), local intercept (head)."""
    w, pk = _init(As, ys)
    heads = [w[0] for _ in As]
    body = w[1:].copy()
    for _ in range(rounds):
        bodies = []
        for j, (A, y) in enumerate(zip(As, ys)):
            wk = np.concatenate([[heads[j]], body])
            for _ in range(E):                                    # head update, body frozen
                wk[0] -= lr * grad(wk, A, y, l2)[0]
            for _ in range(E):                                    # body update, head frozen
                wk[1:] -= lr * grad(wk, A, y, l2)[1:]
            heads[j] = wk[0]
            bodies.append(wk[1:])
        body = np.sum([p * b for p, b in zip(pk, bodies)], axis=0)
    return [np.concatenate([[h], body]) for h in heads]


# ---------------------------------------------------------------------------
# Run every method on one scenario (optionally with symmetric inner-validation tuning)
# ---------------------------------------------------------------------------
HP_NAME = {"Centralized": "l2", "Centralized-FE": "l2", "Local": "l2", "FedAvg": "l2", "SCAFFOLD": "l2",
           "FedRep": "l2", "FedProx": "mu", "Ditto": "lambda"}


def _fit_method(m, As, ys, hp, E, lr):
    """Fit one method with hyper-parameter value hp; returns per-client coefficient vectors."""
    K = len(As)
    l2 = hp if HP_NAME[m] == "l2" else U.L2
    if m == "Centralized":
        return [newton(np.vstack(As), np.concatenate(ys), l2=l2)] * K
    if m == "Centralized-FE":
        return centralized_fe(As, ys, l2=l2)
    if m == "Local":
        return [newton(A, y, l2=l2) for A, y in zip(As, ys)]
    if m == "FedAvg":
        return [fedavg(As, ys, E=E, lr=lr, l2=l2)] * K
    if m == "FedProx":
        return [fedavg(As, ys, E=E, lr=lr, mu_prox=hp)] * K
    if m == "SCAFFOLD":
        return [scaffold(As, ys, E=E, lr=lr, l2=l2)] * K
    if m == "FedRep":
        return fedrep(As, ys, E=E, lr=lr, l2=l2)
    if m == "Ditto":
        return ditto(As, ys, lam=hp, E=E, lr=lr)[1]
    raise ValueError(m)


def _default_hp(m, ditto_lam, fedprox_mu):
    return {"mu": fedprox_mu, "lambda": ditto_lam}.get(HP_NAME[m], U.L2)


def _grid(m, grids=None):
    g = grids or {"l2": U.L2_GRID, "mu": U.MU_GRID, "lambda": U.LAMBDA_GRID}
    return g[HP_NAME[m]]


def fit_all(sc: Scenario, methods=U.METHODS, ditto_lam=U.DITTO_LAMBDA, E=U.LOCAL_STEPS,
            fedprox_mu=U.FEDPROX_MU, tune=False, grids=None):
    """Returns (dict method -> per-client coefficient vectors, scaler, dict method -> selected hp).

    tune=True: every method gets the same budget (|grid| = 5 by default; `grids` overrides it, see
    07_grid_extension.py) for its hyper-parameter, selected by
    the sample-weighted federated validation deviance on an inner chronological split (last
    U.VAL_FRACTION of each client's *training* rows). The selected value is refitted on the full
    training data. Test blocks are never touched.
    """
    Xpool = np.vstack([c.X_tr for c in sc.clients])
    mean, sd = Xpool.mean(axis=0), Xpool.std(axis=0, ddof=0)
    sd[sd == 0] = 1.0
    As = [design(c.X_tr, mean, sd) for c in sc.clients]
    ys = [c.y_tr for c in sc.clients]
    l2max = max(_grid("Local", grids))
    lr = safe_lr(As, [newton(A, y) for A, y in zip(As, ys)], l2max=l2max)
    chosen = {}
    if tune:
        cut = [int(round(len(y) * (1 - U.VAL_FRACTION))) for y in ys]
        Ai, yi = [A[:c] for A, c in zip(As, cut)], [y[:c] for y, c in zip(ys, cut)]
        Av, yv = [A[c:] for A, c in zip(As, cut)], [y[c:] for y, c in zip(ys, cut)]
        nv = np.array([len(y) for y in yv], float)
        lr_in = safe_lr(Ai, [newton(A, y) for A, y in zip(Ai, yi)], l2max=l2max)
        for m in methods:
            scores = []
            for hp in _grid(m, grids):
                ws = _fit_method(m, Ai, yi, hp, E, lr_in)
                dv = [mean_deviance(y, np.exp(np.clip(A @ w, -20, 20))) for A, y, w in zip(Av, yv, ws)]
                scores.append(float(np.sum(nv * dv) / nv.sum()))
            chosen[m] = _grid(m, grids)[int(np.argmin(scores))]
    else:
        chosen = {m: _default_hp(m, ditto_lam, fedprox_mu) for m in methods}
    out = {m: _fit_method(m, As, ys, chosen[m], E, lr) for m in methods}
    return out, (mean, sd), chosen


def local_smoothness(A, w, l2=U.L2) -> float:
    mu = np.exp(np.clip(A @ w, -20, 20))
    return float(np.linalg.eigvalsh((A * mu[:, None]).T @ A / len(mu)).max() + l2)


def safe_lr(As, local_w, base=U.LR, safety=0.9, l2max=None) -> float:
    """Shared step size: min(base, safety / max_k L_k), with L_k the client curvature at its
    local optimum (computed client-side; only the scalar max is sent to the server).
    Uses the largest L2 of the grid so the step is stable for every candidate."""
    l2max = max(U.L2_GRID) if l2max is None else l2max
    L = max(local_smoothness(A, w, l2=l2max) for A, w in zip(As, local_w))
    return float(min(base, safety / L))


def to_reference_slopes(w: np.ndarray, sd_fit: np.ndarray) -> np.ndarray:
    """Convert fitted slopes (per fitted-SD) to the generator-reference scale (per population SD)."""
    sd_ref = np.array([U.GEN_MOMENTS[f][1] for f in U.FEATURES])
    return w[1:] * sd_ref / sd_fit


def evaluate(sc: Scenario, fits: dict, scaler) -> list[dict]:
    mean, sd = scaler
    rows = []
    for m, ws in fits.items():
        for j, c in enumerate(sc.clients):
            A = design(c.X_te, mean, sd)
            mu = np.exp(np.clip(A @ ws[j], -20, 20))
            dev = mean_deviance(c.y_te, mu)
            dev_or = mean_deviance(c.y_te, c.lam_te)
            slopes = to_reference_slopes(ws[j], sd)
            rows.append({
                "shift": sc.shift, "level": sc.level, "replicate": sc.replicate, "method": m,
                "client": c.name, "n_train": len(c.y_tr), "n_test": len(c.y_te),
                "deviance": dev, "deviance_oracle": dev_or, "excess_deviance": dev - dev_or,
                "rmse": float(np.sqrt(np.mean((c.y_te - mu) ** 2))),
                "rmse_rate": float(np.sqrt(np.mean((c.lam_te - mu) ** 2))),
                "slope_rmse": float(np.sqrt(np.mean((slopes - c.beta_true) ** 2))),
                "pm25_slope_hat": float(slopes[U.FEATURES.index(U.FOCAL_EXPOSURE)]),
                "pm25_slope_true": float(c.beta_true[U.FEATURES.index(U.FOCAL_EXPOSURE)]),
            })
    return rows


def predictions(sc: Scenario, fits: dict, scaler) -> list[dict]:
    mean, sd = scaler
    rows = []
    for m, ws in fits.items():
        for j, c in enumerate(sc.clients):
            mu = np.exp(np.clip(design(c.X_te, mean, sd) @ ws[j], -20, 20))
            for yi, li, mi in zip(c.y_te, c.lam_te, mu):
                rows.append((sc.shift, sc.level, sc.replicate, m, c.name, yi, li, mi))
    return rows


# ---------------------------------------------------------------------------
# Differential privacy accounting (Gaussian mechanism, full batch, RDP composition)
# ---------------------------------------------------------------------------
def rdp_epsilon(sigma: float, steps: int, delta: float) -> float:
    alphas = np.concatenate([np.linspace(1.01, 10, 200), np.arange(11, 512)])
    eps = steps * alphas / (2 * sigma ** 2) + np.log(1 / delta) / (alphas - 1)
    return float(eps.min())


def sigma_for_epsilon(eps: float, steps: int, delta: float) -> float:
    lo, hi = 1e-2, 1e5
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if rdp_epsilon(mid, steps, delta) > eps:
            lo = mid
        else:
            hi = mid
    return hi
