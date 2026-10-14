"""Statistics used by the quality checks: Wilson intervals and cross-validated ridge probes."""
from __future__ import annotations

import math

import numpy as np


def wilson_interval(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (draws may be counted as 0.5)."""
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def ridge_cv_r2(x: np.ndarray, y: np.ndarray, alpha: float = 1.0, folds: int = 5) -> float:
    """
    Mean out-of-fold R^2 of a ridge regression probe, like sklearn's
    ``cross_val_score(Ridge(alpha), StandardScaler().fit_transform(X), y, cv=5, scoring="r2")``.

    Folds are contiguous blocks (no shuffling), so when rows are ordered by game
    most games fall entirely inside one fold. Features are standardized on the
    full matrix first, matching the repo's probe scripts.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    std = x.std(axis=0)
    x = (x - x.mean(axis=0)) / np.where(std > 1e-12, std, 1.0)
    n = len(y)
    bounds = np.linspace(0, n, folds + 1).astype(int)
    scores = []
    for i in range(folds):
        test = np.zeros(n, dtype=bool)
        test[bounds[i] : bounds[i + 1]] = True
        xtr, ytr, xte, yte = x[~test], y[~test], x[test], y[test]
        x_mean, y_mean = xtr.mean(axis=0), ytr.mean()
        xc = xtr - x_mean
        w = np.linalg.solve(xc.T @ xc + alpha * np.eye(x.shape[1]), xc.T @ (ytr - y_mean))
        pred = (xte - x_mean) @ w + y_mean
        ss_res = float(((yte - pred) ** 2).sum())
        ss_tot = float(((yte - yte.mean()) ** 2).sum())
        scores.append(1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0)
    return float(np.mean(scores))
