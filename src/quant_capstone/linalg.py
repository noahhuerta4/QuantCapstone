"""Small linear-algebra primitives shared by the signal and risk stages.

Neutralization and signal orthogonalization are the same operation — remove the
component of a vector that lies in the span of some loadings — so it lives once,
here, and gets tested once.
"""

from __future__ import annotations

import numpy as np


def project_out(x: np.ndarray, loadings: np.ndarray | None) -> np.ndarray:
    """Residual of ``x`` after regressing it on the columns of ``loadings``.

    Uses ``lstsq`` rather than a normal-equation inverse because the loading
    matrix is routinely rank-deficient: a sector with one member is collinear
    with the intercept, and an empty sector column is all zeros.

    Guarantees ``loadings.T @ result == 0`` up to numerical precision, which is
    exactly the neutrality constraint ``sum(w) = 0``, ``sum(w*beta) = 0``, etc.
    """
    if loadings is None or loadings.size == 0:
        return x
    finite = np.isfinite(x) & np.isfinite(loadings).all(axis=1)
    out = np.zeros_like(x, dtype=float)
    if finite.sum() == 0:
        return out
    xf, lf = x[finite], loadings[finite]
    coef, *_ = np.linalg.lstsq(lf, xf, rcond=None)
    out[finite] = xf - lf @ coef
    return out


def build_loadings(*blocks: np.ndarray | None) -> np.ndarray | None:
    """Horizontally stack constraint blocks, dropping empties and zero columns."""
    cols = [b.reshape(len(b), -1) for b in blocks if b is not None and b.size]
    if not cols:
        return None
    matrix = np.hstack(cols)
    keep = np.abs(np.nan_to_num(matrix)).sum(axis=0) > 0
    matrix = matrix[:, keep]
    return matrix if matrix.shape[1] else None


def nan_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation over the pairwise-finite overlap."""
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    x, y = a[ok], b[ok]
    sx, sy = x.std(), y.std()
    if sx == 0 or sy == 0:
        return np.nan
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation; robust to the fat tails in daily cross-sections."""
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    from scipy.stats import rankdata

    return nan_corr(rankdata(a[ok]), rankdata(b[ok]))
