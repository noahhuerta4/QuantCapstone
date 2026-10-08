"""Covariance estimation (spec S5.1).

Why not the sample covariance: with ~200 names and a 252-day window the ratio
T/N is barely 1.3, so the sample estimator's smallest eigenvalues are noise
driven toward zero. Any optimizer then piles into those directions because they
look riskless, and realized vol lands far above target. Shrinkage and factor
structure are both ways of refusing to believe the small eigenvalues.

Three estimators, swappable because the choice materially changes realized risk
and that comparison is itself a result:

``ledoit_wolf`` (default)
    Optimal linear shrinkage toward a scaled identity. Well conditioned by
    construction, no tuning.
``pca``
    K statistical factors plus diagonal residuals. Imposes the structure we
    believe in (a few common drivers) and is what the effective-breadth
    calculation in ``law/`` reuses.
``sample`` / ``diagonal``
    Baselines. ``sample`` exists to demonstrate the failure above; ``diagonal``
    ignores correlation entirely and is a surprisingly hard benchmark to beat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import ledoit_wolf

from ..config import RiskConfig

MIN_VALID_FRACTION = 0.6


def _clean_window(window: pd.DataFrame) -> tuple[np.ndarray, pd.Index, pd.Series]:
    """Split a return window into a dense block and per-name variances."""
    valid_fraction = window.notna().mean()
    dense_names = window.columns[valid_fraction >= MIN_VALID_FRACTION]
    block = window[dense_names]
    # Fill residual gaps with the name's own mean, i.e. zero deviation: a missing
    # bar should add no covariance information rather than a fake zero return.
    block = block.fillna(block.mean())
    variances = window.var(ddof=1).fillna(window.var(ddof=1).median())
    return block.to_numpy(dtype=float), dense_names, variances


def _expand(dense: np.ndarray, dense_names: pd.Index, tickers: pd.Index, variances: pd.Series) -> pd.DataFrame:
    """Embed a dense covariance block into the full ticker set.

    Names with too little history get their own variance and zero covariance --
    conservative for a neutral book, since zero covariance means the optimizer
    sees no free hedge it can over-trust.
    """
    full = pd.DataFrame(0.0, index=tickers, columns=tickers)
    full.loc[dense_names, dense_names] = dense
    missing = tickers.difference(dense_names)
    for name in missing:
        full.loc[name, name] = float(variances.get(name, variances.median()))
    return full


def sample_covariance(window: pd.DataFrame, tickers: pd.Index) -> pd.DataFrame:
    dense, names, var = _clean_window(window)
    return _expand(np.cov(dense, rowvar=False), names, tickers, var)


def diagonal_covariance(window: pd.DataFrame, tickers: pd.Index) -> pd.DataFrame:
    var = window.var(ddof=1).reindex(tickers)
    var = var.fillna(var.median()).fillna(1e-4)
    return pd.DataFrame(np.diag(var.to_numpy()), index=tickers, columns=tickers)


def ledoit_wolf_covariance(window: pd.DataFrame, tickers: pd.Index) -> pd.DataFrame:
    dense, names, var = _clean_window(window)
    if dense.shape[0] < 10 or dense.shape[1] < 2:
        return diagonal_covariance(window, tickers)
    cov, _shrinkage = ledoit_wolf(dense, assume_centered=False)
    return _expand(cov, names, tickers, var)


def pca_covariance(window: pd.DataFrame, tickers: pd.Index, n_factors: int = 10) -> pd.DataFrame:
    """K-factor statistical model: Sigma = B Lambda B' + diag(residual var)."""
    dense, names, var = _clean_window(window)
    if dense.shape[0] < 10 or dense.shape[1] < 2:
        return diagonal_covariance(window, tickers)

    centered = dense - dense.mean(axis=0)
    sample = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(sample)
    order = np.argsort(eigvals)[::-1]
    k = min(n_factors, len(eigvals))
    top_vals, top_vecs = np.clip(eigvals[order][:k], 0, None), eigvecs[:, order][:, :k]

    systematic = top_vecs @ np.diag(top_vals) @ top_vecs.T
    residual = np.clip(np.diag(sample) - np.diag(systematic), 1e-10, None)
    return _expand(systematic + np.diag(residual), names, tickers, var)


class CovarianceEstimator:
    """Estimates Sigma on a refresh schedule and caches the result.

    Daily re-estimation on a 252-day window is ~4% new data per day: almost pure
    computation for almost no information. Monthly refresh is the default.
    """

    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg
        self._cache: pd.DataFrame | None = None
        self._matrix: np.ndarray | None = None
        self._positions: np.ndarray | None = None
        self._cached_at: int = -10**9

    def refresh_if_stale(
        self, returns: pd.DataFrame, day_index: int, active: np.ndarray, n_assets: int
    ) -> None:
        """Re-estimate over the names tradable *today*, if the cache has aged out.

        Two properties matter here and both are easy to lose.

        **Causality.** The name set comes from ``active``, which is the universe
        on this day, and the window is ``returns.iloc[:day_index]`` — strictly
        before it. Estimating over "every name that is ever in the universe"
        would be a look-ahead, because that set is only knowable at the end of the
        backtest. ``tests/test_no_lookahead.py`` fails if this regresses.

        **Scope.** Estimating over just the tradable names rather than the whole
        downloaded panel matters for shrinkage: Ledoit-Wolf calibrates to the
        ratio of observations to variables, so handing it 1450 columns in order
        to use a 100-name sub-block shrinks that block far harder than its own
        T/N justifies, flattening the risk model.
        """
        if self._matrix is not None and day_index - self._cached_at < self.cfg.cov_refresh:
            return
        lo = max(0, day_index - self.cfg.cov_lookback)
        window = returns.iloc[lo:day_index, active]
        frame = self._estimate_once(window, window.columns)

        self._cache = frame
        self._matrix = frame.to_numpy(dtype=float)
        self._positions = np.full(n_assets, -1, dtype=int)
        self._positions[active] = np.arange(len(active))
        self._cached_at = day_index

    def block(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Covariance for ``positions``, plus a mask of which ones were available.

        A name that joined the universe since the last refresh has no estimate
        yet. Rather than invent one, it is reported as unavailable and the
        portfolio simply skips it until the next refresh — both refreshes run on
        the same cadence, so this is a handful of names at most.
        """
        local = self._positions[positions]
        keep = local >= 0
        kept = local[keep]
        return self._matrix[np.ix_(kept, kept)], keep

    def estimate(self, returns: pd.DataFrame, day_index: int, tickers: pd.Index) -> pd.DataFrame:
        """Convenience wrapper returning a labelled frame for the given tickers."""
        positions = np.array([returns.columns.get_loc(t) for t in tickers])
        self.refresh_if_stale(returns, day_index, positions, len(returns.columns))
        return self._cache.reindex(index=tickers, columns=tickers)

    def reset(self) -> None:
        self._cache = None
        self._matrix = None
        self._positions = None
        self._cached_at = -10**9

    def _estimate_once(self, window: pd.DataFrame, tickers: pd.Index) -> pd.DataFrame:
        if len(window) < 20:
            return diagonal_covariance(window, tickers)
        method = self.cfg.cov_estimator
        if method == "ledoit_wolf":
            return ledoit_wolf_covariance(window, tickers)
        if method == "pca":
            return pca_covariance(window, tickers, self.cfg.pca_factors)
        if method == "sample":
            return sample_covariance(window, tickers)
        if method == "diagonal":
            return diagonal_covariance(window, tickers)
        raise ValueError(f"unknown covariance estimator {method!r}")


def condition_number(cov: pd.DataFrame) -> float:
    """Diagnostic: how close to singular is this estimate?"""
    eig = np.linalg.eigvalsh(cov.to_numpy())
    lo = max(eig.min(), 1e-18)
    return float(eig.max() / lo)
