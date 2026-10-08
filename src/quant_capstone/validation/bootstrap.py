"""Block bootstrap confidence intervals for the Sharpe ratio (spec S10).

Report the interval, not the point estimate. A Sharpe of 1.4 with a 95% interval
of [0.3, 2.5] and a Sharpe of 1.4 with [1.1, 1.7] are completely different
claims, and only the interval distinguishes them.

Blocks rather than iid resampling because daily strategy returns are not
independent: volatility clusters, and a reversal book's edge comes and goes with
liquidity regimes. Resampling single days would shatter that structure and
produce intervals that are far too narrow. The circular variant wraps at the end
so every observation gets equal weight.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TRADING_DAYS


def _sharpe_matrix(samples: np.ndarray) -> np.ndarray:
    """Annualized Sharpe for each row of a (n_samples, T) matrix."""
    mean = samples.mean(axis=1)
    std = samples.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(std > 0, mean / std * np.sqrt(TRADING_DAYS), np.nan)


def circular_block_indices(
    n_obs: int, block: int, n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """(n_samples, n_obs) index matrix of wrapped consecutive blocks."""
    block = max(int(block), 1)
    n_blocks = int(np.ceil(n_obs / block))
    starts = rng.integers(0, n_obs, size=(n_samples, n_blocks))
    offsets = np.arange(block)
    idx = (starts[:, :, None] + offsets[None, None, :]).reshape(n_samples, -1)
    return idx[:, :n_obs] % n_obs


def block_bootstrap_sharpe(
    returns: pd.Series,
    n_samples: int = 2000,
    block: int = 21,
    confidence: float = 0.95,
    seed: int = 0,
) -> pd.Series:
    """Bootstrap distribution of the annualized Sharpe."""
    r = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < max(3 * block, 30):
        return pd.Series(dtype=float)

    rng = np.random.default_rng(seed)
    values = r.to_numpy(dtype=float)
    idx = circular_block_indices(len(values), block, n_samples, rng)
    distribution = _sharpe_matrix(values[idx])
    distribution = distribution[np.isfinite(distribution)]
    if not len(distribution):
        return pd.Series(dtype=float)

    alpha = (1.0 - confidence) / 2.0
    point = float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    lo, hi = np.quantile(distribution, [alpha, 1 - alpha])
    return pd.Series(
        {
            "sharpe": point,
            "ci_low": float(lo),
            "ci_high": float(hi),
            "ci_width": float(hi - lo),
            "bootstrap_mean": float(distribution.mean()),
            "bootstrap_std": float(distribution.std(ddof=1)),
            "prob_sharpe_below_zero": float((distribution <= 0).mean()),
            "prob_sharpe_below_2": float((distribution < 2.0).mean()),
            "block_days": float(block),
            "n_samples": float(len(distribution)),
        }
    )


def bootstrap_distribution(
    returns: pd.Series, n_samples: int = 2000, block: int = 21, seed: int = 0
) -> np.ndarray:
    """Raw Sharpe draws, for plotting the sampling distribution."""
    r = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < max(3 * block, 30):
        return np.array([])
    rng = np.random.default_rng(seed)
    values = r.to_numpy(dtype=float)
    idx = circular_block_indices(len(values), block, n_samples, rng)
    draws = _sharpe_matrix(values[idx])
    return draws[np.isfinite(draws)]
