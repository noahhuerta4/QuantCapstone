"""Alpha -> target weights, and scaling to a constant ex-ante vol (spec S5.2/S5.4).

Two weighting modes, both implemented because the comparison is informative:

``risk_weighted`` (default)
    ``w_i proportional to alpha_i / sigma_i``. Direction and magnitude come from
    the signal, size comes from inverse volatility, and nothing is inverted. It
    ignores correlations, which is a real cost — but it cannot be destroyed by a
    noisy covariance estimate, which mean-variance can.

``mvo``
    ``w = (1/(2*lambda)) * Sigma^-1 * alpha``, the unconstrained maximizer of
    ``w'alpha - lambda*w'Sigma*w``, then handed to the neutralization and
    constraint stages. Expect it to be sensitive to Sigma: with T/N ~ 1.3, the
    inverse amplifies exactly the eigenvalues the estimator is least sure about.
    That fragility is a discussion point, not a bug to hide.

Vol targeting is a *uniform* scale factor, which is the useful property: it
changes the size of the book without disturbing dollar/beta/sector neutrality.
"""

from __future__ import annotations

import numpy as np

from ..config import TRADING_DAYS


def risk_weighted(alpha: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Inverse-vol sizing with direction and magnitude from the signal."""
    vol = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    return np.nan_to_num(alpha / vol, nan=0.0)


def mean_variance(alpha: np.ndarray, cov: np.ndarray, lam: float) -> np.ndarray:
    """Unconstrained MVO solution; constraints are imposed downstream."""
    n = len(alpha)
    # Ridge the diagonal before inverting. Without it a near-singular Sigma
    # produces enormous offsetting positions in its smallest eigen-directions.
    ridge = 1e-8 * np.trace(cov) / max(n, 1)
    try:
        raw = np.linalg.solve(cov + ridge * np.eye(n), alpha)
    except np.linalg.LinAlgError:
        raw = np.linalg.pinv(cov) @ alpha
    return np.nan_to_num(raw / (2.0 * max(lam, 1e-8)), nan=0.0)


def portfolio_vol(weights: np.ndarray, cov: np.ndarray) -> float:
    """Annualized ex-ante volatility from a daily covariance matrix."""
    variance = float(weights @ cov @ weights)
    return float(np.sqrt(max(variance, 0.0) * TRADING_DAYS))


def scale_to_vol_target(weights: np.ndarray, cov: np.ndarray, target: float) -> np.ndarray:
    """Uniform rescale so ex-ante annualized vol equals ``target``."""
    current = portfolio_vol(weights, cov)
    if current <= 1e-12:
        return np.zeros_like(weights)
    return weights * (target / current)
