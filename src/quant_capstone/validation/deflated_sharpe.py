"""Probabilistic and Deflated Sharpe Ratios (Bailey & Lopez de Prado; spec S10).

The problem they solve: if you try 40 configurations and report the best, its
Sharpe is the maximum of 40 draws, which is high even when every configuration is
worthless. The Deflated Sharpe Ratio asks whether the winner beats the Sharpe you
would *expect* from the best of 40 worthless strategies.

Two corrections, both of which matter here:

* **Multiple testing.** The expected maximum Sharpe under the null grows with the
  number of trials and with the spread of Sharpes across them.
* **Non-normal returns.** Negative skew and fat tails make a Sharpe less
  trustworthy than the normal-theory standard error suggests. A short-term
  reversal book is a liquidity provider and has exactly that return shape, so
  this is not a formality.

All Sharpes here are **per-period** (daily), not annualized. ``annual_to_daily``
converts, and getting this wrong inflates the result enormously.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..config import TRADING_DAYS

EULER_MASCHERONI = 0.5772156649015329


def annual_to_daily(sharpe_annual: float, periods: int = TRADING_DAYS) -> float:
    return sharpe_annual / np.sqrt(periods)


def daily_to_annual(sharpe_daily: float, periods: int = TRADING_DAYS) -> float:
    return sharpe_daily * np.sqrt(periods)


def probabilistic_sharpe(
    observed_sharpe: float, benchmark_sharpe: float, n_obs: int, skew: float, kurtosis: float
) -> float:
    """P(true Sharpe > benchmark), correcting for skew and fat tails.

    ``kurtosis`` is the **non-excess** fourth moment (3.0 for a normal), which is
    what the published formula expects.
    """
    if n_obs < 3 or not np.isfinite(observed_sharpe):
        return np.nan
    variance = 1.0 - skew * observed_sharpe + (kurtosis - 1.0) / 4.0 * observed_sharpe**2
    if variance <= 0:
        return np.nan
    z = (observed_sharpe - benchmark_sharpe) * np.sqrt(n_obs - 1) / np.sqrt(variance)
    return float(norm.cdf(z))


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """Expected maximum Sharpe across ``n_trials`` independent worthless strategies."""
    if n_trials < 2 or sharpe_variance <= 0:
        return 0.0
    n = float(n_trials)
    quantile = (1.0 - EULER_MASCHERONI) * norm.ppf(1.0 - 1.0 / n) + EULER_MASCHERONI * norm.ppf(
        1.0 - 1.0 / (n * np.e)
    )
    return float(np.sqrt(sharpe_variance) * quantile)


def deflated_sharpe(
    returns: pd.Series, n_trials: int, trial_sharpes: pd.Series | np.ndarray | None = None
) -> pd.Series:
    """Full DSR block for one return series.

    ``trial_sharpes`` should be the per-period Sharpes of every configuration
    tried; their variance is what sets the deflation. Passing only the count
    falls back to the observed series' own variance estimate, which understates
    the correction — so pass them when you have them.
    """
    r = returns.replace([np.inf, -np.inf], np.nan).dropna()
    n_obs = len(r)
    if n_obs < 30 or r.std(ddof=1) == 0:
        return pd.Series(dtype=float)

    sr_daily = float(r.mean() / r.std(ddof=1))
    skew = float(r.skew())
    kurt = float(r.kurtosis() + 3.0)  # pandas reports excess kurtosis

    if trial_sharpes is not None and len(np.asarray(trial_sharpes)) > 1:
        variance = float(np.nanvar(np.asarray(trial_sharpes, dtype=float), ddof=1))
    else:
        variance = 1.0 / max(n_obs - 1, 1)

    sr0 = expected_max_sharpe(n_trials, variance)
    return pd.Series(
        {
            "sharpe_annual": daily_to_annual(sr_daily),
            "sharpe_daily": sr_daily,
            "skew": skew,
            "excess_kurtosis": kurt - 3.0,
            "n_obs": float(n_obs),
            "n_trials": float(n_trials),
            "trial_sharpe_variance": variance,
            "expected_max_sharpe_daily": sr0,
            "expected_max_sharpe_annual": daily_to_annual(sr0),
            "psr_vs_zero": probabilistic_sharpe(sr_daily, 0.0, n_obs, skew, kurt),
            "deflated_sharpe_ratio": probabilistic_sharpe(sr_daily, sr0, n_obs, skew, kurt),
            "min_track_record_length_days": minimum_track_record_length(sr_daily, skew, kurt),
        }
    )


def minimum_track_record_length(
    sharpe_daily: float, skew: float, kurtosis: float, confidence: float = 0.95
) -> float:
    """Days of history needed to call this Sharpe significantly positive.

    If this exceeds the sample we have, the honest statement is 'we cannot yet
    distinguish this from zero', whatever the point estimate says.
    """
    if not np.isfinite(sharpe_daily) or sharpe_daily <= 0:
        return np.nan
    variance = 1.0 - skew * sharpe_daily + (kurtosis - 1.0) / 4.0 * sharpe_daily**2
    if variance <= 0:
        return np.nan
    return float(1.0 + variance * (norm.ppf(confidence) / sharpe_daily) ** 2)
