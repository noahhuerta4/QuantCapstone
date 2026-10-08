"""Cross-sectional transforms (spec S4).

Every function here operates **row-wise** on a ``date x ticker`` frame: each day
is a self-contained cross-section, so no transform can ever leak information
across time. That property is what makes them safe, and it is what the tests
assert.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

MAD_TO_SIGMA = 1.4826


def mask_to_universe(signal: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    """Blank out anything outside the point-in-time universe."""
    return signal.where(members.reindex_like(signal).fillna(False))


def winsorize_mad(signal: pd.DataFrame, n_mad: float = 3.0) -> pd.DataFrame:
    """Clip each cross-section at median +/- n * MAD.

    MAD rather than standard deviation because the outliers we are clipping are
    exactly what would inflate a standard-deviation estimate and thereby protect
    themselves from being clipped.
    """
    med = signal.median(axis=1)
    mad = signal.sub(med, axis=0).abs().median(axis=1) * MAD_TO_SIGMA
    # A degenerate cross-section (mad == 0) must not collapse to a constant.
    mad = mad.replace(0.0, np.nan)
    lower, upper = med - n_mad * mad, med + n_mad * mad
    return signal.clip(lower=lower.fillna(-np.inf), upper=upper.fillna(np.inf), axis=0)


def winsorize_quantile(signal: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    lo = signal.quantile(lower, axis=1)
    hi = signal.quantile(upper, axis=1)
    return signal.clip(lower=lo, upper=hi, axis=0)


def zscore(signal: pd.DataFrame) -> pd.DataFrame:
    """Demean and scale each cross-section to unit dispersion."""
    std = signal.std(axis=1, ddof=1).replace(0.0, np.nan)
    return signal.sub(signal.mean(axis=1), axis=0).div(std, axis=0)


def rank_normal(signal: pd.DataFrame) -> pd.DataFrame:
    """Map each cross-section to normal scores via its ranks.

    Discards magnitude and keeps only ordering, which is the honest thing to do
    when the raw signal is a noisy k-day return: we trust 'A reverted more than
    B' far more than 'A reverted 2.3x as much as B'.
    """
    ranks = signal.rank(axis=1, method="average")
    counts = signal.notna().sum(axis=1)
    uniform = ranks.div(counts + 1, axis=0)
    values = norm.ppf(uniform.to_numpy(dtype=float))
    return pd.DataFrame(values, index=signal.index, columns=signal.columns)


def demean(signal: pd.DataFrame) -> pd.DataFrame:
    """Force each cross-section to sum to zero (dollar neutrality of the score)."""
    return signal.sub(signal.mean(axis=1), axis=0)


def standardize(signal: pd.DataFrame, method: str = "zscore") -> pd.DataFrame:
    if method == "zscore":
        return zscore(signal)
    if method == "rank":
        return demean(rank_normal(signal))
    raise ValueError(f"unknown standardize method {method!r}")
