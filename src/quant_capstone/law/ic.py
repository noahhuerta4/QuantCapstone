"""Information coefficient (spec S7).

IC is the cross-sectional correlation between the forecast and the return the
strategy actually gets to trade — which under next-open execution is the t+1
open to t+2 open return, not the next close-to-close. Getting that alignment
wrong inflates IC by exactly the amount of overnight reversal the strategy never
captures, which for a *reversal* signal is a large and flattering amount.

Spearman by default. The raw signal is a k-day return, so its cross-section has
fat tails; Pearson would let a handful of names set the correlation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TRADING_DAYS
from ..linalg import nan_corr, rank_corr


def information_coefficient(
    signal: pd.DataFrame, forward: pd.DataFrame, method: str = "spearman"
) -> pd.Series:
    """Daily cross-sectional IC series."""
    fn = rank_corr if method == "spearman" else nan_corr
    aligned = forward.reindex(index=signal.index, columns=signal.columns)
    sig_np, fwd_np = signal.to_numpy(dtype=float), aligned.to_numpy(dtype=float)
    values = np.array([fn(sig_np[i], fwd_np[i]) for i in range(len(signal))])
    return pd.Series(values, index=signal.index, name=f"ic_{method}")


def ic_summary(ic: pd.Series, periods_per_year: int = TRADING_DAYS) -> pd.Series:
    """Mean IC, its stability, and the resulting t-statistic.

    ``ic_ir`` (mean IC divided by its standard deviation, annualized) is the
    'IC breadth' number: a signal with a small but *stable* IC is worth far more
    than one with a large erratic IC, and this is where that shows up.
    """
    clean = ic.dropna()
    if len(clean) < 2:
        return pd.Series(dtype=float)
    mean, std = float(clean.mean()), float(clean.std(ddof=1))
    return pd.Series(
        {
            "ic_mean": mean,
            "ic_std": std,
            "ic_ir_annualized": mean / std * np.sqrt(periods_per_year) if std > 0 else np.nan,
            "ic_t_stat": mean / std * np.sqrt(len(clean)) if std > 0 else np.nan,
            "ic_hit_rate": float((clean > 0).mean()),
            "n_periods": float(len(clean)),
        }
    )


def ic_by_bucket(
    signal: pd.DataFrame,
    forward: pd.DataFrame,
    characteristic: pd.DataFrame,
    n_buckets: int = 5,
    method: str = "spearman",
    min_names: int = 8,
) -> pd.DataFrame:
    """IC measured *within* cross-sectional buckets of a characteristic.

    Sorting on dollar volume answers the question this strategy lives or dies on:
    is the reversal edge actually stronger in less liquid names, as
    Avramov-Chordia-Goyal (2006) report? Measuring IC on the pooled universe
    averages that structure away.

    Buckets are formed *within each day* on the day's own cross-section, so the
    comparison is between contemporaneous names rather than between market
    regimes. ``Q1`` is the lowest characteristic value (least liquid, when
    sorting on ADV) and ``QN`` the highest.
    """
    fn = rank_corr if method == "spearman" else nan_corr
    sig = signal.to_numpy(dtype=float)
    fwd = forward.reindex_like(signal).to_numpy(dtype=float)
    char = characteristic.reindex_like(signal).to_numpy(dtype=float)

    out = np.full((len(signal), n_buckets), np.nan)
    for i in range(len(signal)):
        usable = np.flatnonzero(np.isfinite(sig[i]) & np.isfinite(fwd[i]) & np.isfinite(char[i]))
        if usable.size < n_buckets * min_names:
            continue
        ordered = usable[np.argsort(char[i][usable])]
        for b, members in enumerate(np.array_split(ordered, n_buckets)):
            if members.size >= min_names:
                out[i, b] = fn(sig[i][members], fwd[i][members])

    return pd.DataFrame(
        out, index=signal.index, columns=[f"Q{b + 1}" for b in range(n_buckets)]
    )


def bucket_summary(ic_buckets: pd.DataFrame) -> pd.DataFrame:
    """Mean IC and t-statistic per bucket, plus the Q1-minus-QN spread.

    The last row is the actual hypothesis test: if the illiquid-minus-liquid IC
    difference is not distinguishable from zero, moving the universe down the
    liquidity spectrum buys nothing and the remaining cost increase is pure loss.
    """
    rows = {}
    for column in ic_buckets.columns:
        series = ic_buckets[column].dropna()
        if len(series) < 3:
            continue
        rows[column] = {
            "ic_mean": float(series.mean()),
            "ic_t_stat": float(series.mean() / series.std(ddof=1) * np.sqrt(len(series))),
            "ic_hit_rate": float((series > 0).mean()),
            "n_periods": float(len(series)),
        }
    table = pd.DataFrame(rows).T

    first, last = ic_buckets.columns[0], ic_buckets.columns[-1]
    spread = (ic_buckets[first] - ic_buckets[last]).dropna()
    if len(spread) > 2:
        table.loc[f"{first} minus {last}"] = {
            "ic_mean": float(spread.mean()),
            "ic_t_stat": float(spread.mean() / spread.std(ddof=1) * np.sqrt(len(spread))),
            "ic_hit_rate": float((spread > 0).mean()),
            "n_periods": float(len(spread)),
        }
    return table


def ic_decay(signal: pd.DataFrame, returns: pd.DataFrame, horizons: range | list[int]) -> pd.Series:
    """IC against returns h days ahead, for h in ``horizons``.

    The decay profile sets the holding period: if IC is gone by day 3, holding for
    a week is paying costs for nothing.
    """
    out = {}
    for h in horizons:
        fwd = returns.shift(-h)
        out[h] = information_coefficient(signal, fwd).mean()
    return pd.Series(out, name="ic_decay").rename_axis("horizon_days")
