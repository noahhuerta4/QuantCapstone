"""Defensive cleaning (spec S3): bad prints, halts, listing gaps.

The rule that matters: forward-fill only *inside* a name's listed span. Filling
before the first print invents history, and filling after the last print invents
a tradable asset that no longer exists -- the latter is a look-ahead bug that
quietly pays you returns on a delisted name.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import FIELDS, PriceData


def _listed_mask(close: pd.DataFrame) -> pd.DataFrame:
    """True between each name's first and last valid print, inclusive."""
    valid = close.notna()
    forward = valid.cummax()                       # has ever traded
    backward = valid[::-1].cummax()[::-1]          # will ever trade again
    return forward & backward


def flag_bad_prints(close: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Spikes that immediately undo themselves are prints, not returns.

    Tested in **log** space, because simple returns do not cancel: a price that
    triples and comes straight back prints +197% then -66%, which sums to +131%
    and looks nothing like a round trip. The log moves are +1.10 and -1.08, which
    do cancel — so the additive test only works on logs.
    """
    positive = close.where(close > 0)
    log_ret = np.log(positive).diff()
    limit = np.log1p(threshold)

    spike = log_ret.abs() > limit
    reverses = (log_ret + log_ret.shift(-1)).abs() < limit / 2
    return (spike & reverses).fillna(False)


def clean_prices(
    data: PriceData,
    max_ffill: int = 3,
    bad_print_threshold: float = 0.5,
    min_observations: int = 252,
) -> PriceData:
    listed = _listed_mask(data.close)
    bad = flag_bad_prints(data.close, bad_print_threshold)

    fields: dict[str, pd.DataFrame] = {}
    for name in FIELDS:
        frame = getattr(data, name).where(~bad)
        frame = frame.where(listed)
        frame = frame.ffill(limit=max_ffill).where(listed)
        fields[name] = frame

    # Volume forward-filled through a halt should be zero, not the last print.
    fields["volume"] = fields["volume"].where(getattr(data, "volume").notna() | ~listed, 0.0)

    keep = fields["close"].notna().sum() >= min_observations
    cols = fields["close"].columns[keep]
    fields = {k: v.reindex(columns=cols) for k, v in fields.items()}

    return PriceData(**fields, sectors=data.sectors.reindex(cols).fillna("Unknown"))


def data_quality_report(data: PriceData) -> pd.Series:
    close = data.close
    listed = _listed_mask(close)
    return pd.Series(
        {
            "tickers": close.shape[1],
            "days": close.shape[0],
            "pct_missing_in_listed_span": float(
                (~close.notna() & listed).to_numpy().sum() / max(listed.to_numpy().sum(), 1)
            ),
            "names_with_full_history": int((close.notna().sum() == len(close)).sum()),
            "max_abs_daily_return": float(np.nanmax(np.abs(close.pct_change().to_numpy()))),
        }
    )
