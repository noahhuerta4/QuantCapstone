"""Synthetic price panel with a *known* reversal effect.

Two jobs. First, the test suite runs offline and deterministically. Second — more
useful — it is a ground truth: we plant a reversal of known strength, so if the
signal/backtest pipeline cannot recover it here, the bug is ours, not the
market's. Returns are built as market + sector + idiosyncratic MA(1), where the
negative MA coefficient is what creates short-horizon mean reversion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import FIELDS, PriceData

SECTORS = [
    "Information Technology", "Financials", "Health Care", "Industrials",
    "Consumer Discretionary", "Consumer Staples", "Energy", "Utilities",
]


def make_synthetic(
    n_assets: int = 60,
    n_days: int = 1500,
    reversal_phi: float = 0.25,
    overnight_phi: float = 0.5,
    market_vol: float = 0.010,
    idio_vol: float = 0.015,
    gap_vol: float = 0.004,
    seed: int = 7,
    start: str = "2015-01-02",
) -> PriceData:
    """Build a ``PriceData`` panel with two independently planted effects.

    ``reversal_phi`` is the MA(1) coefficient on idiosyncratic shocks: a return of
    ``eps[t] - phi*eps[t-1]`` has lag-1 autocorrelation ``-phi/(1+phi**2)``, so
    larger phi means a stronger (more easily detected) reversal.

    ``overnight_phi`` is the fraction of each overnight gap that reverses during
    the following session (Lou-Polk-Skouras). This one has to be planted
    deliberately: an *iid* gap contributes nothing to a tradable overnight
    reversal, because the open-to-open return already differences it out. Without
    it, the ``overnight_reversal`` sleeve has no ground truth to recover and the
    synthetic panel silently stops exercising the default configuration.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days, name="date")
    tickers = pd.Index([f"SYN{i:03d}" for i in range(n_assets)], name="ticker")

    betas = rng.uniform(0.6, 1.4, n_assets)
    sector_ids = rng.integers(0, len(SECTORS), n_assets)

    market = rng.normal(0.0003, market_vol, n_days)
    sector_ret = rng.normal(0.0, 0.006, (n_days, len(SECTORS)))

    eps = rng.normal(0.0, idio_vol, (n_days + 1, n_assets))
    idio = eps[1:] - reversal_phi * eps[:-1]

    # Gaps are drawn first so the reversal they cause can be folded into the
    # close-to-close return of the *following* day.
    gap_values = rng.normal(0, gap_vol, (n_days, n_assets))
    overnight_reversion = np.zeros_like(gap_values)
    overnight_reversion[1:] = -overnight_phi * gap_values[:-1]

    returns = (
        market[:, None] * betas[None, :]
        + sector_ret[:, sector_ids]
        + idio
        + overnight_reversion
    )

    close = pd.DataFrame(
        100.0 * np.exp(np.cumsum(returns, axis=0)), index=dates, columns=tickers
    )
    gap = pd.DataFrame(gap_values, index=dates, columns=tickers)
    open_ = (close.shift(1) * (1 + gap)).bfill()
    high = pd.concat([open_, close]).groupby(level=0).max() * (1 + abs(gap) / 2)
    low = pd.concat([open_, close]).groupby(level=0).min() * (1 - abs(gap) / 2)

    base_volume = rng.lognormal(14.5, 0.8, n_assets)
    volume = pd.DataFrame(
        base_volume[None, :] * rng.lognormal(0, 0.3, (n_days, n_assets)),
        index=dates, columns=tickers,
    ).round()

    panel = {
        "open": open_, "high": high, "low": low, "close": close,
        "open_raw": open_, "close_raw": close, "volume": volume,
    }
    sectors = pd.Series([SECTORS[i] for i in sector_ids], index=tickers)
    return PriceData(**{f: panel[f].astype(float) for f in FIELDS}, sectors=sectors)


class SyntheticSource:
    """DataSource-compatible wrapper so the CLI can run fully offline."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fetch(self, tickers: list[str], start: str, end: str) -> PriceData:
        return make_synthetic(**self.kwargs)
