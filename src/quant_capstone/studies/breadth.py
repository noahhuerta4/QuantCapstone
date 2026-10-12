"""With realistic spreads, does a wider universe buy breadth faster than it buys cost?

S&P 1500, the blend, calibrated spreads, at four universe sizes and three
holding periods.
"""

from __future__ import annotations

import pandas as pd

from ..risk.pipeline import build_portfolio
from ..signals import build_signal
from . import eras
from .common import backtest, sp1500_prices, sub_universe, summary

DESCRIPTION = "Universe size 100-800 x rebalance interval on S&P 1500"


def run() -> dict[str, pd.DataFrame]:
    prices = sp1500_prices(min_observations=300)
    rows = []
    for universe in (100, 200, 400, 800):
        ds = sub_universe(prices, universe, 300)
        cfg = eras.FLAT_SPREAD.override(
            data__universe_size=universe, data__min_history=300,
            signal__alphas=("volume_shock_reversal", "overnight_reversal"),
            costs__use_estimated_spread=True,
        )
        weights = build_portfolio(ds, build_signal(ds, cfg), cfg).weights
        for rebal in (5, 10, 15):
            s = summary(backtest(ds, cfg.override(execution__rebalance_every=rebal), weights))
            rows.append({"universe": universe, "rebal": rebal, **s})
    return {"breadth": pd.DataFrame(rows)}
