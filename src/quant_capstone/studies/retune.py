"""Re-tune the holding period once spreads were calibrated.

Cheaper costs should push the optimum toward faster rebalancing; this measures
whether they did.
"""

from __future__ import annotations

import pandas as pd

from ..data import load_dataset
from ..risk.pipeline import build_portfolio
from ..signals import build_signal
from . import eras
from .common import assert_dense, backtest, summary

DESCRIPTION = "Rebalance interval x blend on calibrated spreads"
BLENDS = {
    "A vsr": ("volume_shock_reversal",),
    "B vsr+overnight": ("volume_shock_reversal", "overnight_reversal"),
    "D B+resmom": ("volume_shock_reversal", "overnight_reversal", "residual_momentum"),
}


def run() -> dict[str, pd.DataFrame]:
    base = eras.FLAT_SPREAD.override(data__min_history=300, costs__use_estimated_spread=True)
    ds = load_dataset(base)
    assert_dense(ds.prices)
    rows = []
    for label, alphas in BLENDS.items():
        cfg = base.override(signal__alphas=alphas)
        weights = build_portfolio(ds, build_signal(ds, cfg), cfg).weights
        for rebal in (1, 2, 3, 5, 10, 15):
            s = summary(backtest(ds, cfg.override(execution__rebalance_every=rebal), weights))
            rows.append({"blend": label, "rebal": rebal, **s})
    return {"retune": pd.DataFrame(rows)}
