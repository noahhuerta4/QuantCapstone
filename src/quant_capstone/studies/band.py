"""The no-trade band, the lever direction C actually found.

Validated two ways: in the walk-forward grid with the embargo active (never
pinned), and swept far enough to show the optimum is interior -- a band wider
than the 1.5% per-name cap means no trade can ever clear it.
"""

from __future__ import annotations

import pandas as pd

from .. import validation as v
from ..data import load_dataset
from ..risk.pipeline import build_portfolio
from ..signals import build_signal
from ..reporting.metrics import sharpe
from . import eras
from .common import backtest, turnover

DESCRIPTION = "No-trade band: walk-forward with the band in the grid, and an interior sweep"
GRID = {"signal__pead_weight": [0.0, 0.10, 0.20], "execution__rebalance_every": [5, 10, 15],
        "execution__no_trade_band": [0.003, 0.006, 0.010]}
BANDS = (0.0, 0.003, 0.006, 0.010, 0.013, 0.016, 0.020, 0.030)


def run() -> dict[str, pd.DataFrame]:
    base = eras.PEAD
    ds = load_dataset(base)
    folds = v.walk_forward(base, ds, GRID).folds

    weights = build_portfolio(ds, build_signal(ds, base), base).weights
    rows = []
    for band in BANDS:
        r = backtest(ds, base.override(execution__no_trade_band=band), weights)
        rows.append({"band": band, "gross": sharpe(r.gross_return), "net": sharpe(r.net_return),
                     "turnover": turnover(r), "trades_day": float(r.n_trades.mean()),
                     "gross_exp": float(r.exposures["gross"].mean())})
    return {"phase2_band_folds": folds, "phase2_band_interior": pd.DataFrame(rows)}
