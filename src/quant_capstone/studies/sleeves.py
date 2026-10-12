"""Seven price sleeves measured standalone, and how they correlate.

This is where the sleeves collapse into three families (reversal, momentum,
volatility) and where momentum's best-in-class IC is shown to buy nothing at an
autocorrelation of 0.995.
"""

from __future__ import annotations

import pandas as pd

from ..backtest import forward_returns
from ..data import load_dataset
from ..signals import build_sleeves, sleeve_correlations
from ..reporting.metrics import sharpe
from . import eras
from .common import assert_dense, backtest, cost_bps, sleeve_row, turnover

DESCRIPTION = "Seven price sleeves: IC, autocorrelation, standalone Sharpe, correlations"
PRICE_SLEEVES = ("reversal", "volume_shock_reversal", "overnight_reversal", "momentum",
                 "residual_momentum", "low_volatility", "idiosyncratic_volatility")


def run() -> dict[str, pd.DataFrame]:
    base = eras.FLAT_SPREAD.override(signal__alphas=PRICE_SLEEVES, data__min_history=300)
    ds = load_dataset(base)
    assert_dense(ds.prices)
    sleeves = build_sleeves(ds, base)
    fwd = forward_returns(ds.prices, "next_open")

    rows = []
    for name, sleeve in sleeves.items():
        r = backtest(ds, base.override(signal__alphas=(name,)))
        row = sleeve_row(name, sleeve, fwd)
        row.update({"gross_sharpe": sharpe(r.gross_return), "net_sharpe": sharpe(r.net_return),
                    "ann_turnover": turnover(r), "cost_bps": cost_bps(r)})
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("net_sharpe", ascending=False)
    return {"sleeves": table, "sleeve_correlations": sleeve_correlations(sleeves)}
