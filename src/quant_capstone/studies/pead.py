"""Earnings surprise: the fundamental signal with an event clock.

Same filings as the static ratios, a different shape in time -- it arrives,
decays over a quarter, and the staggered fiscal calendar keeps the cross-section
turning over. The fine weight sweep checks plateau against spike before the
weight is believed.
"""

from __future__ import annotations

import pandas as pd

from ..backtest import forward_returns
from ..data import load_dataset
from ..signals import build_sleeves
from ..reporting.metrics import sharpe
from . import eras
from .common import backtest, fixed_blend, sleeve_row, turnover

DESCRIPTION = "PEAD sleeve stats, standalone and blended, and a fine blend-weight sweep"
PRICE = ("volume_shock_reversal", "overnight_reversal")


def _weights(w: float) -> dict[str, float]:
    p = (1 - w) / 2
    return {"volume_shock_reversal": p, "overnight_reversal": p, "earnings_surprise": w}


def _row(ds, cfg, **label) -> dict:
    r = backtest(ds, cfg)
    return {**label, "gross": sharpe(r.gross_return), "net": sharpe(r.net_return), "turnover": turnover(r)}


def run() -> dict[str, pd.DataFrame]:
    base = eras.BLEND
    survey = base.override(signal__alphas=PRICE + ("earnings_surprise", "earnings_yield"))
    ds = load_dataset(survey)
    fwd = forward_returns(ds.prices, "next_open")
    stats = pd.DataFrame([sleeve_row(n, s, fwd, ds.members)
                          for n, s in build_sleeves(ds, survey).items()])

    blends = [
        _row(ds, base.override(signal__alphas=("earnings_surprise",)), config="earnings_surprise alone"),
        _row(ds, base.override(signal__alphas=PRICE), config="price only (baseline)"),
        *[_row(ds, fixed_blend(base, _weights(w)), config=f"price + PEAD @ {w:.0%}")
          for w in (0.15, 0.25, 0.40)],
        _row(ds, base.override(signal__alphas=PRICE + ("earnings_surprise",)), config="all three equal"),
    ]
    sweep = [
        _row(ds, base.override(signal__alphas=PRICE) if w == 0 else fixed_blend(base, _weights(w)),
             pead_weight=w)
        for w in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
    ]
    return {"pead_sleeves": stats, "pead": pd.DataFrame(blends), "pead_weight_sweep": pd.DataFrame(sweep)}
