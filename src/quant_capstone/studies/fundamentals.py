"""Static value and quality ratios from point-in-time SEC filings.

Beautifully independent of the price sleeves and useless anyway: every one has
autocorrelation of 0.997+, so it delivers a handful of independent bets a year.
The blend test checks whether they help at *any* weight.
"""

from __future__ import annotations

import pandas as pd

from ..backtest import forward_returns
from ..data import load_dataset
from ..signals import build_sleeves, sleeve_correlations
from ..reporting.metrics import sharpe
from . import eras
from .common import backtest, fixed_blend, sleeve_row, turnover

DESCRIPTION = "Six value/quality sleeves standalone, correlations, and blended at several weights"
PRICE = ("volume_shock_reversal", "overnight_reversal")
RATIOS = ("book_to_price", "earnings_yield", "gross_profitability",
          "return_on_equity", "asset_growth", "accruals")


def _blend_rows(ds, base) -> list[dict]:
    def row(label, cfg):
        r = backtest(ds, cfg)
        return {"config": label, "gross": sharpe(r.gross_return), "net": sharpe(r.net_return),
                "turnover": turnover(r)}

    rows = [row("price only (baseline)", base.override(signal__alphas=PRICE))]
    for w in (0.10, 0.20, 0.35):
        p = (1.0 - w) / 2
        rows.append(row(f"price + earnings_yield @ {w:.0%}", fixed_blend(base, {
            "volume_shock_reversal": p, "overnight_reversal": p, "earnings_yield": w})))
    for w in (0.15, 0.30):
        p, f = (1.0 - w) / 2, w / 2
        rows.append(row(f"price + value/quality @ {w:.0%}", fixed_blend(base, {
            "volume_shock_reversal": p, "overnight_reversal": p,
            "earnings_yield": f, "gross_profitability": f})))
    rows.append(row("all four, equal weight",
                    base.override(signal__alphas=PRICE + ("earnings_yield", "gross_profitability"))))
    return rows


def run() -> dict[str, pd.DataFrame]:
    base = eras.BLEND.override(data__min_history=300)
    survey = base.override(signal__alphas=PRICE + RATIOS)
    ds = load_dataset(survey)
    sleeves = build_sleeves(ds, survey)
    fwd = forward_returns(ds.prices, "next_open")

    rows = []
    for name, sleeve in sleeves.items():
        r = backtest(ds, base.override(signal__alphas=(name,)))
        row = sleeve_row(name, sleeve, fwd, ds.members)
        row.update({"gross": sharpe(r.gross_return), "net": sharpe(r.net_return),
                    "turnover": turnover(r)})
        rows.append(row)
    return {
        "fundamental_sleeves": pd.DataFrame(rows).sort_values("net", ascending=False),
        "all_sleeve_correlations": sleeve_correlations(sleeves),
        "fundamental_blend": pd.DataFrame(_blend_rows(ds, base)),
    }
