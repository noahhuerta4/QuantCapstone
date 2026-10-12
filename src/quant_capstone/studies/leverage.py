"""The final book at 1x, 3x and 5x gross, at $10m and at $1m.

Each is a genuine backtest at that NAV, not a scaled return series, so the
per-order commission minimum, ADV participation, square-root impact and margin
financing are all computed on real dollar sizes.
"""

from __future__ import annotations

import pandas as pd

from ..costs import attribution_bps
from ..data import load_dataset
from ..risk.pipeline import build_portfolio
from ..signals import build_signal
from ..reporting.metrics import drawdown_duration, max_drawdown, sharpe
from . import eras
from .common import backtest, turnover

DESCRIPTION = "Leverage 1x/3x/5x at $10m and $1m NAV, financing charged"


def _levered(ds, cfg, weights) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, curves = [], {}
    for lev in (1, 3, 5):
        r = backtest(ds, cfg, weights * lev)
        a = attribution_bps(r.costs, r.nav)
        final = float(r.nav.iloc[-1] * (1 + r.net_return.iloc[-1]))
        rows.append({
            "leverage": f"{lev}x", "gross_exp": float(r.exposures["gross"].mean()),
            "gross_sharpe": sharpe(r.gross_return), "net_sharpe": sharpe(r.net_return),
            "ann_return": float(r.net_return.mean() * 252), "ann_vol": float(r.net_return.std() * 252 ** 0.5),
            "max_dd": max_drawdown(r.net_return),
            "dd_days": drawdown_duration(r.net_return)["max_drawdown_days"],
            "cost_bps": float(a["total"]), "commission_bps": float(a["commission"]),
            "financing_bps": float(a["financing"]), "borrow_bps": float(a["borrow"]),
            "turnover": turnover(r), "final_nav": final,
        })
        curve = r.nav.copy()
        curve.iloc[-1] = final
        curves[f"{lev}x"] = curve
    return pd.DataFrame(rows), pd.DataFrame(curves)


def run() -> dict[str, pd.DataFrame]:
    out = {}
    for nav, suffix in ((10_000_000.0, ""), (1_000_000.0, "_1m")):
        cfg = eras.FINAL.override(execution__initial_nav=nav)
        ds = load_dataset(cfg)
        weights = build_portfolio(ds, build_signal(ds, cfg), cfg).weights
        summary, curves = _levered(ds, cfg, weights)
        out[f"leverage_summary{suffix}"], out[f"leverage_curves{suffix}"] = summary, curves
    return out
