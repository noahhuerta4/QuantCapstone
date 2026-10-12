"""Phase 2, direction A: six pre-registered event-driven sleeves.

The inclusion gate (all four must pass): significant standalone IC,
autocorrelation low enough for breadth, correlation to the existing sleeves
below ~0.3, and a positive *marginal* contribution to net Sharpe. None passed.
The insider probe checks that the significantly negative insider IC is not an
artefact of zero-filling days with no filing; the revenue test checks whether
the best-IC candidate adds to earnings surprise or duplicates it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest import forward_returns
from ..data import load_dataset
from ..data.edgar_form4 import to_panel
from ..law import ic_summary, information_coefficient
from ..signals import build_sleeves, sleeve_correlations
from ..signals.fundamental import market_cap
from ..signals.insider import insider_flow_fast
from ..reporting.metrics import sharpe
from . import eras
from .common import backtest, fixed_blend, sleeve_row, t_stat, turnover

DESCRIPTION = "Direction A gate, insider artefact probe, revenue vs earnings surprise"
BASE = ("volume_shock_reversal", "overnight_reversal")
CANDIDATES = ("insider_flow", "insider_opportunistic", "insider_purchases",
              "insider_breadth", "revenue_surprise", "announcement_reversal")
CURRENT = {"volume_shock_reversal": 0.45, "overnight_reversal": 0.45, "earnings_surprise": 0.10}


def _scores(ds, weights: dict[str, float]) -> tuple[float, float, float]:
    r = backtest(ds, fixed_blend(eras.PEAD, weights))
    return sharpe(r.gross_return), sharpe(r.net_return), turnover(r)


def _marginal(ds) -> pd.DataFrame:
    gross, net, turn = _scores(ds, CURRENT)
    rows = [{"candidate": "(current blend)", "gross": gross, "net": net, "delta_net": 0.0, "turnover": turn}]
    for cand in CANDIDATES:
        w = {"volume_shock_reversal": 0.40, "overnight_reversal": 0.40, "earnings_surprise": 0.05, cand: 0.15}
        g, n, t = _scores(ds, w)
        rows.append({"candidate": cand, "gross": g, "net": n, "delta_net": n - net, "turnover": t})
    return pd.DataFrame(rows).sort_values("delta_net", ascending=False)


def _insider_probe(ds, fwd) -> pd.DataFrame:
    flow = to_panel(ds.form4, ds.prices.dates, ds.prices.tickers)
    members = ds.members
    cap = market_cap(ds.prices.close_raw, ds.fundamentals["shares"])
    sig = insider_flow_fast(flow, cap, eras.PEAD.signal.insider_halflife).where(members)
    raw = ic_summary(information_coefficient(sig, fwd).dropna())
    filed = flow.rolling(60, min_periods=1).apply(lambda x: (x != 0).any(), raw=True) > 0
    ic_filed = information_coefficient(sig.where(filed), fwd).dropna()
    return pd.DataFrame([{
        # (flow != 0) & members, not flow.where(members) != 0: NaN != 0 is True.
        "filing_rate": float(((flow != 0) & members).sum().sum() / members.sum().sum()),
        "zero_filled_fraction": float((sig == 0).sum().sum() / sig.notna().sum().sum()),
        "ic_mean": raw["ic_mean"], "ic_t": raw["ic_t_stat"],
        "ic_filed_60d": float(ic_filed.mean()), "ic_filed_60d_t": t_stat(ic_filed, ddof=1),
    }])


def _revenue_test(ds) -> pd.DataFrame:
    specs = {
        "current: earnings only": CURRENT,
        "swap: revenue only": {"volume_shock_reversal": .45, "overnight_reversal": .45, "revenue_surprise": .10},
        "both, 5% each": {"volume_shock_reversal": .45, "overnight_reversal": .45,
                          "earnings_surprise": .05, "revenue_surprise": .05},
    }
    rows = []
    for label, w in specs.items():
        g, n, _ = _scores(ds, w)
        rows.append({"config": label, "gross": g, "net": n})
    return pd.DataFrame(rows)


def run() -> dict[str, pd.DataFrame]:
    cfg = eras.PEAD.override(signal__alphas=BASE + CANDIDATES + ("earnings_surprise",),
                             signal__pead_weight=0.0)
    ds = load_dataset(cfg)
    sleeves = build_sleeves(ds, cfg)
    fwd = forward_returns(ds.prices, "next_open")
    return {
        "phase2_gate": pd.DataFrame([sleeve_row(n, s, fwd, ds.members) for n, s in sleeves.items()]),
        "phase2_sleeve_correlations": sleeve_correlations(sleeves),
        "phase2_marginal": _marginal(ds),
        "phase2_insider_probe": _insider_probe(ds, fwd),
        "phase2_revenue_test": _revenue_test(ds),
    }
