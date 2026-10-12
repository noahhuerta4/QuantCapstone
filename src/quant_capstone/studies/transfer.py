"""Raising the transfer coefficient: which constraint costs the most alpha?

The answer is that loosening any of them makes things worse -- TC rises from
0.63 to 0.84 while net Sharpe falls from 0.78 to 0.39. The constraints are
regularization against a noisy alpha (Jagannathan & Ma 2003).
"""

from __future__ import annotations

import pandas as pd

from ..data import load_dataset
from ..law.transfer import transfer_coefficient
from ..risk.pipeline import build_portfolio
from ..signals import build_signal
from ..reporting.metrics import sharpe
from . import eras
from .common import backtest, turnover

DESCRIPTION = "Caps, neutrality and MVO variants against the transfer coefficient"
VARIANTS = {
    "baseline (cap 1.5%)": {},
    "cap 2.5%": {"risk__max_weight": 0.025},
    "cap 4%": {"risk__max_weight": 0.04},
    "cap 8%": {"risk__max_weight": 0.08},
    "no sector neutrality": {"risk__neutralize_sector": False},
    "no beta neutrality": {"risk__neutralize_beta": False},
    "cap 4% + no sector": {"risk__max_weight": 0.04, "risk__neutralize_sector": False},
    "mvo weights": {"risk__weight_mode": "mvo"},
    "mvo + cap 4%": {"risk__weight_mode": "mvo", "risk__max_weight": 0.04},
    "more iters (20)": {"risk__constraint_iters": 20},
}


def run() -> dict[str, pd.DataFrame]:
    base = eras.BLEND
    ds = load_dataset(base)
    signal = build_signal(ds, base)
    rows = []
    for label, overrides in VARIANTS.items():
        cfg = base.override(**overrides)
        port = build_portfolio(ds, signal, cfg)
        r = backtest(ds, cfg, port.weights)
        live = port.diagnostics.query("n_active>0")
        rows.append({
            "variant": label,
            "tc": float(transfer_coefficient(port.ideal_weights, port.weights).dropna().mean()),
            "gross": sharpe(r.gross_return), "net": sharpe(r.net_return),
            "ex_ante_vol": float(live["ex_ante_vol"].mean()),
            "gross_exp": float(r.exposures["gross"].mean()),
            "n_at_cap": float(live["n_at_cap"].mean()),
            "turnover": turnover(r),
        })
    return {"transfer_coefficient": pd.DataFrame(rows).sort_values("net", ascending=False)}
