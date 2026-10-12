"""Which sleeves to blend: one per family, then families added.

Run under both spread assumptions of the time. The "est" rows use the
Corwin-Schultz bar estimator, which predates the calibrated spread model and is
why they net negative. The vsr + overnight pairing is the single largest gain in
the project.
"""

from __future__ import annotations

import pandas as pd

from ..data import load_dataset
from ..risk.pipeline import build_portfolio
from ..signals import build_signal
from . import eras
from .common import assert_dense, backtest, summary

DESCRIPTION = "Six blend combinations, flat vs estimated spread"
BLENDS = {
    "A vsr only": ("volume_shock_reversal",),
    "B vsr+overnight": ("volume_shock_reversal", "overnight_reversal"),
    "C vsr+resmom": ("volume_shock_reversal", "residual_momentum"),
    "D vsr+overnight+resmom": ("volume_shock_reversal", "overnight_reversal", "residual_momentum"),
    "E rev+vsr+overnight+resmom": ("reversal", "volume_shock_reversal", "overnight_reversal",
                                   "residual_momentum"),
    "F D + idiovol (control)": ("volume_shock_reversal", "overnight_reversal", "residual_momentum",
                                "idiosyncratic_volatility"),
}


def run() -> dict[str, pd.DataFrame]:
    base = eras.FLAT_SPREAD.override(data__min_history=300)
    ds = load_dataset(base)
    assert_dense(ds.prices)
    rows = []
    for label, alphas in BLENDS.items():
        cfg = base.override(signal__alphas=alphas)
        weights = build_portfolio(ds, build_signal(ds, cfg), cfg).weights
        for estimated in (False, True):
            s = summary(backtest(ds, cfg.override(costs__use_estimated_spread=estimated,
                                                 costs__spread_estimator="corwin_schultz"), weights))
            rows.append({"blend": label, "n": len(alphas), "spread": "est" if estimated else "flat",
                         "gross": s["gross"], "net": s["net"],
                         "turnover": s["turnover"], "cost_bps": s["cost_bps"]})
    return {"blends": pd.DataFrame(rows)}
