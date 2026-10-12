"""Phase 2, direction C: does conditioning risk on a regime add anything?

Blend IC rises monotonically with cross-sectional dispersion, so scaling the book
with dispersion looks promising. The control is what decides it: a *constant*
multiplier at the same average size, which does better -- the "regime" gain was
reduced turnover, and the no-trade band is the direct way to get that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import validation as v
from ..backtest import forward_returns
from ..data import load_dataset
from ..linalg import rank_corr
from ..risk.pipeline import build_portfolio
from ..signals import build_signal, build_sleeves, cross_sectional_dispersion, regime_multiplier
from ..reporting.metrics import sharpe
from . import eras
from .common import backtest, cost_bps, turnover

DESCRIPTION = "Regime IC diagnostic, three regime overlays, walk-forward, constant-scale control"
SPECS = {
    "control (unconditional)": {},
    "C1 linear, floor 0.4": {"risk__regime_scaling": "linear", "risk__regime_floor": 0.4},
    "C2 linear, floor 0.6": {"risk__regime_scaling": "linear", "risk__regime_floor": 0.6},
    "C3 binary, floor 0.4": {"risk__regime_scaling": "binary", "risk__regime_floor": 0.4},
}
#: The walk-forward grid in force when this ran, plus the regime switch.
GRID = {"signal__pead_weight": [0.0, 0.10, 0.20], "execution__rebalance_every": [5, 10, 15],
        "risk__regime_scaling": ["off", "linear"]}


def _daily_ic(score: pd.DataFrame, fwd: np.ndarray) -> pd.Series:
    s = score.to_numpy(float)
    return pd.Series([rank_corr(s[i], fwd[i]) for i in range(len(score))], index=score.index)


def _buckets(ic: pd.Series, var: pd.Series, name: str) -> pd.DataFrame:
    ok = ic.notna() & var.notna()
    q = pd.qcut(var[ok], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    g = ic[ok].groupby(q, observed=True)
    out = pd.DataFrame({"ic_mean": g.mean(), "ic_t": g.mean() / g.std() * np.sqrt(g.count()), "n": g.count()})
    return out.assign(variable=name).rename_axis("quartile").reset_index()


def _diagnostic(ds, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal, sleeves = build_signal(ds, cfg), build_sleeves(ds, cfg)
    fwd = forward_returns(ds.prices, "next_open").reindex_like(signal).to_numpy(float)
    rets = ds.prices.returns().where(ds.members)
    dispersion = rets.std(axis=1).rolling(21, min_periods=10).mean().shift(1)
    market_vol = rets.mean(axis=1).rolling(21, min_periods=10).std().shift(1)
    ic = _daily_ic(signal, fwd)
    blend = pd.concat([_buckets(ic, dispersion, "dispersion"), _buckets(ic, market_vol, "market_vol")])

    q = pd.qcut(dispersion[dispersion.notna()], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    per_sleeve = pd.DataFrame({
        name: _daily_ic(s, fwd)[dispersion.notna()].groupby(q, observed=True).mean()
        for name, s in sleeves.items()
    })
    return blend, per_sleeve


def _control(ds, base, signal) -> pd.DataFrame:
    disp = cross_sectional_dispersion(ds.prices.returns(), ds.members)
    weights = build_portfolio(ds, signal, base).weights

    def score(w, cfg, label):
        r = backtest(ds, cfg, w)
        return {"spec": label, "gross": sharpe(r.gross_return), "net": sharpe(r.net_return),
                "turnover": turnover(r), "cost_bps": cost_bps(r)}

    rows = [score(weights, base, "control (no scaling)")]
    for mode, label in (("linear", "CONSTANT scale = linear's mean"), ("binary", "CONSTANT scale = binary's mean")):
        m = float(regime_multiplier(disp, mode, 0.4, 1.0).mean())
        rows.append(score(weights * m, base, f"{label} ({m:.3f})"))
    for mode in ("linear", "binary"):
        cfg = base.override(risk__regime_scaling=mode, risk__regime_floor=0.4)
        rows.append(score(build_portfolio(ds, signal, cfg).weights, cfg, f"REGIME {mode}, floor 0.4"))
    for band in (0.006, 0.010):
        rows.append(score(weights, base.override(execution__no_trade_band=band), f"wider no-trade band {band}"))
    return pd.DataFrame(rows)


def run() -> dict[str, pd.DataFrame]:
    base = eras.PEAD
    ds = load_dataset(base)
    signal = build_signal(ds, base)
    blend_ic, sleeve_ic = _diagnostic(ds, base)

    specs = []
    for label, overrides in SPECS.items():
        cfg = base.override(**overrides)
        r = backtest(ds, cfg, build_portfolio(ds, signal, cfg).weights)
        specs.append({"spec": label, "gross": sharpe(r.gross_return), "net": sharpe(r.net_return),
                      "ann_vol": float(r.net_return.std() * 252 ** 0.5),
                      "turnover": turnover(r), "cost_bps": cost_bps(r)})

    wf = v.walk_forward(base, ds, GRID)
    return {
        "phase2_regime_ic": blend_ic,
        "phase2_regime_sleeve_ic": sleeve_ic,
        "phase2_regime": pd.DataFrame(specs),
        "phase2_regime_folds": wf.folds,
        "phase2_regime_control": _control(ds, base, signal),
    }
