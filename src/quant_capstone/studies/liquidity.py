"""Is reversal alpha concentrated in illiquid names? Tested on S&P 1500.

Part A buckets daily IC by ADV decile next to the spread each decile pays. Part B
runs the full backtest at four universe sizes under both spread assumptions of
the time -- the "estimated" rows use the Corwin-Schultz bar estimator, which is
the result that later showed bar-based estimators fail at this liquidity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest import forward_returns
from ..costs import rolling_half_spread_bps
from ..law import bucket_summary, ic_by_bucket
from ..risk.pipeline import build_portfolio
from ..signals import build_signal
from ..reporting.metrics import sharpe
from . import eras
from .common import backtest, cost_bps, sp1500_prices, sub_universe

DESCRIPTION = "IC and spread by ADV decile on S&P 1500; backtest by universe size"


def _decile_medians(char: np.ndarray, values: np.ndarray, n_rows: int) -> tuple[dict, dict]:
    spread, adv = {}, {}
    for b in range(10):
        vs, va = [], []
        for i in range(0, n_rows, 5):
            ok = np.flatnonzero(np.isfinite(char[i]))
            if ok.size < 100:
                continue
            group = np.array_split(ok[np.argsort(char[i][ok])], 10)[b]
            vs.append(np.nanmedian(values[i][group]))
            va.append(np.nanmedian(char[i][group]))
        spread[f"Q{b + 1}"], adv[f"Q{b + 1}"] = np.nanmedian(vs), np.nanmedian(va)
    return spread, adv


def run() -> dict[str, pd.DataFrame]:
    prices = sp1500_prices(min_observations=252)
    base = eras.FLAT_SPREAD

    ds = sub_universe(prices, 1000, 252)
    cfg = base.override(data__universe_size=1000)
    signal = build_signal(ds, cfg)
    adv = prices.adv(63).where(ds.members)
    half = rolling_half_spread_bps(prices.high, prices.low, window=21, estimator="corwin_schultz")

    deciles = bucket_summary(ic_by_bucket(signal, forward_returns(prices, "next_open"), adv, n_buckets=10))
    spread, med_adv = _decile_medians(adv.to_numpy(float), half.where(ds.members).to_numpy(float), len(signal))
    deciles["half_spread_bps"] = pd.Series(spread)
    deciles["median_adv_$m"] = pd.Series(med_adv) / 1e6
    deciles["ic_per_bp_spread"] = deciles["ic_mean"] / deciles["half_spread_bps"]

    rows = []
    for universe in (100, 200, 400, 800):
        ds_u = sub_universe(prices, universe, 252)
        cfg_u = base.override(data__universe_size=universe)
        weights = build_portfolio(ds_u, build_signal(ds_u, cfg_u), cfg_u).weights
        for estimated in (False, True):
            r = backtest(ds_u, cfg_u.override(costs__use_estimated_spread=estimated,
                                              costs__spread_estimator="corwin_schultz"), weights)
            rows.append({"universe": universe, "spread": "estimated" if estimated else "flat 2bps",
                         "gross_sharpe": sharpe(r.gross_return), "net_sharpe": sharpe(r.net_return),
                         "cost_bps": cost_bps(r), "trades_day": float(r.n_trades.mean())})
    return {"liquidity_deciles": deciles, "universe_sweep": pd.DataFrame(rows)}
