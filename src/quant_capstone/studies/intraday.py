"""Hourly reversal: twenty times the breadth, and what it costs to collect.

A diagnostic rather than a second backtest engine. It measures IC and
autocorrelation on IB hourly bars, converts them to an implied Sharpe through the
Fundamental Law, and sets that against what trading at hourly frequency costs.

The cost inputs are measured on the daily book and carried over, which is
optimistic: an hourly book crosses the spread far more often.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..linalg import rank_corr
from .common import t_stat

DESCRIPTION = "Hourly reversal IC and the cost of harvesting its breadth (IB bars)"
HOURLY_CACHE = Path("data_cache/ib_hourly.parquet")

#: Measured on the daily strategy (see the transfer and cost studies).
BPS_PER_DOLLAR = 1.51  # turnover-linked cost: 40.2 bps/yr over 26.7x turnover
BORROW_BPS = 25.0      # borrow does not scale with trading frequency
BOOK_VOL_BPS = 370.0   # realized annual volatility of the book
TRANSFER_COEFFICIENT = 0.63
INDEPENDENT_FRACTION = 0.25  # independent directions per name, measured daily


def _panel() -> tuple[pd.DataFrame, float]:
    df = pd.read_parquet(HOURLY_CACHE)
    df["ts"] = pd.to_datetime(df["ts"].str.replace(r"\s+US/Eastern$", "", regex=True),
                              format="%Y%m%d %H:%M:%S", errors="coerce")
    close = df.dropna(subset=["ts"]).pivot_table(index="ts", columns="ticker", values="close").sort_index()
    bars_per_day = float(close.groupby(close.index.date).size().median())
    return close, bars_per_day


def _ic_by_horizon(close: pd.DataFrame) -> pd.DataFrame:
    ret = close.pct_change()
    ret_x = ret.sub(ret.mean(axis=1), axis=0)  # relative, market-neutral reversion
    fwd = ret_x.shift(-1).to_numpy()
    rows = []
    for k in (1, 2, 3, 6, 13):
        sig = -ret_x.rolling(k).sum()
        sig = sig.sub(sig.mean(axis=1), axis=0).div(sig.std(axis=1).replace(0, np.nan), axis=0)
        s = sig.to_numpy()
        ic = np.array([rank_corr(s[i], fwd[i]) for i in range(len(sig))])
        ic = ic[np.isfinite(ic)]
        if len(ic) < 100:
            continue
        rows.append({"k_hours": k, "ic_mean": ic.mean(), "ic_t": t_stat(ic),
                     "autocorr": float(sig.corrwith(sig.shift(1), axis=1).mean()), "n_obs": len(ic)})
    return pd.DataFrame(rows)


def run() -> dict[str, pd.DataFrame]:
    close, bars_per_day = _panel()
    table = _ic_by_horizon(close)

    best = table.loc[table["ic_t"].idxmax()]
    periods = bars_per_day * 252
    rho = float(best["autocorr"])
    breadth = INDEPENDENT_FRACTION * close.shape[1] * periods * (1 - rho) / (1 + rho)
    gross_tc1 = float(best["ic_mean"]) * np.sqrt(breadth)
    gross = gross_tc1 * TRANSFER_COEFFICIENT

    rows = []
    for frac in (0.25, 0.15, 0.10, 0.05, 0.02):
        turnover = periods * frac
        cost = turnover * BPS_PER_DOLLAR + BORROW_BPS
        rows.append({"frac_per_bar": frac, "turnover": turnover, "cost_bps": cost,
                     "implied_net": gross - cost / BOOK_VOL_BPS})
    law = pd.DataFrame([{
        "k_hours": int(best["k_hours"]), "n_names": close.shape[1], "n_bars": len(close),
        "bars_per_year": periods, "effective_breadth": breadth,
        "implied_gross_tc1": gross_tc1, "implied_gross": gross,
        "break_even_frac_per_bar": (gross * BOOK_VOL_BPS - BORROW_BPS) / BPS_PER_DOLLAR / periods,
    }])
    return {"intraday_ic": table, "intraday_costs": pd.DataFrame(rows), "intraday_law": law}
