"""The k-day reversal signal (spec S4).

``close.pct_change(k)`` evaluated at row t spans close[t-k] .. close[t], so the
signal on day t uses data through the day-t close and nothing after it. Execution
timing (t+1 open by default) is handled downstream in the backtest engine, not
here — this module never shifts anything.
"""

from __future__ import annotations

import pandas as pd


def trailing_return(close: pd.DataFrame, k: int) -> pd.DataFrame:
    """Simple k-day return per name, as of the day-t close."""
    if k < 1:
        raise ValueError("k must be >= 1")
    return close.pct_change(k)


def reversal_signal(close: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Raw reversal score: positive means 'recent loser, buy it'.

    The sign flip is the whole strategy — we are betting that the cross-sectional
    dispersion of k-day returns is partly liquidity-provision noise that reverts,
    not information that persists.
    """
    return -trailing_return(close, k)


def signal_autocorrelation(signal: pd.DataFrame, lags: int = 10) -> pd.Series:
    """Mean cross-sectional autocorrelation of the score at each lag.

    Feeds the effective-breadth calculation: a signal with lag-1 autocorrelation
    of 0.8 is not delivering a fresh independent bet every day, whatever the
    naive ``BR = 252*N`` count claims.
    """
    out = {}
    for lag in range(1, lags + 1):
        lagged = signal.shift(lag)
        out[lag] = signal.corrwith(lagged, axis=1).mean()
    return pd.Series(out, name="signal_autocorr").rename_axis("lag")
