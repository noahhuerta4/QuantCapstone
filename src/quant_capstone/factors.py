"""Market proxy, rolling betas and sector loadings.

Used in two places for two purposes: to orthogonalize the raw signal (S4) and to
impose beta/sector neutrality on the weights (S5). Same estimates either way.

Point-in-time note: ``returns[t]`` is known at the close of day t, and a signal
formed on day t is allowed to use data through the t close, so a beta estimated
on a window ending at t is legitimate — no shift is applied or needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def market_return(returns: pd.DataFrame, members: pd.DataFrame | None = None) -> pd.Series:
    """Equal-weighted universe return as the market proxy.

    Deliberately not an index ETF: the book is neutral against *the universe it
    trades*, and an equal-weight universe average is the portfolio a zero-signal
    version of this strategy would hold.
    """
    masked = returns.where(members) if members is not None else returns
    return masked.mean(axis=1).rename("market")


def rolling_beta(
    returns: pd.DataFrame, market: pd.Series, lookback: int = 126, min_periods: int | None = None
) -> pd.DataFrame:
    """Rolling univariate beta of each name against the market proxy."""
    min_periods = min_periods or max(lookback // 2, 20)
    mkt = market.reindex(returns.index)
    prod = returns.mul(mkt, axis=0)

    mean_prod = prod.rolling(lookback, min_periods=min_periods).mean()
    mean_r = returns.rolling(lookback, min_periods=min_periods).mean()
    mean_m = mkt.rolling(lookback, min_periods=min_periods).mean()
    var_m = mkt.rolling(lookback, min_periods=min_periods).var(ddof=0)

    cov = mean_prod.sub(mean_r.mul(mean_m, axis=0))
    beta = cov.div(var_m.replace(0.0, np.nan), axis=0)
    # A name with too little history gets beta 1.0 rather than being dropped:
    # excluding it would silently shrink the universe, which distorts breadth.
    return beta.clip(-3.0, 3.0).fillna(1.0)


def sector_dummies(sectors: pd.Series, tickers: pd.Index) -> pd.DataFrame:
    """One-hot ``ticker x sector`` loading matrix."""
    labels = sectors.reindex(tickers).fillna("Unknown")
    return pd.get_dummies(labels, dtype=float).reindex(index=tickers).fillna(0.0)


def realized_beta(portfolio_returns: pd.Series, market: pd.Series) -> float:
    """Ex-post beta of the book. Should come out ~0 if neutralization works."""
    joined = pd.concat([portfolio_returns, market], axis=1).dropna()
    if len(joined) < 3 or joined.iloc[:, 1].std() == 0:
        return np.nan
    cov = joined.cov().iloc[0, 1]
    return float(cov / joined.iloc[:, 1].var())
