"""Performance metrics (spec S11).

Risk-free rate is taken as zero throughout. For a dollar-neutral book that is
close to right by construction: the long leg is funded by the short proceeds, so
the return series is already an excess return. It does mean these Sharpes are not
directly comparable to a long-only strategy's, which is worth one line in the
write-up.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TRADING_DAYS


def _clean(returns: pd.Series) -> pd.Series:
    return returns.replace([np.inf, -np.inf], np.nan).dropna()


def _dispersion(returns: pd.Series) -> float:
    """Sample standard deviation, or NaN if the series is effectively constant.

    A literal ``std == 0`` test is not enough: the sample deviation of a constant
    series comes out around 1e-19 rather than exactly zero, and dividing by that
    reports a Sharpe of 1e16 instead of admitting there is no variation to
    measure. Scale the tolerance to the data.
    """
    if len(returns) < 2:
        return np.nan
    std = float(returns.std(ddof=1))
    if not np.isfinite(std) or std <= float(returns.abs().mean()) * 1e-12:
        return np.nan
    return std


def annualized_return(returns: pd.Series) -> float:
    """Geometric (CAGR), the number an investor would actually have earned."""
    r = _clean(returns)
    if r.empty:
        return np.nan
    total = float((1.0 + r).prod())
    if total <= 0:
        return -1.0
    return total ** (TRADING_DAYS / len(r)) - 1.0


def annualized_volatility(returns: pd.Series) -> float:
    r = _clean(returns)
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(r) > 1 else np.nan


def sharpe(returns: pd.Series) -> float:
    r = _clean(returns)
    std = _dispersion(r)
    if not np.isfinite(std):
        return np.nan
    return float(r.mean() / std * np.sqrt(TRADING_DAYS))


def sortino(returns: pd.Series) -> float:
    r = _clean(returns)
    downside = r[r < 0]
    if len(r) < 2 or downside.empty:
        return np.nan
    # Downside deviation is taken about zero, not about the mean, so a strategy
    # is not rewarded for having a high mean when measuring its downside.
    dd = float(np.sqrt((downside**2).mean()))
    return float(r.mean() / dd * np.sqrt(TRADING_DAYS)) if dd > 0 else np.nan


def equity_curve(returns: pd.Series) -> pd.Series:
    return (1.0 + _clean(returns)).cumprod()


def drawdown_series(returns: pd.Series) -> pd.Series:
    curve = equity_curve(returns)
    return curve / curve.cummax() - 1.0


def max_drawdown(returns: pd.Series) -> float:
    dd = drawdown_series(returns)
    return float(dd.min()) if not dd.empty else np.nan


def drawdown_duration(returns: pd.Series) -> dict[str, float]:
    """Depth, and how long the worst drawdown lasted (spec S11).

    Duration matters more than depth for a capstone: a 12% drawdown that recovers
    in three weeks is noise, and the same 12% taking two years to recover is the
    thing that gets a strategy shut down.
    """
    curve = equity_curve(returns)
    if curve.empty:
        return {"max_drawdown": np.nan, "max_drawdown_days": np.nan, "longest_underwater_days": np.nan}

    peaks = curve.cummax()
    underwater = curve < peaks

    trough = (curve / peaks - 1.0).idxmin()
    peak_before = curve.loc[:trough].idxmax()
    after = curve.loc[trough:]
    recovered = after[after >= curve.loc[peak_before]]
    recovery = recovered.index[0] if len(recovered) else curve.index[-1]

    # Longest unbroken underwater stretch anywhere in the sample.
    group = (~underwater).cumsum()
    spells = underwater.groupby(group).sum()
    return {
        "max_drawdown": float((curve / peaks - 1.0).min()),
        "max_drawdown_days": float(len(curve.loc[peak_before:recovery]) - 1),
        "longest_underwater_days": float(spells.max()) if len(spells) else 0.0,
    }


def calmar(returns: pd.Series) -> float:
    mdd = max_drawdown(returns)
    if not np.isfinite(mdd) or mdd == 0:
        return np.nan
    return float(annualized_return(returns) / abs(mdd))


def hit_rate(returns: pd.Series) -> float:
    r = _clean(returns)
    nonzero = r[r != 0]
    return float((nonzero > 0).mean()) if len(nonzero) else np.nan


def win_loss(returns: pd.Series) -> dict[str, float]:
    r = _clean(returns)
    wins, losses = r[r > 0], r[r < 0]
    return {
        "avg_win": float(wins.mean()) if len(wins) else np.nan,
        "avg_loss": float(losses.mean()) if len(losses) else np.nan,
        "win_loss_ratio": float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) and losses.mean() != 0 else np.nan,
    }


def value_at_risk(returns: pd.Series, level: float = 0.05, method: str = "historical") -> float:
    """Daily VaR as a positive loss number."""
    r = _clean(returns)
    if r.empty:
        return np.nan
    if method == "historical":
        return float(-np.quantile(r, level))
    if method == "parametric":
        from scipy.stats import norm

        return float(-(r.mean() + norm.ppf(level) * r.std(ddof=1)))
    raise ValueError(f"unknown VaR method {method!r}")


def conditional_var(returns: pd.Series, level: float = 0.05) -> float:
    """Historical CVaR: mean loss in the worst ``level`` tail."""
    r = _clean(returns)
    if r.empty:
        return np.nan
    cutoff = np.quantile(r, level)
    tail = r[r <= cutoff]
    return float(-tail.mean()) if len(tail) else np.nan


def rolling_sharpe(returns: pd.Series, window: int = 126) -> pd.Series:
    """Rolling annualized Sharpe. Default window is ~6 months (spec S11)."""
    r = _clean(returns)
    mean = r.rolling(window, min_periods=window // 2).mean()
    std = r.rolling(window, min_periods=window // 2).std(ddof=1)
    return (mean / std.replace(0.0, np.nan) * np.sqrt(TRADING_DAYS)).rename(f"rolling_sharpe_{window}d")


def summarize(returns: pd.Series, name: str = "") -> pd.Series:
    """The full metric block for one return series."""
    stats = {
        "sharpe": sharpe(returns),
        "sortino": sortino(returns),
        "calmar": calmar(returns),
        "ann_return": annualized_return(returns),
        "ann_vol": annualized_volatility(returns),
        **drawdown_duration(returns),
        "hit_rate": hit_rate(returns),
        **win_loss(returns),
        "var_95_hist": value_at_risk(returns, 0.05, "historical"),
        "var_95_param": value_at_risk(returns, 0.05, "parametric"),
        "cvar_95": conditional_var(returns, 0.05),
        "skew": float(_clean(returns).skew()),
        "kurtosis": float(_clean(returns).kurtosis()),
        "n_days": float(len(_clean(returns))),
    }
    return pd.Series(stats, name=name or "metrics")
