"""A library of cross-sectional alpha sleeves.

One signal at IC ~0.014 cannot reach a Sharpe of 1.4: the Fundamental Law caps it
near 0.6 gross at our breadth. Combining weakly-correlated sleeves can, because
IR grows with the square root of the number of *independent* bets — which is how
multi-strategy books actually get there, rather than by finding one great signal.

Two design rules the sleeves here follow.

**Every sleeve is a published effect with an economic story.** Not a grid search
over transformations. Each docstring names the mechanism and the paper, because a
sleeve we cannot explain is a sleeve we cannot trust out of sample.

**Turnover is part of the design, not an afterthought.** Reversal decays in days
and costs a fortune to run; momentum and low-volatility decay in months and are
nearly free. Blending them dilutes the cost burden per unit of alpha, which on
this book matters more than the raw IC of any one sleeve.

Point-in-time discipline: every function returns a score for day t computed from
data through the day-t close and nothing later. Execution lag is applied once,
downstream, in the backtest engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def reversal(close: pd.DataFrame, k: int = 2) -> pd.DataFrame:
    """Short-term reversal: buy recent losers (Lehmann 1990; Lo-MacKinlay 1990).

    Mechanism: a dealer absorbing a liquidity-motivated order moves the price and
    is compensated when it reverts. We are being paid to provide that liquidity.
    Decays within days, so it is the expensive sleeve to run.
    """
    return -close.pct_change(k)


def momentum(close: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """12-month momentum skipping the most recent month (Jegadeesh-Titman 1993).

    Mechanism: under-reaction to information; winners keep winning over 3-12
    month horizons. The one-month skip is not decoration -- the most recent month
    carries the short-term *reversal* effect, and including it makes momentum
    fight the reversal sleeve instead of complementing it.

    Turnover is low: the ranking barely moves day to day, so this sleeve adds
    alpha almost for free in cost terms.
    """
    return close.shift(skip) / close.shift(lookback) - 1.0


def residual_momentum(
    returns: pd.DataFrame, beta: pd.DataFrame, market: pd.Series,
    lookback: int = 252, skip: int = 21,
) -> pd.DataFrame:
    """Momentum of market-adjusted returns (Blitz, Huij & Martens 2011).

    Plain momentum accumulates a market and sector bet: a 12-month winner is
    often just a high-beta name in a rising market. Stripping the market
    component first gives a cleaner, more stable ranking, and one our
    beta-neutral book can actually express.
    """
    residual = returns.sub(beta.mul(market, axis=0), fill_value=0.0)
    cumulative = residual.rolling(lookback - skip, min_periods=(lookback - skip) // 2).sum()
    return cumulative.shift(skip)


def low_volatility(returns: pd.DataFrame, lookback: int = 126) -> pd.DataFrame:
    """Long low-volatility names, short high-volatility ones (Frazzini-Pedersen 2014).

    Mechanism: leverage-constrained investors reach for return by buying volatile
    stocks instead of levering safe ones, so high-vol names are persistently
    overpriced. Extremely low turnover -- realized volatility is a slow-moving
    characteristic.
    """
    return -returns.rolling(lookback, min_periods=lookback // 2).std()


def idiosyncratic_volatility(
    returns: pd.DataFrame, beta: pd.DataFrame, market: pd.Series, lookback: int = 126
) -> pd.DataFrame:
    """Low *idiosyncratic* volatility (Ang, Hodrick, Xing & Zhang 2006).

    The residual-risk cousin of ``low_volatility``. Kept separate because a
    beta-neutral book has already hedged the systematic part, so the residual
    version is the one it can actually harvest.
    """
    residual = returns.sub(beta.mul(market, axis=0), fill_value=0.0)
    return -residual.rolling(lookback, min_periods=lookback // 2).std()


def volume_shock_reversal(
    close: pd.DataFrame, volume: pd.DataFrame, k: int = 2,
    baseline: int = 63, cap: float = 5.0,
) -> pd.DataFrame:
    """Reversal, scaled by how unusual the volume behind the move was
    (Campbell, Grossman & Wang 1993).

    The distinction that makes this more than a re-parameterised reversal: a
    price move on heavy volume is usually someone *demanding liquidity*, and it
    reverts. The same move on light volume is more likely to be *information*,
    and it does not. Scaling the reversal score by relative volume tilts toward
    the revertible half.

    The ratio is capped because a single halt-and-reopen day can otherwise print
    a volume multiple of fifty and dominate the entire cross-section.
    """
    recent = volume.rolling(k, min_periods=1).mean()
    normal = volume.rolling(baseline, min_periods=baseline // 2).mean()
    ratio = (recent / normal.replace(0.0, np.nan)).clip(upper=cap)
    return -close.pct_change(k) * ratio


def overnight_reversal(
    open_: pd.DataFrame, close: pd.DataFrame, k: int = 5
) -> pd.DataFrame:
    """Reverse the overnight component of recent returns (Lou, Polk & Skouras 2019).

    Overnight and intraday returns have opposite predictability: the overnight
    session is thin, order-driven and prone to over-shooting, and it reverts
    during the following day. Isolating that component is a genuinely different
    bet from total-return reversal, which is what makes it worth its own sleeve.
    """
    overnight = open_ / close.shift(1) - 1.0
    return -overnight.rolling(k, min_periods=max(k // 2, 1)).sum()


#: Registry of sleeves. ``build`` dispatches on these names.
ALPHAS = {
    "reversal": reversal,
    "momentum": momentum,
    "residual_momentum": residual_momentum,
    "low_volatility": low_volatility,
    "idiosyncratic_volatility": idiosyncratic_volatility,
    "volume_shock_reversal": volume_shock_reversal,
    "overnight_reversal": overnight_reversal,
}

#: Roughly how many bars of history each sleeve needs before it produces a score.
WARMUP = {
    "reversal": 5,
    "momentum": 273,
    "residual_momentum": 273,
    "low_volatility": 126,
    "idiosyncratic_volatility": 126,
    "volume_shock_reversal": 63,
    "overnight_reversal": 10,
}
