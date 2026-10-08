"""Value, quality and investment sleeves built on point-in-time fundamentals.

These matter because of what the Fundamental Law says about our position. The
price-based blend delivers IC ~0.011 and the reversal and overnight sleeves,
being two views of the same microstructure effect, correlate at 0.29. More
sleeves of that kind add little. Accounting-based signals are driven by an
entirely different mechanism operating on a different horizon, so they are the
realistic route to genuinely independent bets — which is the only term in
``IR = IC * sqrt(BR)`` we can still move much.

Every input arrives from ``data/sec_fundamentals.py`` already lagged to its SEC
filing date, so nothing here needs a defensive lag of its own: a value is visible
on the day it became public and not before.

A caution that applies to all of them. Fundamental ratios are noisy at the
denominator: book-to-price explodes when equity approaches zero, and earnings
yield is meaningless for a company losing money. Each function handles its own
degenerate cases explicitly rather than leaning on the downstream winsorizer,
because by the time a ratio is 10,000 the cross-sectional median has already
moved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_MARKET_CAP = 1e7
"""Below this, the ratios are noise rather than signal."""


def market_cap(close_raw: pd.DataFrame, shares: pd.DataFrame) -> pd.DataFrame:
    """Shares outstanding times the unadjusted price.

    Unadjusted deliberately: shares outstanding is a raw count from the filing,
    so it must be multiplied by the price that actually printed. Using the
    split-adjusted price would misstate the cap by the cumulative split factor.
    """
    cap = close_raw * shares.reindex_like(close_raw)
    return cap.where(cap > MIN_MARKET_CAP)


def book_to_price(equity: pd.DataFrame, cap: pd.DataFrame) -> pd.DataFrame:
    """Classic value (Fama-French HML). Positive = cheap.

    Negative book equity is dropped rather than ranked. A company with negative
    equity is not "infinitely expensive"; the ratio simply stops meaning anything,
    and ranking on it would put the most distressed names at one extreme of the
    book for a reason unrelated to value.
    """
    positive_equity = equity.where(equity > 0)
    return positive_equity / cap


def earnings_yield(net_income: pd.DataFrame, cap: pd.DataFrame) -> pd.DataFrame:
    """Trailing annual earnings over market cap. Positive = cheap.

    Loss-makers are kept with a negative yield, unlike book-to-price above. That
    is deliberate and the asymmetry is meaningful: negative earnings are a real,
    interpretable signal about the business, whereas negative book equity is
    usually an accounting artefact of buybacks (Apple's equity fell from $119bn
    to $57bn over this sample while the company thrived).
    """
    return net_income / cap


def gross_profitability(
    revenue: pd.DataFrame, cost_of_revenue: pd.DataFrame, assets: pd.DataFrame
) -> pd.DataFrame:
    """(Revenue - COGS) / Assets (Novy-Marx 2013).

    Novy-Marx's point is that gross profit is the cleanest accounting measure of
    profitability: the further down the income statement you go, the more
    management discretion has been applied. Empirically it is about as powerful
    as book-to-price and, usefully here, *negatively* correlated with it —
    profitable firms are rarely cheap on book.
    """
    profit = revenue - cost_of_revenue.reindex_like(revenue)
    return profit / assets.reindex_like(revenue).where(lambda a: a > 0)


def return_on_equity(net_income: pd.DataFrame, equity: pd.DataFrame) -> pd.DataFrame:
    """Quality: earnings per unit of book equity."""
    return net_income / equity.where(equity > 0)


def asset_growth(assets: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    """Negative asset growth (Cooper, Gulen & Schill 2008). Positive = conservative.

    Firms that expand their balance sheets aggressively subsequently
    underperform — empire-building and over-investment. Sign is flipped so that,
    like every other sleeve here, a higher score means buy.
    """
    growth = assets / assets.shift(lookback) - 1.0
    return -growth.replace([np.inf, -np.inf], np.nan)


def accruals(
    net_income: pd.DataFrame, operating_income: pd.DataFrame, assets: pd.DataFrame
) -> pd.DataFrame:
    """Low accruals are higher quality (Sloan 1996).

    Earnings that are not backed by operating performance tend to reverse. This
    is a crude proxy for the real Sloan measure (which needs a full cash-flow
    statement); it uses the gap between net and operating income, scaled by
    assets, and the sign is flipped so higher is better.
    """
    gap = (net_income - operating_income.reindex_like(net_income)) / assets.reindex_like(
        net_income
    ).where(lambda a: a > 0)
    return -gap


#: Fundamental sleeves, by name. Each takes the panels dict plus price data.
FUNDAMENTAL_ALPHAS = {
    "book_to_price": lambda p, cap, **_: book_to_price(p["equity"], cap),
    "earnings_yield": lambda p, cap, **_: earnings_yield(p["net_income"], cap),
    "gross_profitability": lambda p, cap, **_: gross_profitability(
        p["revenue"], p["cost_of_revenue"], p["assets"]
    ),
    "return_on_equity": lambda p, cap, **_: return_on_equity(p["net_income"], p["equity"]),
    "asset_growth": lambda p, cap, **_: asset_growth(p["assets"]),
    "accruals": lambda p, cap, **_: accruals(
        p["net_income"], p["operating_income"], p["assets"]
    ),
}

#: Concepts each sleeve needs, so coverage can be checked before building it.
FUNDAMENTAL_REQUIRES = {
    "book_to_price": ("equity", "shares"),
    "earnings_yield": ("net_income", "shares"),
    "gross_profitability": ("revenue", "cost_of_revenue", "assets"),
    "return_on_equity": ("net_income", "equity"),
    "asset_growth": ("assets",),
    "accruals": ("net_income", "operating_income", "assets"),
}


def standardized_unexpected_earnings(
    quarterly_income: pd.DataFrame, lookback: int = 8 * 63, decay_days: int = 60
) -> pd.DataFrame:
    """Post-earnings announcement drift (Bernard & Thomas 1989).

    The one fundamental signal with a plausible route around the breadth problem
    that sinks the others. Static ratios like book-to-price sit at an
    autocorrelation of 0.999 because they only move when a filing lands and then
    stay put. An earnings *surprise* is different: it is an event with its own
    clock, it decays over the weeks after the announcement, and fiscal calendars
    are staggered across names — so the cross-section keeps turning over even
    though each individual company reports four times a year.

    The surprise is the seasonal random-walk form: this quarter's earnings minus
    the same quarter a year ago, scaled by the volatility of that difference.
    Year-on-year rather than quarter-on-quarter because most businesses have
    seasonality that would otherwise swamp the signal.

    ``decay_days`` linearly fades the score to zero after the announcement.
    Bernard-Thomas put the drift at roughly a quarter, and fading rather than
    holding keeps the signal from behaving like yet another static ratio.
    """
    income = quarterly_income
    n_days = len(income)
    out = pd.DataFrame(np.nan, index=income.index, columns=income.columns)

    for column in income.columns:
        series = income[column]
        # A filing is a row where the forward-filled value actually changes.
        is_event = series.notna() & series.ne(series.shift(1))
        positions = np.flatnonzero(is_event.to_numpy())
        if len(positions) < 6:
            continue

        reported = series.to_numpy()[positions]
        delta = np.full(len(reported), np.nan)
        delta[4:] = reported[4:] - reported[:-4]      # four filings back = a year
        scale = pd.Series(delta).rolling(8, min_periods=4).std().to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            sue = np.where(scale > 0, delta / scale, np.nan)

        column_values = np.full(n_days, np.nan)
        for i, start_row in enumerate(positions):
            if not np.isfinite(sue[i]):
                continue
            stop_row = min(start_row + decay_days, n_days)
            fade = 1.0 - np.arange(stop_row - start_row) / decay_days
            # A later announcement supersedes the one still decaying.
            column_values[start_row:stop_row] = sue[i] * fade
        out[column] = column_values

    return out


FUNDAMENTAL_ALPHAS["earnings_surprise"] = lambda p, cap, **_: (
    standardized_unexpected_earnings(p["net_income_q"])
)
FUNDAMENTAL_REQUIRES["earnings_surprise"] = ("net_income_q",)
