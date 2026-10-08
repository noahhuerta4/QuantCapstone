"""Insider-flow sleeve from SEC Form 4 filings.

The candidate with the best prior odds of the whole extension, for a reason that
follows directly from the Phase-1 diagnosis. Insider filings arrive *continuously*
across the universe — a Form 4 is due within two business days of the trade, and
hundreds of companies have insiders trading in any given week — so the
cross-section keeps turning over. That is the breadth property that static
quarterly ratios lacked and the reason they contributed nothing however good
their forecasting power.

It is also new information rather than a re-parameterisation of prices, which
matters because the Deflated Sharpe discounts search but not evidence.

Three design decisions worth stating:

**Filing date, not transaction date.** The trade happens up to two business days
before the market is told. Keying off the transaction date would be a small but
real look-ahead.

**Normalize by market capitalization.** A $2m purchase means something different
at a $200m company than at a $200bn one. Raw dollar flow would rank the sleeve by
company size.

**Separate opportunistic from routine.** An insider who sells every February is
diversifying on a schedule, not expressing a view. Cohen, Malloy & Pomorski
(2012) find the predictive content sits almost entirely in the trades that break
an insider's own pattern, and the routine majority dilutes it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _decay_weights(length: int, halflife: float) -> np.ndarray:
    return 0.5 ** (np.arange(length) / halflife)


def insider_flow(
    flow_panel: pd.DataFrame,
    market_cap: pd.DataFrame,
    window: int = 60,
    halflife: float = 20.0,
) -> pd.DataFrame:
    """Exponentially-decayed net insider buying, scaled by market cap.

    ``flow_panel`` is signed dollar flow per name per filing date (positive =
    net buying), already keyed on the filing date by
    ``data/edgar_form4.to_panel``.

    The decay is what gives the sleeve a usable horizon. A filing is a point
    event; without decay the signal would be a step function that jumps once and
    holds, which is the static-ratio failure mode again. A 20-day half-life over
    a 60-day window keeps the cross-section moving while letting a cluster of
    purchases accumulate.
    """
    weights = _decay_weights(window, halflife)
    # Reverse-weighted rolling sum: today's filing gets weight 1, a filing
    # `window` days ago gets the smallest weight.
    decayed = flow_panel.rolling(window, min_periods=1).apply(
        lambda x: float(np.dot(x[::-1], weights[: len(x)])), raw=True
    )
    scale = market_cap.reindex_like(decayed)
    return decayed / scale.where(scale > 0)


def insider_flow_fast(
    flow_panel: pd.DataFrame,
    market_cap: pd.DataFrame,
    halflife: float = 20.0,
) -> pd.DataFrame:
    """Same signal via an EWM recursion instead of a rolling dot product.

    Mathematically an infinite-window version of :func:`insider_flow` and roughly
    two orders of magnitude faster, which matters on a 2,516 x 100 panel. This is
    the one the pipeline uses; the windowed form above is kept because it is
    easier to read and is what the test checks against.
    """
    decayed = flow_panel.ewm(halflife=halflife, adjust=False).mean() * halflife
    scale = market_cap.reindex_like(decayed)
    return decayed / scale.where(scale > 0)


def net_buy_count(
    trades: pd.DataFrame, dates: pd.DatetimeIndex, tickers: pd.Index, halflife: float = 20.0
) -> pd.DataFrame:
    """Decayed count of buyers minus sellers, ignoring trade size.

    A deliberately different view of the same filings: five separate executives
    each buying a little is stronger evidence than one buying a lot, and a
    dollar-weighted measure cannot see that distinction. Kept as a separate
    candidate rather than blended in, so the gate can judge it on its own.
    """
    if trades.empty:
        return pd.DataFrame(0.0, index=dates, columns=tickers)

    signed = np.sign(trades["shares"])
    counts = (
        signed.groupby([trades["filing_date"].dt.normalize(), trades["ticker"]])
        .sum()
        .unstack(fill_value=0.0)
        .reindex(index=dates, columns=tickers, fill_value=0.0)
        .fillna(0.0)
    )
    return counts.ewm(halflife=halflife, adjust=False).mean()


def officer_filter(trades: pd.DataFrame) -> pd.DataFrame:
    """Keep trades by senior officers and directors.

    Seniority proxies for information: a chief executive or finance chief sees
    the whole business, a divisional officer sees part of it. The relationship
    field is a free-text mess across filers, so this matches on substrings and is
    inclusive rather than precise — the aim is to drop clearly peripheral filers,
    not to build an org chart.
    """
    text = (trades["relationship"].fillna("") + " " + trades["title"].fillna("")).str.lower()
    senior = text.str.contains(
        r"chief exec|chief financ|\bceo\b|\bcfo\b|president|chairman|director|officer",
        regex=True, na=False,
    )
    return trades[senior]


def insider_diagnostics(flow_panel: pd.DataFrame, members: pd.DataFrame) -> pd.Series:
    """Does the event clock tick often enough for this sleeve to have breadth?"""
    active = (flow_panel != 0).where(members.reindex_like(flow_panel).fillna(False))
    per_day = active.sum(axis=1)
    return pd.Series(
        {
            "mean_names_filing_per_day": float(per_day.mean()),
            "median_names_filing_per_day": float(per_day.median()),
            "days_with_any_filing": float((per_day > 0).mean()),
            "max_names_one_day": float(per_day.max()),
        }
    )
