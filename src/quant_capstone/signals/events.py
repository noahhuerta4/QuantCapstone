"""Registry of event-driven sleeves and the data each one needs.

Kept separate from ``fundamental.py`` because these signals are a different
species. An accounting ratio is a *level* computed from a panel; an event sleeve
needs the underlying record of what happened and when it was disclosed, and has
to build its own clock from that. The registry therefore hands each builder the
whole dataset rather than a dict of pre-made panels.

The candidate list is deliberately short and fixed in advance. Every sleeve
tested counts against the Deflated Sharpe whether it is adopted or rejected, so
a wide search is self-defeating: the honest way to buy Sharpe here is new
information, not more configurations over the same data. Five candidates, each
with a stated prior reason to work.
"""

from __future__ import annotations

import logging

import pandas as pd

from .earnings_extra import announcement_reversal, revenue_surprise
from .insider import insider_flow_fast, net_buy_count, officer_filter

log = logging.getLogger(__name__)


def _market_cap(dataset) -> pd.DataFrame:
    from .fundamental import market_cap

    return market_cap(dataset.prices.close_raw, dataset.fundamentals["shares"])


def _flow_panel(dataset, trades: pd.DataFrame) -> pd.DataFrame:
    from ..data.edgar_form4 import to_panel

    return to_panel(trades, dataset.prices.dates, dataset.prices.tickers)


# ------------------------------------------------------------------ builders
def _insider_all(dataset, cfg) -> pd.DataFrame:
    """Decayed net insider dollar flow over market cap, every open-market trade."""
    return insider_flow_fast(
        _flow_panel(dataset, dataset.form4), _market_cap(dataset), cfg.signal.insider_halflife
    )


def _insider_opportunistic(dataset, cfg) -> pd.DataFrame:
    """Same, restricted to trades that break the insider's own calendar pattern.

    Cohen, Malloy & Pomorski (2012): the routine majority is diversification and
    dilutes whatever signal the discretionary minority carries.
    """
    from ..data.edgar_form4 import classify_routine

    trades = classify_routine(dataset.form4)
    return insider_flow_fast(
        _flow_panel(dataset, trades[trades["opportunistic"]]),
        _market_cap(dataset), cfg.signal.insider_halflife,
    )


def _insider_purchases(dataset, cfg) -> pd.DataFrame:
    """Purchases only, by senior officers and directors.

    Motivated by the data rather than by search: the sample runs 13 sales for
    every purchase, because executives are compensated in stock and sell to
    diversify, pay tax and fund life. A sale carries many explanations; a
    purchase has essentially one. Lakonishok & Lee (2001) find the predictive
    content sits on the buy side, and a net-flow measure lets the noisy side
    dominate by sheer count.
    """
    trades = officer_filter(dataset.form4)
    purchases = trades[trades["code"] == "P"]
    return insider_flow_fast(
        _flow_panel(dataset, purchases), _market_cap(dataset), cfg.signal.insider_halflife
    )


def _insider_breadth(dataset, cfg) -> pd.DataFrame:
    """Decayed count of buyers minus sellers, ignoring trade size.

    Five executives each buying modestly is stronger evidence than one buying
    heavily, and a dollar-weighted measure cannot express that.
    """
    return net_buy_count(
        dataset.form4, dataset.prices.dates, dataset.prices.tickers, cfg.signal.insider_halflife
    )


def _revenue_surprise(dataset, cfg) -> pd.DataFrame:
    return revenue_surprise(dataset.fundamentals["revenue_q"])


def _announcement_reversal(dataset, cfg) -> pd.DataFrame:
    return announcement_reversal(
        dataset.prices.close,
        dataset.fundamentals["net_income_q"],
        cfg.signal.announcement_window,
        cfg.signal.announcement_decay,
    )


#: name -> builder. Each takes (dataset, cfg) and returns a raw score panel.
EVENT_ALPHAS = {
    "insider_flow": _insider_all,
    "insider_opportunistic": _insider_opportunistic,
    "insider_purchases": _insider_purchases,
    "insider_breadth": _insider_breadth,
    "revenue_surprise": _revenue_surprise,
    "announcement_reversal": _announcement_reversal,
}

#: What each sleeve needs loaded. ``form4`` means the Form 4 trade table;
#: everything else is an XBRL concept panel.
EVENT_REQUIRES = {
    "insider_flow": ("form4", "shares"),
    "insider_opportunistic": ("form4", "shares"),
    "insider_purchases": ("form4", "shares"),
    "insider_breadth": ("form4",),
    "revenue_surprise": ("revenue_q",),
    "announcement_reversal": ("net_income_q",),
}

INSIDER_ALPHAS = {n for n, r in EVENT_REQUIRES.items() if "form4" in r}


def needs_form4(alphas) -> bool:
    return any(name in INSIDER_ALPHAS for name in alphas)


def needs_fundamentals(alphas) -> bool:
    from .fundamental import FUNDAMENTAL_ALPHAS

    if any(name in FUNDAMENTAL_ALPHAS for name in alphas):
        return True
    return any(
        any(req != "form4" for req in EVENT_REQUIRES[name])
        for name in alphas
        if name in EVENT_REQUIRES
    )
