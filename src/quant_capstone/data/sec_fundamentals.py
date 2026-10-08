"""Point-in-time fundamentals from the SEC's XBRL API.

Why this source, after two others failed. IB's fundamental data needs a paid
Reuters subscription (error 10358 without it). yfinance returns only five to
seven quarters, nowhere near a ten-year backtest. The SEC's XBRL API is free,
official, goes back to ~2009, and — the part that actually matters — stamps every
fact with the date it was **filed**.

That filing date is the whole point. Accounting data is the classic source of
look-ahead bias in equity backtests: Q4 figures with a period end of December 31
are not public until the 10-K lands in February, so ranking stocks on them in
January is trading on information nobody had. The usual workaround is a blanket
six-month lag (Fama-French do this), which is safe but throws away months of
genuinely available information. With ``filed`` we do not have to guess: a fact
becomes visible on the exact day it was filed, and not before.

Restatements fall out correctly for free. A company that refiles 2019 numbers in
2021 produces a second fact for the same period with a later ``filed`` date, so
asking "what was the latest filed value as of date t" returns the figure the
market actually had at t, not the corrected one it learned later.

Usage note: the SEC asks for a descriptive User-Agent and no more than ten
requests a second. Both are respected below. Set ``SEC_USER_AGENT`` to your own
contact string if you run this at volume — the default is deliberately generic
rather than carrying anyone's personal address.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

SEC_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT", "quant-capstone academic research (contact via project repository)"
)
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
REQUESTS_PER_SECOND = 8.0

#: Concepts we need, with fallbacks: XBRL tag names changed over the years and no
#: single tag covers a decade for every filer.
CONCEPTS: dict[str, tuple[str, ...]] = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "SalesRevenueServicesNet",
    ),
    "cost_of_revenue": (
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
        "CostOfServices",
    ),
    "net_income": ("NetIncomeLoss",),
    "operating_income": ("OperatingIncomeLoss",),
    "assets": ("Assets",),
    "equity": (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ),
    "shares": ("CommonStockSharesOutstanding", "CommonStockSharesIssued"),
}

#: Which concepts are point-in-time stocks (balance sheet) vs annual flows.
INSTANT = {"assets", "equity", "shares"}
ANNUAL_MIN_DAYS, ANNUAL_MAX_DAYS = 350, 380
QUARTER_MIN_DAYS, QUARTER_MAX_DAYS = 80, 100

#: Concepts also extracted at quarterly frequency, for earnings-surprise signals.
#: Quarterly matters because a *change* in earnings is an event with its own
#: clock: staggered fiscal calendars mean the cross-section keeps moving even
#: though each name only reports four times a year, which is the breadth that
#: static quarterly ratios conspicuously lack.
QUARTERLY_CONCEPTS = ("net_income", "revenue")


def _session():
    import requests

    session = requests.Session()
    session.headers.update({"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"})
    return session


def ticker_to_cik(cache_dir: Path) -> pd.Series:
    """Ticker -> CIK, cached."""
    cache = Path(cache_dir) / "sec_ticker_cik.parquet"
    if cache.exists():
        return pd.read_parquet(cache)["cik"]

    payload = _session().get(TICKER_MAP_URL, timeout=30).json()
    frame = pd.DataFrame(payload.values())
    # ``.to_numpy()`` is load-bearing: passing the Series directly makes pandas
    # align its RangeIndex against the ticker index and silently yield all-NaN.
    out = pd.DataFrame(
        {"cik": frame["cik_str"].astype(int).to_numpy()},
        index=pd.Index(
            frame["ticker"].astype(str).str.replace(".", "-", regex=False).to_numpy(),
            name="ticker",
        ),
    )
    out = out[~out.index.duplicated(keep="first")]
    out.to_parquet(cache)
    return out["cik"]


def _extract(
    facts: dict, tags: tuple[str, ...], instant: bool, period: str = "annual"
) -> list[tuple[str, float]]:
    """(filed_date, value) pairs for the first tag that has usable data.

    ``period`` selects the duration window for flow items: ``annual`` keeps
    ~365-day facts, ``quarterly`` keeps ~90-day ones. The narrow quarterly window
    is deliberate -- many filers tag year-to-date spans too (Q2 at ~180 days,
    Q3 at ~270), and mixing those with discrete quarters would make a
    year-on-year earnings comparison meaningless.
    """
    lo, hi = (
        (QUARTER_MIN_DAYS, QUARTER_MAX_DAYS) if period == "quarterly"
        else (ANNUAL_MIN_DAYS, ANNUAL_MAX_DAYS)
    )
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    for tag in tags:
        entry = us_gaap.get(tag)
        if not entry:
            continue
        rows: list[tuple[str, str, float]] = []
        for unit_facts in entry.get("units", {}).values():
            for fact in unit_facts:
                filed, end, val = fact.get("filed"), fact.get("end"), fact.get("val")
                if filed is None or end is None or val is None:
                    continue
                if instant:
                    if fact.get("start") is not None:
                        continue
                else:
                    start = fact.get("start")
                    if start is None:
                        continue
                    span = (pd.Timestamp(end) - pd.Timestamp(start)).days
                    if not lo <= span <= hi:
                        continue
                rows.append((filed, end, float(val)))
        if rows:
            # One value per filing date: the one covering the most recent period.
            frame = pd.DataFrame(rows, columns=["filed", "end", "val"]).sort_values(["filed", "end"])
            frame = frame.drop_duplicates("filed", keep="last")
            return list(zip(frame["filed"], frame["val"]))
    return []


def fetch_company_facts(tickers: list[str], cache_dir: Path) -> pd.DataFrame:
    """Long frame of (ticker, concept, filed, value) for every requested ticker.

    Cached as one parquet; only missing tickers are fetched on a re-run.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / "sec_fundamentals.parquet"

    existing = pd.read_parquet(cache) if cache.exists() else pd.DataFrame(
        columns=["ticker", "concept", "filed", "value"]
    )
    have = set(existing["ticker"].unique())
    todo = [t for t in tickers if t not in have]
    if not todo:
        return existing

    cik_map = ticker_to_cik(cache_dir)
    session = _session()
    interval = 1.0 / REQUESTS_PER_SECOND
    collected: list[dict] = []

    log.info("fetching SEC fundamentals for %d tickers", len(todo))
    for i, ticker in enumerate(todo, 1):
        cik = cik_map.get(ticker)
        if cik is None or not np.isfinite(cik):
            continue
        started = time.monotonic()
        try:
            response = session.get(COMPANY_FACTS_URL.format(cik=int(cik)), timeout=30)
            if response.status_code != 200:
                continue
            facts = response.json()
        except Exception as exc:
            log.debug("SEC fetch failed for %s: %s", ticker, exc)
            continue
        finally:
            time.sleep(max(0.0, interval - (time.monotonic() - started)))

        for concept, tags in CONCEPTS.items():
            for filed, value in _extract(facts, tags, concept in INSTANT):
                collected.append(
                    {"ticker": ticker, "concept": concept, "filed": filed, "value": value}
                )
        for concept in QUARTERLY_CONCEPTS:
            for filed, value in _extract(facts, CONCEPTS[concept], False, "quarterly"):
                collected.append(
                    {"ticker": ticker, "concept": f"{concept}_q", "filed": filed, "value": value}
                )
        if i % 100 == 0:
            log.info("  %d/%d tickers", i, len(todo))

    fresh = pd.DataFrame(collected)
    if not len(fresh):
        out = existing
    elif not len(existing):
        out = fresh          # concat with an empty frame warns about dtypes
    else:
        out = pd.concat([existing, fresh], ignore_index=True)
    out.to_parquet(cache, index=False)
    return out


def to_panels(
    facts: pd.DataFrame, dates: pd.DatetimeIndex, tickers: pd.Index
) -> dict[str, pd.DataFrame]:
    """Point-in-time ``date x ticker`` frames, one per concept.

    Each cell is the most recent value **filed on or before** that date. The
    forward-fill is what makes it point-in-time: a value appears the day it was
    filed and persists until superseded, which is exactly what a trader knew.
    """
    panels: dict[str, pd.DataFrame] = {}
    for concept, group in facts.groupby("concept"):
        wide = (
            group.assign(filed=pd.to_datetime(group["filed"]))
            .sort_values("filed")
            .drop_duplicates(["ticker", "filed"], keep="last")
            .pivot(index="filed", columns="ticker", values="value")
        )
        # Union the filing dates with the trading calendar, fill forward, then
        # keep only trading days -- filings land on days the market may be shut.
        combined = wide.reindex(wide.index.union(dates)).ffill()
        panels[concept] = combined.reindex(index=dates, columns=tickers)
    return panels


def coverage_report(panels: dict[str, pd.DataFrame], members: pd.DataFrame) -> pd.Series:
    """Fraction of universe name-days each concept actually covers.

    Worth checking before trusting any fundamental signal: a sleeve built on a
    concept covering 40% of the book is mostly a bet on which companies happen to
    tag their filings a particular way.
    """
    in_universe = members.to_numpy(dtype=bool)
    total = in_universe.sum()
    return pd.Series(
        {
            concept: float((panel.notna().to_numpy() & in_universe).sum() / max(total, 1))
            for concept, panel in panels.items()
        },
        name="universe_coverage",
    ).sort_values(ascending=False)
