"""Insider transactions (SEC Form 4) from the bulk structured-data sets.

Why the bulk sets and not the filings themselves: a Form 4 is a small XML
document, and the S&P 500 generates well over a hundred thousand of them across a
decade. Fetching them individually would be ~500k requests. The SEC publishes the
same content pre-parsed as quarterly TSV bundles, so forty downloads replace all
of it.

**Point-in-time discipline.** Each submission carries two dates and they are not
interchangeable. ``TRANS_DATE`` is when the insider traded; ``FILING_DATE`` is
when the public found out. Form 4 must be filed within two business days, so the
gap is short — but it is not zero, and ranking stocks on the transaction date
would be trading on information that was still private. Everything here keys off
the filing date.

**Why this sleeve should have breadth.** Insider filings arrive continuously
across the universe rather than on a quarterly cadence, so the cross-section
keeps turning over. That is the property the static value ratios lacked, and the
reason this is a better candidate than another accounting ratio.
"""

from __future__ import annotations

import io
import logging
import time
import zipfile
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

BULK_URL = (
    "https://www.sec.gov/files/structureddata/data/"
    "insider-transactions-data-sets/{year}q{quarter}_form345.zip"
)
REQUEST_PAUSE = 0.4

#: Open-market purchases and sales. Deliberately narrow: code P and S are
#: discretionary trades at market prices. Codes A (grant), M (option exercise),
#: F (tax withholding) and G (gift) dominate by count and carry no view -- an
#: executive receiving a scheduled stock grant is not expressing an opinion.
INFORMATIVE_CODES = ("P", "S")

_COLUMNS = [
    "ticker", "filing_date", "trans_date", "code", "shares", "price",
    "value", "relationship", "title",
]


def _session():
    import requests

    from .sec_fundamentals import SEC_USER_AGENT

    session = requests.Session()
    session.headers.update({"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"})
    return session


#: The bulk TSVs date-stamp as ``31-JAN-2024``. Specifying the format avoids
#: pandas falling back to per-element dateutil parsing across 200k+ rows.
DATE_FORMAT = "%d-%b-%Y"


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format=DATE_FORMAT, errors="coerce")


def _parse_quarter(blob: bytes, tickers: set[str] | None) -> pd.DataFrame:
    """Join the three TSVs in one quarterly bundle into a flat trade table."""
    archive = zipfile.ZipFile(io.BytesIO(blob))

    submission = pd.read_csv(
        archive.open("SUBMISSION.tsv"), sep="\t", dtype=str,
        usecols=["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERTRADINGSYMBOL"],
    )
    submission = submission[submission["DOCUMENT_TYPE"] == "4"]
    submission["ticker"] = (
        submission["ISSUERTRADINGSYMBOL"].str.strip().str.upper().str.replace(".", "-", regex=False)
    )
    if tickers is not None:
        submission = submission[submission["ticker"].isin(tickers)]
    if submission.empty:
        return pd.DataFrame(columns=_COLUMNS)

    trades = pd.read_csv(
        archive.open("NONDERIV_TRANS.tsv"), sep="\t", dtype=str,
        usecols=["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES",
                 "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD"],
    )
    trades = trades[trades["TRANS_CODE"].isin(INFORMATIVE_CODES)]

    owners = pd.read_csv(
        archive.open("REPORTINGOWNER.tsv"), sep="\t", dtype=str,
        usecols=["ACCESSION_NUMBER", "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE"],
    ).drop_duplicates("ACCESSION_NUMBER")

    merged = trades.merge(submission, on="ACCESSION_NUMBER").merge(
        owners, on="ACCESSION_NUMBER", how="left"
    )
    if merged.empty:
        return pd.DataFrame(columns=_COLUMNS)

    shares = pd.to_numeric(merged["TRANS_SHARES"], errors="coerce")
    price = pd.to_numeric(merged["TRANS_PRICEPERSHARE"], errors="coerce")
    # Sign from the acquired/disposed flag rather than the code, because it is
    # the field the filer is actually required to get right.
    sign = merged["TRANS_ACQUIRED_DISP_CD"].map({"A": 1.0, "D": -1.0})

    out = pd.DataFrame(
        {
            "ticker": merged["ticker"],
            "filing_date": _to_datetime(merged["FILING_DATE"]),
            "trans_date": _to_datetime(merged["TRANS_DATE"]),
            "code": merged["TRANS_CODE"],
            "shares": shares * sign,
            "price": price,
            "value": shares * price * sign,
            "relationship": merged["RPTOWNER_RELATIONSHIP"].fillna(""),
            "title": merged["RPTOWNER_TITLE"].fillna(""),
        }
    )
    return out.dropna(subset=["filing_date", "trans_date", "shares"])


def fetch_form4(
    start: str, end: str, cache_dir: Path | str = "data_cache",
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Every Form 4 open-market trade for ``tickers`` between two dates.

    Cached as one parquet keyed on the quarter range; a re-run reads it back
    rather than re-downloading ~550 MB.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    cache = cache_dir / f"form4_{first.year}_{last.year}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    wanted = {t.upper() for t in tickers} if tickers else None
    session = _session()
    quarters = pd.period_range(first, last, freq="Q")
    frames: list[pd.DataFrame] = []

    log.info("downloading %d quarterly Form 4 bundles", len(quarters))
    for i, period in enumerate(quarters, 1):
        url = BULK_URL.format(year=period.year, quarter=period.quarter)
        try:
            response = session.get(url, timeout=180)
            if response.status_code != 200:
                log.warning("Form 4 bundle unavailable: %sQ%s", period.year, period.quarter)
                continue
            frames.append(_parse_quarter(response.content, wanted))
        except Exception as exc:
            log.warning("Form 4 %sQ%s failed: %s", period.year, period.quarter, exc)
        finally:
            time.sleep(REQUEST_PAUSE)
        if i % 8 == 0:
            log.info("  %d/%d quarters, %d trades so far", i, len(quarters),
                     sum(len(f) for f in frames))

    if not frames:
        raise RuntimeError("no Form 4 data retrieved")
    out = pd.concat(frames, ignore_index=True)
    out = out[(out["filing_date"] >= first) & (out["filing_date"] <= last)]
    out.to_parquet(cache, index=False)
    return out


def classify_routine(trades: pd.DataFrame, min_years: int = 3) -> pd.DataFrame:
    """Split trades into routine and opportunistic (Cohen, Malloy & Pomorski 2012).

    An insider who buys every March for three straight years is following a
    personal schedule, not acting on a view, and their trades carry no signal.
    The subset that survives this filter is where the information is. Adds a
    boolean ``opportunistic`` column.

    Classification uses only each insider's *prior* history at the time of the
    trade, so it cannot use future filings to decide what was routine.
    """
    work = trades.sort_values("filing_date").copy()
    work["month"] = work["trans_date"].dt.month
    work["year"] = work["trans_date"].dt.year
    work["insider"] = work["ticker"] + "|" + work["title"].str.lower().str.strip()

    opportunistic = []
    history: dict[tuple[str, int], set[int]] = {}
    for insider, month, year in zip(work["insider"], work["month"], work["year"]):
        seen = history.setdefault((insider, month), set())
        # Routine once the same insider has traded this calendar month in at
        # least ``min_years`` earlier years.
        opportunistic.append(len(seen - {year}) < min_years)
        seen.add(year)

    work["opportunistic"] = opportunistic
    return work.drop(columns=["month", "year", "insider"])


def to_panel(
    trades: pd.DataFrame, dates: pd.DatetimeIndex, tickers: pd.Index,
    value_column: str = "value",
) -> pd.DataFrame:
    """Daily ``date x ticker`` panel of net insider flow, keyed on filing date.

    Zero where a name had no filings that day — genuinely no insider activity is
    information, unlike a missing fundamental, so it is not NaN.
    """
    if trades.empty:
        return pd.DataFrame(0.0, index=dates, columns=tickers)

    daily = (
        trades.groupby([trades["filing_date"].dt.normalize(), "ticker"])[value_column]
        .sum()
        .unstack(fill_value=0.0)
    )
    return daily.reindex(index=dates, columns=tickers, fill_value=0.0).fillna(0.0)


def coverage_report(trades: pd.DataFrame, members: pd.DataFrame) -> pd.Series:
    """Sanity numbers before trusting any signal built on this."""
    in_universe = members.columns[members.any()]
    covered = trades["ticker"].isin(in_universe)
    return pd.Series(
        {
            "trades": float(len(trades)),
            "tickers": float(trades["ticker"].nunique()),
            "universe_tickers_covered": float(trades.loc[covered, "ticker"].nunique()),
            "purchases": float((trades["code"] == "P").sum()),
            "sales": float((trades["code"] == "S").sum()),
            "median_filing_lag_days": float(
                (trades["filing_date"] - trades["trans_date"]).dt.days.median()
            ),
            "first_filing": trades["filing_date"].min(),
            "last_filing": trades["filing_date"].max(),
        }
    )
