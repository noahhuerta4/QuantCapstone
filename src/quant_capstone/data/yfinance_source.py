"""yfinance loader with an on-disk parquet cache.

SURVIVORSHIP BIAS (spec S3): yfinance serves only currently-listed tickers, and
we seed the universe from *today's* S&P 500 membership. Names that blew up and
delisted are therefore absent from the whole history. For a reversal strategy
this biases results **optimistic**: the missing names are disproportionately the
ones whose 'cheap after a big drop' signal never mean-reverted. Treat every
number this source produces as an upper bound and see ``docs`` in the README for
how much we think it is worth.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

from .base import FIELDS, PriceData

log = logging.getLogger(__name__)

WIKI_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKI_SP400 = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
WIKI_SP600 = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"

INDEX_PAGES = {"sp500": WIKI_SP500, "sp400": WIKI_SP400, "sp600": WIKI_SP600}


class YFinanceSource:
    def __init__(self, cache_dir: str | Path = "data_cache", use_cache: bool = True):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache

    # ---------------------------------------------------------------- universe
    def sp500_constituents(self) -> pd.DataFrame:
        """Current S&P 500 membership with GICS sectors (cached).

        This is a *current* snapshot, not point-in-time -- see the module
        docstring. ``universe.py`` still applies a point-in-time liquidity screen
        on top so at least the ADV ranking uses no future information.
        """
        cache = self.cache_dir / "sp500_constituents.parquet"
        if self.use_cache and cache.exists():
            return pd.read_parquet(cache)

        out = self._scrape_index("sp500")
        out.to_parquet(cache, index=False)
        return out

    def _scrape_index(self, name: str) -> pd.DataFrame:
        """Ticker + GICS sector from one S&P index page."""
        # Wikipedia rejects pandas' default urllib user-agent with a 403, so
        # fetch the page ourselves and hand read_html the markup.
        import io

        import requests

        response = requests.get(
            INDEX_PAGES[name],
            headers={"User-Agent": "quant-capstone/0.1 (research project; contact via repo)"},
            timeout=30,
        )
        response.raise_for_status()

        # Column names and table position differ across the three index pages,
        # so take the first table that carries both a ticker and a sector.
        for table in pd.read_html(io.StringIO(response.text)):
            cols = {str(c).strip().lower(): c for c in table.columns}
            ticker_col = next((cols[k] for k in ("symbol", "ticker symbol", "ticker") if k in cols), None)
            sector_col = next((cols[k] for k in ("gics sector", "sector") if k in cols), None)
            if ticker_col is None or sector_col is None:
                continue
            return pd.DataFrame(
                {
                    # Yahoo spells class shares with a dash, Wikipedia with a dot.
                    "ticker": table[ticker_col].astype(str).str.replace(".", "-", regex=False).str.strip(),
                    "sector": table[sector_col].astype(str).str.strip(),
                    "index": name,
                }
            ).drop_duplicates("ticker")
        raise RuntimeError(f"no constituent table found on the {name} page")

    def sp1500_constituents(self) -> pd.DataFrame:
        """S&P 500 + 400 + 600 -- a ~1500-name large/mid/small liquidity spectrum.

        The reason for the wider list: short-term reversal alpha is concentrated
        in *less* liquid names (Avramov-Chordia-Goyal 2006), so a universe drawn
        only from mega-caps samples the weakest part of the effect by
        construction.

        Same survivorship caveat as the 500 list, and it bites harder here --
        small-cap delisting rates are far above large-cap ones, so the optimistic
        bias grows as you move down the size spectrum. Any cross-decile
        comparison of alpha is therefore biased *in favour* of the small end, and
        that has to be stated alongside the result.
        """
        cache = self.cache_dir / "sp1500_constituents.parquet"
        if self.use_cache and cache.exists():
            return pd.read_parquet(cache)

        frames = []
        for name in ("sp500", "sp400", "sp600"):
            try:
                frames.append(self._scrape_index(name))
            except Exception as exc:
                log.warning("could not fetch %s constituents: %s", name, exc)
        if not frames:
            raise RuntimeError("no constituent lists could be fetched")

        out = pd.concat(frames, ignore_index=True).drop_duplicates("ticker")
        out.to_parquet(cache, index=False)
        return out

    # ------------------------------------------------------------------- bars
    def _cache_key(self, tickers: list[str], start: str, end: str) -> str:
        """Stable cache key.

        Deliberately hashlib and not ``hash()``: Python randomises string hashing
        per process unless PYTHONHASHSEED is pinned, so a ``hash()``-derived
        filename differs on every run and the cache never hits. That silently
        re-downloads the whole panel each time and gets the session rate-limited.
        """
        digest = hashlib.sha1("|".join(sorted(tickers)).encode()).hexdigest()[:12]
        return f"bars_{start}_{end}_{len(tickers)}_{digest}"

    @staticmethod
    def _populated_tickers(path: Path, min_rows: int = 252) -> set[str]:
        """Tickers in a cached panel that actually carry price data.

        Presence of a column is not evidence of data: a rate-limited yfinance
        response still returns the full column set, just empty. Counting nulls is
        the only way to tell a real panel from a poisoned one, and reading null
        counts straight from the parquet metadata makes it cheap enough to do on
        every candidate file.
        """
        try:
            import pyarrow.parquet as pq

            handle = pq.ParquetFile(path)
            close_columns = [n for n in handle.schema_arrow.names if n.startswith("('Close',")]
            if not close_columns:
                return set()
            table = handle.read(columns=close_columns)
            n_rows = table.num_rows
            return {
                name.split("'")[3]
                for name, column in zip(close_columns, table.columns)
                if n_rows - column.null_count >= min_rows
            }
        except Exception:
            try:
                close = pd.read_parquet(path)["Close"]
                counts = close.notna().sum()
                return set(counts[counts >= min_rows].index)
            except Exception:
                return set()

    def _find_covering_cache(self, tickers: list[str], start: str, end: str) -> Path | None:
        """Reuse any cached panel for this date range that covers these tickers.

        Lets a narrow request ride on a previously downloaded wider panel (asking
        for the S&P 500 when the S&P 1500 is already on disk), and rescues files
        written under the old unstable key scheme. Coverage is measured on
        *populated* tickers so a truncated download cannot masquerade as a hit.
        """
        wanted = set(tickers)
        best: tuple[int, Path] | None = None
        for path in sorted(self.cache_dir.glob(f"bars_{start}_{end}_*.parquet")):
            covered = len(wanted & self._populated_tickers(path))
            if covered >= 0.95 * len(wanted) and (best is None or covered > best[0]):
                best = (covered, path)
        if best:
            log.debug("cache %s covers %d/%d tickers", best[1].name, best[0], len(wanted))
        return best[1] if best else None

    def fetch(self, tickers: list[str], start: str, end: str) -> PriceData:
        import yfinance as yf

        cache = self.cache_dir / f"{self._cache_key(tickers, start, end)}.parquet"
        if self.use_cache and not cache.exists():
            covering = self._find_covering_cache(tickers, start, end)
            if covering is not None:
                log.info("reusing cached panel %s", covering.name)
                cache = covering

        if self.use_cache and cache.exists():
            raw = pd.read_parquet(cache)
        else:
            log.info("downloading %d tickers from yfinance (%s to %s)", len(tickers), start, end)
            raw = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=False,  # we need both raw and adjusted prices
                actions=False,
                progress=False,
                group_by="column",
                threads=True,
            )
            if raw is None or raw.empty:
                raise RuntimeError("yfinance returned no data; check tickers/date range/network")

            # A rate-limited response comes back structurally complete and almost
            # entirely null. Caching that poisons every later run, so refuse it
            # rather than persisting a panel we would not trust.
            close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
            populated = (close.notna().sum() >= 252).sum()
            if populated < 0.8 * len(tickers):
                raise RuntimeError(
                    f"yfinance returned data for only {populated}/{len(tickers)} tickers "
                    "(likely rate limiting); not caching. Wait and retry."
                )
            raw.to_parquet(cache)

        return self._to_price_data(raw, tickers)

    def _to_price_data(self, raw: pd.DataFrame, tickers: list[str]) -> PriceData:
        def field(name: str) -> pd.DataFrame:
            frame = raw[name] if isinstance(raw.columns, pd.MultiIndex) else raw[[name]]
            return frame.sort_index()

        open_raw, high_raw, low_raw = field("Open"), field("High"), field("Low")
        close_raw, adj_close, volume = field("Close"), field("Adj Close"), field("Volume")

        # One factor per name per day converts raw OHLC into adjusted space.
        factor = (adj_close / close_raw).replace([float("inf"), -float("inf")], pd.NA)

        panel = {
            "open": open_raw * factor,
            "high": high_raw * factor,
            "low": low_raw * factor,
            "close": adj_close,
            "open_raw": open_raw,
            "close_raw": close_raw,
            "volume": volume,
        }
        cols = pd.Index([t for t in tickers if t in close_raw.columns])
        panel = {k: v.reindex(columns=cols).astype(float) for k, v in panel.items()}

        sectors = self._sector_map().reindex(cols).fillna("Unknown")
        return PriceData(**{f: panel[f] for f in FIELDS}, sectors=sectors)

    def _sector_map(self) -> pd.Series:
        """Widest cached sector mapping available.

        Must prefer the 1500 list when it exists: reading only the S&P 500 list
        for a 1500-name panel leaves two thirds of the names labelled 'Unknown',
        which then forms one enormous pseudo-sector that sector-neutralization
        happily hedges against. The neutrality constraint would be satisfied and
        meaningless.
        """
        for loader in (self.sp1500_constituents, self.sp500_constituents):
            try:
                cached = self.cache_dir / f"{loader.__name__.split('_')[0]}_constituents.parquet"
                if not cached.exists() and loader is self.sp1500_constituents:
                    continue  # don't trigger a scrape just to label sectors
                return loader().set_index("ticker")["sector"]
            except Exception:
                continue
        log.warning("sector map unavailable; falling back to a single 'Unknown' sector")
        return pd.Series(dtype=str)
