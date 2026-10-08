"""Data container and source interface.

Everything downstream speaks ``PriceData``: a bundle of ``date x ticker`` frames
sharing one index and one column set. Sources (yfinance today, IBKR later) only
have to produce this, which is what keeps the data layer swappable per spec S3.

Adjusted vs raw prices both matter and are kept separately:
  * returns and signals use **adjusted** prices (split/dividend consistent);
  * share counts, per-share commissions and ADV use **raw** prices, because a
    $0.0035/share commission is charged on the shares you actually traded at the
    price that actually printed, not on a back-adjusted one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

FIELDS = ("open", "high", "low", "close", "open_raw", "close_raw", "volume")


@dataclass(frozen=True)
class PriceData:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    open_raw: pd.DataFrame
    close_raw: pd.DataFrame
    volume: pd.DataFrame
    sectors: pd.Series
    """Ticker -> GICS sector label. Missing names get 'Unknown'."""

    def __post_init__(self) -> None:
        ref = self.close
        for name in FIELDS:
            frame = getattr(self, name)
            if not frame.index.equals(ref.index) or not frame.columns.equals(ref.columns):
                raise ValueError(f"PriceData field {name!r} is not aligned with close")

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    @property
    def tickers(self) -> pd.Index:
        return self.close.columns

    def returns(self) -> pd.DataFrame:
        """Adjusted close-to-close simple returns."""
        return self.close.pct_change()

    def dollar_volume(self) -> pd.DataFrame:
        """Raw traded dollars per name per day."""
        return self.close_raw * self.volume

    def adv(self, lookback: int) -> pd.DataFrame:
        """Trailing average dollar volume, shifted so day t sees only data < t."""
        return self.dollar_volume().rolling(lookback, min_periods=lookback // 2).mean().shift(1)

    def adv_shares(self, lookback: int) -> pd.DataFrame:
        return self.volume.rolling(lookback, min_periods=lookback // 2).mean().shift(1)

    def subset(self, tickers: pd.Index | list[str]) -> "PriceData":
        cols = pd.Index(tickers)
        return PriceData(
            **{name: getattr(self, name).reindex(columns=cols) for name in FIELDS},
            sectors=self.sectors.reindex(cols).fillna("Unknown"),
        )

    def slice_dates(self, start=None, end=None) -> "PriceData":
        sl = slice(start, end)
        return PriceData(
            **{name: getattr(self, name).loc[sl] for name in FIELDS},
            sectors=self.sectors,
        )


class DataSource(Protocol):
    """Any price source. Implement this to swap yfinance out for IBKR."""

    def fetch(self, tickers: list[str], start: str, end: str) -> PriceData: ...
