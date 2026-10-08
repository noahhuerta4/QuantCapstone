"""One call that turns a Config into (prices, universe membership)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..config import Config, DataConfig
from .base import DataSource, PriceData
from .clean import clean_prices
from .synthetic import SyntheticSource
from .universe import build_universe
from .yfinance_source import YFinanceSource

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Dataset:
    prices: PriceData
    members: pd.DataFrame
    """Boolean date x ticker point-in-time universe membership."""
    data_cfg: DataConfig | None = None
    """The config that produced ``members``, so reuse can detect a stale universe."""
    form4: "pd.DataFrame | None" = None
    """Insider trade table keyed on filing date. None unless a sleeve needs it.

    Note for callers: ``fundamentals`` sits after this field, so construct a
    Dataset with keyword arguments. Passing a fundamentals dict positionally
    lands it here instead, which fails later and far from the cause."""
    fundamentals: dict = field(default_factory=dict)
    """Point-in-time SEC panels, keyed by concept. Empty unless a fundamental
    sleeve was requested -- fetching them costs a few minutes and most runs do
    not need them."""

    @property
    def returns(self) -> pd.DataFrame:
        return self.prices.returns()

    def rebuilt_for(self, data_cfg: DataConfig) -> "Dataset":
        """Return a dataset whose universe matches ``data_cfg``, reusing the panel.

        Sweeping a universe parameter while reusing a loaded panel is the normal
        case, and it has to actually take effect: silently keeping the old
        membership makes every cell of such a sweep identical, which looks like a
        flat sensitivity surface rather than like a bug.
        """
        if self.data_cfg is not None and self.data_cfg == data_cfg:
            return self

        if self.data_cfg is not None:
            for field in ("source", "start", "end"):
                if getattr(self.data_cfg, field) != getattr(data_cfg, field):
                    raise ValueError(
                        f"cannot rebuild a universe for a different data.{field} "
                        f"({getattr(self.data_cfg, field)!r} -> {getattr(data_cfg, field)!r}); "
                        "reload the dataset instead of reusing this panel"
                    )

        from .universe import build_universe

        return Dataset(
            self.prices, build_universe(self.prices, data_cfg), data_cfg,
            self.form4, self.fundamentals,
        )


def make_source(cfg: Config) -> DataSource:
    if cfg.data.source == "synthetic":
        return SyntheticSource()
    if cfg.data.source == "yfinance":
        return YFinanceSource(cache_dir=cfg.data.cache_dir)
    raise ValueError(f"unknown data source {cfg.data.source!r}")


def load_dataset(cfg: Config, tickers: list[str] | None = None) -> Dataset:
    source = make_source(cfg)

    if tickers is None and isinstance(source, YFinanceSource):
        tickers = source.sp500_constituents()["ticker"].tolist()
    tickers = tickers or []

    prices = source.fetch(tickers, cfg.data.start, cfg.data.end)
    prices = clean_prices(prices, min_observations=cfg.data.min_history)
    members = build_universe(prices, cfg.data)

    log.info(
        "dataset: %d dates, %d tickers, mean universe %.0f names",
        len(prices.dates), len(prices.tickers), members.sum(axis=1).mean(),
    )
    form4 = None
    if _needs_form4(cfg):
        from .edgar_form4 import fetch_form4

        form4 = fetch_form4(
            cfg.data.start, cfg.data.end, Path(cfg.data.cache_dir), list(prices.tickers)
        )
        log.info("form 4: %d insider trades across %d tickers", len(form4), form4.ticker.nunique())

    fundamentals: dict = {}
    if _needs_fundamentals(cfg):
        from .sec_fundamentals import coverage_report, fetch_company_facts, to_panels

        facts = fetch_company_facts(list(prices.tickers), Path(cfg.data.cache_dir))
        fundamentals = to_panels(facts, prices.dates, prices.tickers)
        log.info("fundamental coverage:\n%s", coverage_report(fundamentals, members).round(3))

    return Dataset(
        prices=prices, members=members, data_cfg=cfg.data,
        form4=form4, fundamentals=fundamentals,
    )


def _needs_fundamentals(cfg: Config) -> bool:
    from ..signals.build import requested_alphas
    from ..signals.events import needs_fundamentals

    return needs_fundamentals(requested_alphas(cfg))


def _needs_form4(cfg: Config) -> bool:
    from ..signals.build import requested_alphas
    from ..signals.events import needs_form4

    return needs_form4(requested_alphas(cfg))
