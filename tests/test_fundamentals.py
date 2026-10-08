"""Point-in-time SEC fundamentals (spec S3: no look-ahead).

The claim worth testing is the one the whole data layer rests on: a fundamental
value must become visible on the date it was FILED, not the date the accounting
period ended. Getting that wrong hands the backtest months of future information
and is the classic way fundamental strategies fake an edge.
"""

import numpy as np
import pandas as pd
import pytest

from quant_capstone.data.sec_fundamentals import _extract, coverage_report, to_panels
from quant_capstone.signals.fundamental import (
    asset_growth,
    book_to_price,
    earnings_yield,
    gross_profitability,
    market_cap,
)


def _facts(rows):
    return pd.DataFrame(rows, columns=["ticker", "concept", "filed", "value"])


def test_value_appears_on_the_filing_date_not_the_period_end():
    """A December year-end filed in February must be invisible in January."""
    dates = pd.bdate_range("2020-01-01", "2020-04-01")
    facts = _facts([("AAA", "assets", "2020-02-14", 1000.0)])
    panel = to_panels(facts, dates, pd.Index(["AAA"]))["assets"]["AAA"]

    assert panel.loc["2020-01-15"] != panel.loc["2020-01-15"], "NaN before the filing"
    assert np.isnan(panel.loc["2020-02-13"]), "still unknown the day before filing"
    assert panel.loc["2020-02-14"] == 1000.0, "known on the filing date"
    assert panel.loc["2020-03-31"] == 1000.0, "and persists afterwards"


def test_a_restatement_does_not_rewrite_history():
    """Refiling 2020 numbers in 2021 must not change what we knew in 2020."""
    dates = pd.bdate_range("2020-01-01", "2021-06-30")
    facts = _facts([
        ("AAA", "assets", "2020-02-14", 1000.0),
        ("AAA", "assets", "2021-03-01", 1200.0),   # restated
    ])
    panel = to_panels(facts, dates, pd.Index(["AAA"]))["assets"]["AAA"]

    assert panel.loc["2020-06-30"] == 1000.0, "must see the originally filed figure"
    assert panel.loc["2021-06-30"] == 1200.0, "and the restated one only after it lands"


def test_future_filings_cannot_leak_backwards():
    dates = pd.bdate_range("2020-01-01", "2020-12-31")
    facts = _facts([("AAA", "equity", "2020-07-01", 500.0)])
    panel = to_panels(facts, dates, pd.Index(["AAA"]))["equity"]["AAA"]
    assert panel.loc[:"2020-06-30"].isna().all()
    assert panel.loc["2020-07-01":].notna().all()


def test_extract_keeps_instants_and_rejects_durations():
    facts = {"facts": {"us-gaap": {"Assets": {"units": {"USD": [
        {"end": "2020-12-31", "filed": "2021-02-01", "val": 10.0},                      # instant
        {"start": "2020-01-01", "end": "2020-12-31", "filed": "2021-02-01", "val": 99.0},  # duration
    ]}}}}}
    assert _extract(facts, ("Assets",), instant=True) == [("2021-02-01", 10.0)]


def test_extract_keeps_only_annual_durations():
    """Quarterly and YTD-partial facts must not be mistaken for annual flows."""
    facts = {"facts": {"us-gaap": {"NetIncomeLoss": {"units": {"USD": [
        {"start": "2020-01-01", "end": "2020-03-31", "filed": "2020-05-01", "val": 5.0},   # quarter
        {"start": "2020-01-01", "end": "2020-12-31", "filed": "2021-02-01", "val": 20.0},  # year
    ]}}}}}
    assert _extract(facts, ("NetIncomeLoss",), instant=False) == [("2021-02-01", 20.0)]


def test_extract_falls_back_through_tag_aliases():
    facts = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": [
        {"start": "2020-01-01", "end": "2020-12-31", "filed": "2021-02-01", "val": 42.0},
    ]}}}}}
    assert _extract(facts, ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues"),
                    instant=False) == [("2021-02-01", 42.0)]
    assert _extract(facts, ("NotATag",), instant=False) == []


# ------------------------------------------------------------------- sleeves
def _frame(values):
    return pd.DataFrame({"A": [values]}, index=pd.bdate_range("2020-01-01", periods=1))


def test_market_cap_uses_unadjusted_price():
    cap = market_cap(_frame(100.0), _frame(1e9))
    assert cap.iloc[0, 0] == pytest.approx(1e11)


def test_tiny_market_caps_are_dropped():
    assert market_cap(_frame(0.01), _frame(1000.0)).isna().all().all()


def test_book_to_price_drops_negative_equity():
    """Negative book is an accounting artefact, not 'infinitely expensive'."""
    cap = _frame(1e9)
    assert np.isnan(book_to_price(_frame(-5e8), cap).iloc[0, 0])
    assert book_to_price(_frame(5e8), cap).iloc[0, 0] == pytest.approx(0.5)


def test_earnings_yield_keeps_losses_as_negative():
    """Unlike book, negative earnings are interpretable and must be ranked."""
    assert earnings_yield(_frame(-1e8), _frame(1e9)).iloc[0, 0] == pytest.approx(-0.1)


def test_gross_profitability_is_profit_over_assets():
    value = gross_profitability(_frame(1000.0), _frame(600.0), _frame(2000.0)).iloc[0, 0]
    assert value == pytest.approx(0.2)


def test_asset_growth_sign_is_flipped_so_higher_is_better():
    index = pd.bdate_range("2020-01-01", periods=300)
    assets = pd.DataFrame({"grower": np.linspace(100, 200, 300),
                           "steady": np.full(300, 100.0)}, index=index)
    signal = asset_growth(assets, lookback=252).iloc[-1]
    assert signal["grower"] < signal["steady"], "aggressive expansion should score lower"


def test_coverage_report_measures_universe_overlap():
    dates = pd.bdate_range("2020-01-01", periods=10)
    tickers = pd.Index(["A", "B"])
    members = pd.DataFrame(True, index=dates, columns=tickers)
    panels = {"assets": pd.DataFrame({"A": 1.0, "B": np.nan}, index=dates)}
    assert coverage_report(panels, members)["assets"] == pytest.approx(0.5)


# --------------------------------------------------- PEAD wiring end-to-end
def _synthetic_filings(dates, tickers, seed=0):
    """Quarterly net income with varying surprises, on staggered calendars."""
    rng = np.random.default_rng(seed)
    panel = pd.DataFrame(np.nan, index=dates, columns=tickers)
    for offset, ticker in enumerate(tickers):
        rows = range(offset * 7, len(dates), 63)   # staggered fiscal calendars
        level = 100.0
        for r in rows:
            level += rng.normal(5, 15)
            panel.iloc[r:, panel.columns.get_loc(ticker)] = level
    return panel


def test_pead_weight_pulls_in_the_sleeve_without_listing_it():
    from quant_capstone.config import Config
    from quant_capstone.signals.build import requested_alphas

    off = Config().override(signal__pead_weight=0.0)
    assert "earnings_surprise" not in requested_alphas(off)

    on = Config().override(signal__pead_weight=0.1)
    assert "earnings_surprise" in requested_alphas(on)


def test_pead_sleeve_builds_and_carries_less_persistence_than_a_static_ratio(dataset, cfg):
    """The whole point of PEAD: an event clock gives it breadth a ratio lacks."""
    from quant_capstone.data.loader import Dataset
    from quant_capstone.signals import build_sleeves
    from quant_capstone.signals.fundamental import standardized_unexpected_earnings

    dates, tickers = dataset.prices.dates, dataset.prices.tickers
    quarterly = _synthetic_filings(dates, tickers)
    static = quarterly.copy()          # a level, not a surprise

    with_funds = Dataset(
        dataset.prices, dataset.members, dataset.data_cfg,
        fundamentals={"net_income_q": quarterly, "assets": static},
    )
    sleeves = build_sleeves(with_funds, cfg.override(signal__pead_weight=0.2))
    assert "earnings_surprise" in sleeves

    pead = sleeves["earnings_surprise"]
    assert pead.notna().to_numpy().sum() > 0, "sleeve must produce values"

    pead_persistence = float(pead.corrwith(pead.shift(1), axis=1).mean())
    raw = standardized_unexpected_earnings(quarterly)
    assert np.isfinite(pead_persistence)
    # A decaying event score must turn over faster than the level it came from.
    level_persistence = float(static.corrwith(static.shift(1), axis=1).mean())
    assert pead_persistence < level_persistence


def test_a_fundamental_sleeve_without_data_fails_loudly(dataset, cfg):
    from quant_capstone.signals import build_sleeves

    with pytest.raises(ValueError, match="no fundamentals"):
        build_sleeves(dataset, cfg.override(signal__pead_weight=0.2))
