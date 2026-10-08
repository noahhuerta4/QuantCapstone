"""Data cleaning and universe construction (spec S3)."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_capstone.config import DataConfig
from quant_capstone.data import PriceData, build_universe, clean_prices, make_synthetic
from quant_capstone.data.clean import _listed_mask, flag_bad_prints


def _panel(close: pd.DataFrame) -> PriceData:
    volume = pd.DataFrame(1e6, index=close.index, columns=close.columns)
    return PriceData(
        open=close, high=close, low=close, close=close,
        open_raw=close, close_raw=close, volume=volume,
        sectors=pd.Series("Tech", index=close.columns),
    )


def test_price_data_rejects_misaligned_fields():
    index = pd.bdate_range("2020-01-01", periods=5)
    close = pd.DataFrame(1.0, index=index, columns=["A"])
    with pytest.raises(ValueError, match="not aligned"):
        PriceData(
            open=close, high=close, low=close, close=close,
            open_raw=close, close_raw=pd.DataFrame(1.0, index=index, columns=["B"]),
            volume=close, sectors=pd.Series("Tech", index=["A"]),
        )


def test_listed_mask_covers_only_the_traded_span():
    index = pd.bdate_range("2020-01-01", periods=6)
    close = pd.DataFrame({"A": [np.nan, 10.0, 11.0, 12.0, np.nan, np.nan]}, index=index)
    mask = _listed_mask(close)["A"].tolist()
    assert mask == [False, True, True, True, False, False]


def test_forward_fill_never_crosses_a_delisting():
    """The bug this prevents pays returns on a stock that no longer exists."""
    index = pd.bdate_range("2020-01-01", periods=8)
    close = pd.DataFrame({"A": [10.0, 11.0, 12.0, np.nan, np.nan, np.nan, np.nan, np.nan]}, index=index)

    cleaned = clean_prices(_panel(close), max_ffill=3, min_observations=1)
    assert cleaned.close["A"].iloc[3:].isna().all(), "must not resurrect a delisted name"
    assert cleaned.close["A"].iloc[:3].notna().all()


def test_forward_fill_bridges_a_halt_inside_the_listed_span():
    index = pd.bdate_range("2020-01-01", periods=6)
    close = pd.DataFrame({"A": [10.0, np.nan, np.nan, 12.0, 13.0, 14.0]}, index=index)
    cleaned = clean_prices(_panel(close), max_ffill=3, min_observations=1)
    assert cleaned.close["A"].iloc[1] == 10.0
    assert cleaned.close["A"].notna().all()


def test_forward_fill_respects_its_limit():
    index = pd.bdate_range("2020-01-01", periods=8)
    close = pd.DataFrame({"A": [10.0, *([np.nan] * 5), 12.0, 13.0]}, index=index)
    cleaned = clean_prices(_panel(close), max_ffill=2, min_observations=1)
    assert cleaned.close["A"].iloc[3:6].isna().any()


def test_bad_prints_are_spikes_that_immediately_reverse():
    index = pd.bdate_range("2020-01-01", periods=5)
    close = pd.DataFrame({"A": [100.0, 101.0, 300.0, 102.0, 103.0]}, index=index)
    assert flag_bad_prints(close)["A"].iloc[2]
    # A genuine sustained move is not a bad print.
    trend = pd.DataFrame({"A": [100.0, 101.0, 300.0, 305.0, 310.0]}, index=index)
    assert not flag_bad_prints(trend)["A"].any()


def test_names_with_too_little_history_are_dropped():
    index = pd.bdate_range("2020-01-01", periods=10)
    close = pd.DataFrame(
        {"A": np.arange(10.0, 20.0), "B": [np.nan] * 8 + [10.0, 11.0]}, index=index
    )
    cleaned = clean_prices(_panel(close), min_observations=5)
    assert "A" in cleaned.tickers and "B" not in cleaned.tickers


def test_adv_is_lagged_so_today_cannot_see_today():
    data = make_synthetic(n_assets=5, n_days=60, seed=1)
    adv = data.adv(10)
    dollar_volume = data.dollar_volume()
    expected = dollar_volume.rolling(10, min_periods=5).mean().shift(1)
    pd.testing.assert_frame_equal(adv, expected)
    assert adv.iloc[0].isna().all()


def test_universe_picks_the_most_liquid_names():
    data = make_synthetic(n_assets=20, n_days=500, seed=2)
    cfg = DataConfig(universe_size=5, adv_lookback=20, min_history=50, min_price=0.0)
    members = build_universe(data, cfg)

    live = members.loc[members.any(axis=1)]
    assert live.sum(axis=1).max() <= 5
    # The selected names should be the high-ADV ones.
    chosen = members.iloc[-1]
    adv_last = data.adv(20).iloc[-1]
    assert adv_last[chosen[chosen].index].min() >= adv_last[~chosen].max()


def test_universe_refresh_holds_membership_between_rebalances():
    data = make_synthetic(n_assets=20, n_days=400, seed=3)
    cfg = DataConfig(universe_size=6, adv_lookback=20, min_history=50, universe_refresh=21, min_price=0.0)
    members = build_universe(data, cfg)

    live = members.loc[members.any(axis=1)]
    changes = (live != live.shift(1)).any(axis=1).sum()
    assert changes < len(live) / 5, "membership should be sticky between refreshes"


def test_subset_and_slice_preserve_alignment():
    data = make_synthetic(n_assets=8, n_days=40, seed=4)
    subset = data.subset(data.tickers[:3])
    assert len(subset.tickers) == 3 and len(subset.dates) == 40

    sliced = data.slice_dates(end=data.dates[19])
    assert len(sliced.dates) == 20 and len(sliced.tickers) == 8


def test_reused_dataset_rebuilds_the_universe_when_the_config_changes(dataset, cfg):
    """Regression: sweeping a universe parameter against a cached panel used to
    reuse the old membership, making every cell of the sweep identical."""
    narrow = dataset.rebuilt_for(replace(cfg.data, universe_size=8))
    assert narrow.members.sum(axis=1).max() <= 8
    assert narrow.members.sum().sum() < dataset.members.sum().sum()
    # Same config must hand back the very same object, not recompute.
    assert dataset.rebuilt_for(cfg.data) is dataset


def test_rebuilding_across_a_different_date_range_is_refused(dataset, cfg):
    with pytest.raises(ValueError, match="reload the dataset"):
        dataset.rebuilt_for(replace(cfg.data, start="2001-01-01"))


def test_universe_size_actually_changes_the_strategy(dataset, cfg):
    from quant_capstone.strategy import run_strategy

    wide = run_strategy(cfg.override(data__universe_size=30), dataset)
    narrow = run_strategy(cfg.override(data__universe_size=10), dataset)
    assert wide.portfolio.diagnostics["n_active"].mean() > narrow.portfolio.diagnostics["n_active"].mean()
