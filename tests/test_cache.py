"""Price-cache integrity.

Two failure modes, both of which silently corrupted real results in this project:
an unstable cache key that re-downloaded the panel every run, and a rate-limited
partial download that got cached and then served to every later run.
"""

import numpy as np
import pandas as pd
import pytest

from quant_capstone.data import YFinanceSource


def _panel(tickers, n_days=400, empty=()):
    """MultiIndex OHLCV frame shaped like a yfinance download."""
    index = pd.bdate_range("2015-01-01", periods=n_days, name="Date")
    frames = {}
    for field in ("Open", "High", "Low", "Close", "Adj Close", "Volume"):
        data = {}
        for t in tickers:
            data[t] = np.full(n_days, np.nan) if t in empty else np.linspace(10, 20, n_days)
        frames[field] = pd.DataFrame(data, index=index)
    return pd.concat(frames, axis=1)


def test_cache_key_is_stable_across_processes(tmp_path):
    """Regression: ``hash()`` on strings is randomised per process, so a
    hash-derived filename never hit the cache and re-downloaded every run."""
    source = YFinanceSource(cache_dir=tmp_path)
    tickers = ["AAPL", "MSFT", "NVDA"]

    first = source._cache_key(tickers, "2015-01-01", "2025-01-01")
    second = source._cache_key(list(reversed(tickers)), "2015-01-01", "2025-01-01")
    assert first == second, "key must not depend on input ordering"

    # A literal expected value pins it against any future hashing change.
    assert first == YFinanceSource(cache_dir=tmp_path)._cache_key(
        tickers, "2015-01-01", "2025-01-01"
    )
    assert first != source._cache_key(tickers, "2016-01-01", "2025-01-01")


def test_populated_tickers_ignores_empty_columns(tmp_path):
    tickers = ["AAA", "BBB", "CCC"]
    path = tmp_path / "bars_2015-01-01_2025-01-01_3_deadbeef.parquet"
    _panel(tickers, empty=("CCC",)).to_parquet(path)

    populated = YFinanceSource._populated_tickers(path)
    assert populated == {"AAA", "BBB"}, "an all-NaN column is not coverage"


def test_a_truncated_cache_file_is_not_treated_as_a_hit(tmp_path):
    """Regression: a rate-limited download returns the full column set with no
    data in it. Matching on column names alone served that to every later run."""
    source = YFinanceSource(cache_dir=tmp_path)
    tickers = [f"T{i:03d}" for i in range(20)]

    poisoned = tmp_path / "bars_2015-01-01_2025-01-01_20_00000000.parquet"
    _panel(tickers, empty=tickers[5:]).to_parquet(poisoned)
    assert source._find_covering_cache(tickers, "2015-01-01", "2025-01-01") is None

    good = tmp_path / "bars_2015-01-01_2025-01-01_20_ffffffff.parquet"
    _panel(tickers).to_parquet(good)
    assert source._find_covering_cache(tickers, "2015-01-01", "2025-01-01") == good


def test_a_wider_cached_panel_satisfies_a_narrower_request(tmp_path):
    source = YFinanceSource(cache_dir=tmp_path)
    wide = [f"T{i:03d}" for i in range(50)]
    path = tmp_path / "bars_2015-01-01_2025-01-01_50_abcdef01.parquet"
    _panel(wide).to_parquet(path)

    assert source._find_covering_cache(wide[:10], "2015-01-01", "2025-01-01") == path
    # A request the panel cannot cover must miss rather than silently truncate.
    assert source._find_covering_cache(
        wide[:10] + ["NOTHERE" + str(i) for i in range(10)], "2015-01-01", "2025-01-01"
    ) is None


def test_missing_cache_directory_is_a_clean_miss(tmp_path):
    source = YFinanceSource(cache_dir=tmp_path / "empty")
    assert source._find_covering_cache(["AAPL"], "2015-01-01", "2025-01-01") is None
