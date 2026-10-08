"""Event-driven sleeves and their ingestion (Phase 2, Direction A).

The properties worth pinning: insider flow must key off the FILING date rather
than the transaction date, the routine/opportunistic split must not peek at an
insider's future trades, and the trial ledger must count honestly.
"""

import numpy as np
import pandas as pd
import pytest

from quant_capstone.data.edgar_form4 import classify_routine, coverage_report, to_panel
from quant_capstone.signals.earnings_extra import (
    announcement_reversal,
    earnings_event_dates,
    event_density,
)
from quant_capstone.signals.events import EVENT_ALPHAS, EVENT_REQUIRES, needs_form4
from quant_capstone.signals.insider import (
    insider_flow,
    insider_flow_fast,
    net_buy_count,
    officer_filter,
)
from quant_capstone.validation import TrialLedger


def _trades(rows):
    frame = pd.DataFrame(rows, columns=["ticker", "filing_date", "trans_date", "code",
                                        "shares", "price", "value", "relationship", "title"])
    frame["filing_date"] = pd.to_datetime(frame["filing_date"])
    frame["trans_date"] = pd.to_datetime(frame["trans_date"])
    return frame


# ------------------------------------------------------------------ ingestion
def test_flow_panel_keys_on_filing_date_not_trade_date():
    """The insider trades before the market is told; only the filing is public."""
    dates = pd.bdate_range("2020-01-01", periods=10)
    trades = _trades([("AAA", "2020-01-09", "2020-01-06", "P", 100.0, 10.0, 1000.0, "Officer", "CEO")])
    panel = to_panel(trades, dates, pd.Index(["AAA"]))["AAA"]

    assert panel.loc["2020-01-06"] == 0.0, "must not appear on the transaction date"
    assert panel.loc["2020-01-08"] == 0.0
    assert panel.loc["2020-01-09"] == 1000.0, "appears on the filing date"


def test_flow_panel_sums_same_day_filings():
    dates = pd.bdate_range("2020-01-01", periods=5)
    trades = _trades([
        ("AAA", "2020-01-03", "2020-01-02", "P", 100.0, 10.0, 1000.0, "Officer", "CEO"),
        ("AAA", "2020-01-03", "2020-01-02", "S", -50.0, 10.0, -500.0, "Director", ""),
    ])
    assert to_panel(trades, dates, pd.Index(["AAA"]))["AAA"].loc["2020-01-03"] == 500.0


def test_no_filing_is_zero_not_missing():
    """Genuinely no insider activity is information, unlike a missing fundamental."""
    dates = pd.bdate_range("2020-01-01", periods=5)
    panel = to_panel(_trades([]), dates, pd.Index(["AAA", "BBB"]))
    assert (panel == 0.0).all().all()
    assert panel.notna().all().all()


def test_routine_classification_uses_only_prior_history():
    """An insider is 'routine' only once they have ALREADY repeated the pattern."""
    rows = [("AAA", f"{y}-03-05", f"{y}-03-02", "P", 100.0, 10.0, 1000.0, "Officer", "CEO")
            for y in (2017, 2018, 2019, 2020, 2021)]
    flagged = classify_routine(_trades(rows), min_years=3)

    marks = flagged.sort_values("filing_date")["opportunistic"].tolist()
    assert marks[:3] == [True, True, True], "cannot be routine before the pattern exists"
    assert marks[-1] is False or marks[-1] == False, "routine once repeated enough"


def test_a_one_off_trade_is_never_routine():
    rows = [("AAA", "2020-03-05", "2020-03-02", "P", 100.0, 10.0, 1000.0, "Officer", "CEO")]
    assert classify_routine(_trades(rows))["opportunistic"].all()


def test_officer_filter_keeps_seniors_and_drops_others():
    rows = [
        ("AAA", "2020-01-03", "2020-01-02", "P", 1.0, 1.0, 1.0, "Officer", "Chief Executive Officer"),
        ("BBB", "2020-01-03", "2020-01-02", "P", 1.0, 1.0, 1.0, "Director", ""),
        ("CCC", "2020-01-03", "2020-01-02", "P", 1.0, 1.0, 1.0, "10% Owner", "Beneficial holder"),
    ]
    kept = set(officer_filter(_trades(rows))["ticker"])
    assert {"AAA", "BBB"} <= kept
    assert "CCC" not in kept


# --------------------------------------------------------------------- sleeve
def test_insider_flow_decays_after_a_filing():
    dates = pd.bdate_range("2020-01-01", periods=80)
    flow = pd.DataFrame(0.0, index=dates, columns=["AAA"])
    flow.iloc[10, 0] = 1e6
    cap = pd.DataFrame(1e9, index=dates, columns=["AAA"])

    signal = insider_flow_fast(flow, cap, halflife=20.0)["AAA"]
    assert signal.iloc[10] > 0
    assert signal.iloc[30] < signal.iloc[10], "must decay"
    assert signal.iloc[30] == pytest.approx(signal.iloc[10] * 0.5, rel=0.15), "~half after one half-life"
    assert signal.iloc[9] == 0.0, "nothing before the filing"


def test_fast_and_windowed_flow_agree_in_shape():
    dates = pd.bdate_range("2020-01-01", periods=60)
    rng = np.random.default_rng(0)
    flow = pd.DataFrame(rng.normal(0, 1e5, (60, 1)), index=dates, columns=["AAA"])
    cap = pd.DataFrame(1e9, index=dates, columns=["AAA"])

    slow = insider_flow(flow, cap, window=60, halflife=20.0)["AAA"]
    fast = insider_flow_fast(flow, cap, halflife=20.0)["AAA"]
    assert slow.iloc[30:].corr(fast.iloc[30:]) > 0.98


def test_sells_score_below_buys():
    dates = pd.bdate_range("2020-01-01", periods=30)
    flow = pd.DataFrame(0.0, index=dates, columns=["BUY", "SELL"])
    flow.iloc[5, 0] = 1e6
    flow.iloc[5, 1] = -1e6
    cap = pd.DataFrame(1e9, index=dates, columns=["BUY", "SELL"])
    signal = insider_flow_fast(flow, cap).iloc[10]
    assert signal["BUY"] > 0 > signal["SELL"]


def test_net_buy_count_ignores_trade_size():
    dates = pd.bdate_range("2020-01-01", periods=20)
    trades = _trades([
        ("BIG", "2020-01-03", "2020-01-02", "P", 1_000_000.0, 1.0, 1e6, "Officer", "CEO"),
        ("MANY", "2020-01-03", "2020-01-02", "P", 1.0, 1.0, 1.0, "Officer", "CEO"),
        ("MANY", "2020-01-03", "2020-01-02", "P", 1.0, 1.0, 1.0, "Director", ""),
    ])
    counts = net_buy_count(trades, dates, pd.Index(["BIG", "MANY"]), halflife=20.0)
    assert counts["MANY"].iloc[5] > counts["BIG"].iloc[5], "two buyers beat one big buyer"


# ------------------------------------------------------------------- earnings
def test_event_dates_mark_only_new_reports():
    dates = pd.bdate_range("2020-01-01", periods=10)
    income = pd.DataFrame({"A": [np.nan, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0]}, index=dates)
    marks = earnings_event_dates(income)["A"].tolist()
    assert marks == [False, True, False, False, True, False, False, False, True, False]


def test_announcement_reversal_fades_and_opposes_the_move():
    dates = pd.bdate_range("2020-01-01", periods=20)
    close = pd.DataFrame({"A": np.concatenate([np.full(5, 100.0), np.full(15, 110.0)])}, index=dates)
    income = pd.DataFrame({"A": [np.nan] * 6 + [1.0] * 14, "B": [np.nan] * 20}, index=dates)
    close["B"] = 100.0

    signal = announcement_reversal(close, income, reaction_window=3, decay_days=5)["A"]
    active = signal.dropna()
    assert active.iloc[0] < 0, "price rose into the report, so fade it"
    assert abs(active.iloc[2]) < abs(active.iloc[0]), "must decay"


def test_event_density_reports_how_often_the_clock_ticks():
    dates = pd.bdate_range("2020-01-01", periods=10)
    tickers = pd.Index(["A", "B"])
    events = pd.DataFrame(False, index=dates, columns=tickers)
    events.iloc[3, 0] = True
    members = pd.DataFrame(True, index=dates, columns=tickers)
    stats = event_density(events, members)
    assert stats["total_events"] == 1.0
    assert stats["days_with_any_event"] == pytest.approx(0.1)


# -------------------------------------------------------------- registry/ledger
def test_every_event_sleeve_declares_its_data_requirements():
    assert set(EVENT_ALPHAS) == set(EVENT_REQUIRES)
    assert needs_form4(["insider_flow"]) and not needs_form4(["revenue_surprise"])


def test_ledger_counts_and_deduplicates(tmp_path):
    path = tmp_path / "ledger.json"
    ledger = TrialLedger(path).record("a", 5, "first").record("b", 3, "second")
    assert ledger.total == 8

    ledger.record("a", 7, "first, corrected")
    assert ledger.total == 10, "re-recording a key replaces it rather than double-counting"
    assert TrialLedger(path).total == 10, "must survive a reload"


def test_ledger_rejects_negative_counts(tmp_path):
    with pytest.raises(ValueError):
        TrialLedger(tmp_path / "l.json").record("a", -1, "nope")
