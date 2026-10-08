"""Signal construction (spec S0: a priority test target).

The bugs worth guarding against here are silent ones: a sign flip that turns the
strategy into momentum, an off-by-one that lets the signal see the return it is
predicting, and a standardization that leaks across dates.
"""

import numpy as np
import pandas as pd
import pytest

from quant_capstone.signals import (
    demean,
    rank_normal,
    reversal_signal,
    standardize,
    trailing_return,
    winsorize_mad,
    zscore,
)
from quant_capstone.signals.build import signal_diagnostics


def test_reversal_is_the_negative_of_the_trailing_return():
    close = pd.DataFrame(
        {"A": [100.0, 110.0, 121.0], "B": [100.0, 90.0, 81.0]},
        index=pd.bdate_range("2020-01-01", periods=3),
    )
    signal = reversal_signal(close, k=2)

    # A rose 21% over two days, B fell 19%. Reversal buys the loser.
    assert signal.iloc[-1]["A"] == pytest.approx(-0.21)
    assert signal.iloc[-1]["B"] == pytest.approx(0.19)
    assert signal.iloc[-1]["B"] > signal.iloc[-1]["A"]


def test_trailing_return_uses_only_past_and_present():
    close = pd.DataFrame({"A": np.arange(1.0, 11.0)}, index=pd.bdate_range("2020-01-01", periods=10))
    original = trailing_return(close, k=3)

    # Rewriting the future must not change any earlier value of the signal.
    tampered = close.copy()
    tampered.iloc[6:] *= 100.0
    assert original.iloc[:6].equals(trailing_return(tampered, k=3).iloc[:6])


def test_first_k_rows_are_undefined():
    close = pd.DataFrame({"A": np.arange(1.0, 11.0)}, index=pd.bdate_range("2020-01-01", periods=10))
    signal = reversal_signal(close, k=5)
    assert signal.iloc[:5]["A"].isna().all()
    assert signal.iloc[5:]["A"].notna().all()


def test_reversal_rejects_nonpositive_k():
    close = pd.DataFrame({"A": [1.0, 2.0]})
    with pytest.raises(ValueError):
        reversal_signal(close, k=0)


def test_zscore_is_row_wise():
    frame = pd.DataFrame([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    z = zscore(frame)
    assert z.mean(axis=1).abs().max() < 1e-12
    assert z.std(axis=1, ddof=1).sub(1.0).abs().max() < 1e-12
    # Two rows that differ only by scale must standardize identically.
    assert z.iloc[0].to_numpy() == pytest.approx(z.iloc[1].to_numpy())


def test_transforms_do_not_leak_across_dates():
    frame = pd.DataFrame(np.random.default_rng(0).normal(size=(20, 8)))
    before = standardize(frame, "zscore")

    tampered = frame.copy()
    tampered.iloc[10:] += 1000.0
    after = standardize(tampered, "zscore")
    pd.testing.assert_frame_equal(before.iloc[:10], after.iloc[:10])


def test_zscore_handles_a_constant_cross_section():
    frame = pd.DataFrame([[5.0, 5.0, 5.0]])
    assert zscore(frame).isna().all().all()


def test_winsorize_clips_outliers_without_reordering_the_middle():
    row = pd.DataFrame([[-100.0, -1.0, 0.0, 1.0, 2.0, 100.0]])
    clipped = winsorize_mad(row, n_mad=3.0)

    assert clipped.iloc[0, 0] > -100.0
    assert clipped.iloc[0, -1] < 100.0
    middle = clipped.iloc[0, 1:5].to_numpy()
    assert np.all(np.diff(middle) > 0)


def test_winsorize_leaves_a_degenerate_cross_section_alone():
    # Zero MAD must not collapse every name onto the median.
    row = pd.DataFrame([[1.0, 1.0, 1.0, 1.0, 9.0]])
    assert winsorize_mad(row, 3.0).iloc[0, -1] == 9.0


def test_rank_normal_depends_only_on_ordering():
    a = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]])
    b = pd.DataFrame([[1.0, 100.0, 1000.0, 10000.0]])
    assert rank_normal(a).to_numpy() == pytest.approx(rank_normal(b).to_numpy())


def test_demean_gives_a_zero_sum_cross_section():
    frame = pd.DataFrame(np.random.default_rng(1).normal(size=(5, 9)))
    assert demean(frame).sum(axis=1).abs().max() < 1e-12


def test_built_signal_is_standardized_and_masked(signal, dataset):
    diagnostics = signal_diagnostics(signal)
    assert diagnostics["max_abs_cross_sectional_sum"] < 1e-8
    assert diagnostics["mean_cross_sectional_std"] == pytest.approx(1.0, abs=1e-6)
    # Nothing outside the point-in-time universe may carry a score.
    assert not (signal.notna() & ~dataset.members).to_numpy().any()


def test_signal_predicts_the_planted_reversal(signal, dataset, cfg):
    """The synthetic panel has a known reversal; the pipeline must recover it."""
    from quant_capstone.backtest import forward_returns
    from quant_capstone.law import information_coefficient

    fwd = forward_returns(dataset.prices, cfg.execution.timing)
    ic = information_coefficient(signal, fwd).dropna()
    t_stat = ic.mean() / ic.std(ddof=1) * np.sqrt(len(ic))
    assert ic.mean() > 0
    assert t_stat > 5
