"""Fundamental Law diagnostics (spec S7)."""

import numpy as np
import pandas as pd
import pytest

from quant_capstone.law import (
    analyze,
    autocorrelation_adjustment,
    breadth_implied_by_performance,
    cost_transfer,
    information_coefficient,
    participation_ratio,
    required_ic,
    residual_correlation,
    transfer_coefficient,
)


def test_participation_ratio_counts_independent_directions():
    n = 20
    assert participation_ratio(np.eye(n)) == pytest.approx(n)
    # Perfectly correlated names are one bet wearing n hats.
    assert participation_ratio(np.ones((n, n))) == pytest.approx(1.0, abs=1e-6)


def test_participation_ratio_falls_as_correlation_rises():
    n = 30
    low = 0.1 * np.ones((n, n)) + 0.9 * np.eye(n)
    high = 0.8 * np.ones((n, n)) + 0.2 * np.eye(n)
    assert participation_ratio(high) < participation_ratio(low) < n


def test_residual_correlation_survives_a_leading_all_nan_row():
    """Regression test: ``returns()`` always starts with an all-NaN row, and
    dropping columns with any NaN before dropping it deletes the whole panel."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(300, 12)))
    frame.iloc[0] = np.nan

    corr = residual_correlation(frame, n_factors=1)
    assert corr.shape == (12, 12)
    assert participation_ratio(corr) > 5


def test_removing_the_market_factor_raises_effective_breadth():
    rng = np.random.default_rng(1)
    n, t = 25, 600
    market = rng.normal(0, 0.01, (t, 1))
    returns = pd.DataFrame(market + rng.normal(0, 0.004, (t, n)))

    raw = participation_ratio(np.corrcoef(returns.to_numpy(), rowvar=False))
    residual = participation_ratio(residual_correlation(returns, n_factors=1))
    assert residual > raw


def test_autocorrelation_adjustment_shrinks_persistent_signals():
    assert autocorrelation_adjustment(0.0) == pytest.approx(1.0)
    assert autocorrelation_adjustment(0.8) == pytest.approx(1 / 9, rel=1e-9)
    assert autocorrelation_adjustment(0.95) < autocorrelation_adjustment(0.5)


def test_required_ic_shrinks_with_the_square_root_of_breadth():
    assert required_ic(2.0, 10_000) == pytest.approx(0.02)
    # Quartering breadth doubles the IC you need.
    assert required_ic(2.0, 2_500) == pytest.approx(0.04)


def test_required_ic_rises_as_the_transfer_coefficient_falls():
    assert required_ic(2.0, 10_000, 0.5) == pytest.approx(2 * required_ic(2.0, 10_000, 1.0))


def test_the_seductive_lie_is_arithmetically_real():
    """A trivial IC really does imply IR 2 under the naive bet count -- which is
    exactly why the naive bet count has to be wrong."""
    assert required_ic(2.0, 252 * 200) == pytest.approx(0.0089, abs=1e-4)


def test_breadth_implied_by_performance_inverts_the_law():
    # IR = TC * IC * sqrt(BR), so BR = (IR / (TC*IC))^2.
    assert breadth_implied_by_performance(2.0, 0.02, 1.0) == pytest.approx(10_000)
    assert np.isnan(breadth_implied_by_performance(2.0, 0.0, 1.0))


def test_information_coefficient_detects_a_planted_relationship():
    rng = np.random.default_rng(3)
    signal = pd.DataFrame(rng.normal(size=(400, 30)))
    forward = signal * 0.05 + rng.normal(0, 0.01, (400, 30))
    assert information_coefficient(signal, forward).mean() > 0.8


def test_information_coefficient_of_unrelated_series_is_near_zero():
    rng = np.random.default_rng(4)
    signal = pd.DataFrame(rng.normal(size=(600, 30)))
    forward = pd.DataFrame(rng.normal(size=(600, 30)))
    assert abs(information_coefficient(signal, forward).mean()) < 0.05


def test_transfer_coefficient_is_one_for_unconstrained_weights():
    rng = np.random.default_rng(5)
    ideal = pd.DataFrame(rng.normal(size=(50, 20)))
    assert transfer_coefficient(ideal, ideal).mean() == pytest.approx(1.0)


def test_transfer_coefficient_falls_when_constraints_bite():
    rng = np.random.default_rng(6)
    ideal = pd.DataFrame(rng.normal(size=(200, 20)))
    clipped = ideal.clip(-0.2, 0.2)
    tc = transfer_coefficient(ideal, clipped).mean()
    assert 0.0 < tc < 1.0


def test_cost_transfer_is_the_net_over_gross_ratio():
    assert cost_transfer(2.0, 1.0) == pytest.approx(0.5)
    assert cost_transfer(2.0, -0.5) == pytest.approx(-0.25), "costs can invert the edge"
    assert np.isnan(cost_transfer(0.0, 1.0))


def test_full_diagnostic_is_internally_consistent(run):
    report = analyze(run)

    assert report.breadth["breadth_effective"] < report.breadth["breadth_naive"]
    assert 0 < report.breadth["independent_period_fraction"] <= 1.0
    assert report.breadth["n_eff_after_factor_removal"] <= report.breadth["mean_names"] + 1e-9
    assert -1.0 <= report.transfer["tc_constraints"] <= 1.0
    assert report.ic["ic_mean"] > 0, "the synthetic panel has a planted reversal"
    assert report.verdict and isinstance(report.verdict, str)
    assert not report.to_frame().empty


def test_seductive_lie_table_required_ic_rises_monotonically(run):
    table = analyze(run).seductive_lie
    ics = table["required_ic"].to_numpy()
    assert np.all(np.diff(ics) > 0), "each correction must make the bar higher"


# ------------------------------------------------------------ IC by bucket
def test_ic_by_bucket_finds_a_planted_gradient():
    """Signal works only in the low-characteristic names; buckets must show it."""
    from quant_capstone.law import bucket_summary, ic_by_bucket

    rng = np.random.default_rng(9)
    n_days, n_names = 500, 60
    signal = pd.DataFrame(rng.normal(size=(n_days, n_names)))
    characteristic = pd.DataFrame(
        np.tile(np.arange(n_names, dtype=float), (n_days, 1))
    )
    # Predictive power decays linearly across the characteristic.
    strength = np.linspace(0.9, 0.0, n_names)
    forward = signal * strength + pd.DataFrame(rng.normal(0, 0.5, (n_days, n_names)))

    buckets = ic_by_bucket(signal, forward, characteristic, n_buckets=5)
    means = buckets.mean()
    assert means["Q1"] > means["Q5"], "edge must be strongest in the low bucket"
    assert list(means.sort_values(ascending=False).index)[0] == "Q1"

    summary = bucket_summary(buckets)
    assert summary.loc["Q1 minus Q5", "ic_mean"] > 0
    assert summary.loc["Q1 minus Q5", "ic_t_stat"] > 3


def test_ic_by_bucket_is_flat_when_there_is_no_gradient():
    from quant_capstone.law import bucket_summary, ic_by_bucket

    rng = np.random.default_rng(10)
    signal = pd.DataFrame(rng.normal(size=(600, 60)))
    forward = signal * 0.4 + pd.DataFrame(rng.normal(0, 0.5, (600, 60)))
    characteristic = pd.DataFrame(rng.normal(size=(600, 60)))

    summary = bucket_summary(ic_by_bucket(signal, forward, characteristic, n_buckets=5))
    assert abs(summary.loc["Q1 minus Q5", "ic_t_stat"]) < 3, "no gradient should test flat"


def test_ic_by_bucket_skips_thin_cross_sections():
    from quant_capstone.law import ic_by_bucket

    rng = np.random.default_rng(11)
    small = pd.DataFrame(rng.normal(size=(50, 6)))
    buckets = ic_by_bucket(small, small, small, n_buckets=5, min_names=8)
    assert buckets.isna().all().all()
