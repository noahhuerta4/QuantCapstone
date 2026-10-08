"""Overfitting controls (spec S10).

These test the statistics themselves against cases where the right answer is
known — pure noise, a planted edge, an exact interpolation — rather than just
checking that the functions return a number.
"""

import numpy as np
import pandas as pd
import pytest

from quant_capstone.config import Config
from quant_capstone.validation import (
    block_bootstrap_sharpe,
    break_even_multiplier,
    deflated_sharpe,
    expected_max_sharpe,
    grid_configs,
    make_folds,
    minimum_track_record_length,
    plateau_score,
    probabilistic_sharpe,
    probability_of_backtest_overfitting,
)


# --------------------------------------------------------------------- folds
def test_folds_are_chronological_and_do_not_overlap():
    dates = pd.bdate_range("2015-01-01", periods=1500)
    folds = make_folds(dates, train_days=500, test_days=100)

    assert len(folds) > 5
    for fold in folds:
        assert fold.train_end < fold.test_start, "train must end before test begins"
    for earlier, later in zip(folds, folds[1:]):
        assert earlier.test_end < later.test_start, "test windows must not overlap"


def test_expanding_and_rolling_windows_differ():
    dates = pd.bdate_range("2015-01-01", periods=1200)
    expanding = make_folds(dates, 400, 100, expanding=True)
    rolling = make_folds(dates, 400, 100, expanding=False)
    assert expanding[-1].train_start < rolling[-1].train_start


def test_no_folds_when_history_is_too_short():
    assert make_folds(pd.bdate_range("2020-01-01", periods=50), 500, 100) == []


# ------------------------------------------------------------ deflated sharpe
def test_expected_max_sharpe_grows_with_the_number_of_trials():
    variance = 0.01
    values = [expected_max_sharpe(n, variance) for n in (2, 10, 100, 1000)]
    assert all(a < b for a, b in zip(values, values[1:]))


def test_a_single_trial_needs_no_deflation():
    assert expected_max_sharpe(1, 0.01) == 0.0


def test_deflated_sharpe_falls_as_more_configurations_are_tried():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0006, 0.01, 1500))
    trials = pd.Series(rng.normal(0.03, 0.05, 50))

    few = deflated_sharpe(returns, 2, trials)["deflated_sharpe_ratio"]
    many = deflated_sharpe(returns, 500, trials)["deflated_sharpe_ratio"]
    assert many < few


def test_probabilistic_sharpe_rises_with_a_better_observed_sharpe():
    weak = probabilistic_sharpe(0.02, 0.0, 1000, 0.0, 3.0)
    strong = probabilistic_sharpe(0.10, 0.0, 1000, 0.0, 3.0)
    assert 0.0 <= weak < strong <= 1.0


def test_negative_skew_and_fat_tails_penalise_the_sharpe():
    normal = probabilistic_sharpe(0.06, 0.0, 1000, 0.0, 3.0)
    skewed = probabilistic_sharpe(0.06, 0.0, 1000, -1.5, 9.0)
    assert skewed < normal


def test_minimum_track_record_length_is_longer_for_weaker_signals():
    strong = minimum_track_record_length(0.10, 0.0, 3.0)
    weak = minimum_track_record_length(0.02, 0.0, 3.0)
    assert weak > strong > 0


def test_deflated_sharpe_declines_to_report_on_too_little_data():
    assert deflated_sharpe(pd.Series([0.01, -0.01, 0.02]), 5).empty


# ------------------------------------------------------------------------ PBO
def test_pbo_is_about_a_half_for_indistinguishable_configurations():
    rng = np.random.default_rng(42)
    noise = pd.DataFrame(rng.normal(0.0, 0.01, (1200, 8)))
    pbo = probability_of_backtest_overfitting(noise, n_blocks=10)["pbo"]
    assert 0.3 < pbo < 0.7, "selecting among identical strategies should be a coin flip"


def test_pbo_is_low_when_one_configuration_is_genuinely_better():
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(rng.normal(0.0, 0.01, (1200, 6)))
    frame[0] += 0.0015  # a real, persistent edge
    pbo = probability_of_backtest_overfitting(frame, n_blocks=10)["pbo"]
    assert pbo < 0.2


def test_pbo_needs_more_than_one_configuration():
    single = pd.DataFrame({"only": np.random.default_rng(0).normal(size=500)})
    assert np.isnan(probability_of_backtest_overfitting(single)["pbo"])


# ------------------------------------------------------------------ bootstrap
def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0.0005, 0.01, 1500))
    stats = block_bootstrap_sharpe(returns, n_samples=500, block=21, seed=0)
    assert stats["ci_low"] < stats["sharpe"] < stats["ci_high"]
    assert stats["ci_width"] > 0


def test_a_stronger_signal_shifts_the_interval_upward():
    rng = np.random.default_rng(2)
    weak = pd.Series(rng.normal(0.0002, 0.01, 1500))
    strong = pd.Series(rng.normal(0.0020, 0.01, 1500))
    assert (
        block_bootstrap_sharpe(strong, 500, 21, seed=0)["ci_low"]
        > block_bootstrap_sharpe(weak, 500, 21, seed=0)["ci_low"]
    )


def test_bootstrap_is_reproducible_for_a_given_seed():
    returns = pd.Series(np.random.default_rng(3).normal(0.0005, 0.01, 800))
    a = block_bootstrap_sharpe(returns, 300, 21, seed=11)
    b = block_bootstrap_sharpe(returns, 300, 21, seed=11)
    pd.testing.assert_series_equal(a, b)


def test_bootstrap_declines_to_report_on_too_little_data():
    assert block_bootstrap_sharpe(pd.Series([0.01] * 10), 100, 21).empty


# ------------------------------------------------------------ cost break-even
def test_break_even_interpolates_the_zero_crossing():
    sweep = pd.DataFrame({"net_sharpe": [2.0, 1.0, -1.0]}, index=[1.0, 2.0, 3.0])
    sweep.index.name = "cost_multiplier"
    # Between 2.0 and 3.0 the Sharpe falls 1.0 -> -1.0, crossing zero at 2.5.
    assert break_even_multiplier(sweep) == pytest.approx(2.5)


def test_break_even_is_infinite_when_the_edge_survives_the_whole_sweep():
    sweep = pd.DataFrame({"net_sharpe": [2.0, 1.8, 1.5]}, index=[1.0, 2.0, 3.0])
    assert np.isinf(break_even_multiplier(sweep))


def test_break_even_is_undefined_when_there_is_no_edge_to_begin_with():
    sweep = pd.DataFrame({"net_sharpe": [-0.5, -1.0]}, index=[1.0, 2.0])
    assert np.isnan(break_even_multiplier(sweep))


# -------------------------------------------------------------------- plateau
def test_plateau_scores_higher_than_a_cliff():
    plateau = np.array([[1.9, 2.0, 1.95], [1.92, 2.01, 1.97]])
    cliff = np.array([[0.1, 2.0, 0.05], [0.02, 0.1, 0.08]])
    assert (
        plateau_score(plateau)["fraction_within_tolerance"]
        > plateau_score(cliff)["fraction_within_tolerance"]
    )
    assert plateau_score(cliff)["peak_over_median"] > plateau_score(plateau)["peak_over_median"]


# --------------------------------------------------------------------- config
def test_grid_expands_to_the_full_cartesian_product():
    combos = grid_configs(Config(), {"signal__k": [1, 5], "risk__vol_target": [0.1, 0.2]})
    assert len(combos) == 4
    assert {(p["signal__k"], p["risk__vol_target"]) for p, _ in combos} == {
        (1, 0.1), (1, 0.2), (5, 0.1), (5, 0.2)
    }


def test_grid_configs_actually_apply_the_overrides():
    (_, cfg), = grid_configs(Config(), {"signal__k": [7]})
    assert cfg.signal.k == 7


def test_empty_grid_yields_the_base_configuration():
    combos = grid_configs(Config(), {})
    assert len(combos) == 1 and combos[0][0] == {}


def test_override_is_non_mutating_and_validated():
    base = Config()
    modified = base.override(signal__k=9, costs__cost_multiplier=3.0)

    assert base.signal.k != 9, "override must not mutate the original"
    assert modified.signal.k == 9
    assert modified.costs.cost_multiplier == 3.0
    assert modified.risk == base.risk, "untouched sections are shared unchanged"

    with pytest.raises(KeyError):
        base.override(signal__not_a_field=1)
    with pytest.raises(KeyError):
        base.override(nonexistent__field=1)


# ------------------------------------------------------- walk-forward wiring
def test_walk_forward_compares_against_the_real_defaults(dataset, cfg):
    """Regression: with a non-empty grid there was no entry named ``base``, so
    the 'fixed defaults' column silently reported an arbitrary grid corner."""
    from quant_capstone.validation import label, walk_forward

    result = walk_forward(cfg, dataset, {"signal__k": [3, 10]})

    assert label({}) in result.cache.runs, "the base config must always be evaluated"
    base_returns = result.cache.returns([label({})]).iloc[:, 0]
    expected = base_returns.reindex(result.oos_returns_default.index)
    pd.testing.assert_series_equal(
        result.oos_returns_default, expected, check_names=False
    )


def test_walk_forward_selects_from_the_grid_each_fold(dataset, cfg):
    from quant_capstone.validation import walk_forward

    result = walk_forward(cfg, dataset, {"signal__k": [3, 10]})
    assert len(result.folds) > 1
    assert result.folds["selected"].isin(result.cache.runs).all()
    # Test windows must tile the OOS series exactly once.
    assert len(result.oos_returns) == result.folds["n_test_days"].sum()
    assert result.oos_returns.index.is_monotonic_increasing
    assert not result.oos_returns.index.duplicated().any()


def test_embargo_purges_the_end_of_training():
    """A position opened near the end of training is still open into the test
    window; without an embargo its return lands in both sets."""
    dates = pd.bdate_range("2015-01-01", periods=1200)
    plain = make_folds(dates, 500, 100, embargo=0)
    purged = make_folds(dates, 500, 100, embargo=10)

    assert len(plain) == len(purged)
    for a, b in zip(plain, purged):
        assert b.train_end < a.train_end, "training must end earlier"
        assert b.test_start == a.test_start, "test window must be untouched"
        gap = (b.test_start - b.train_end).days
        assert gap > (a.test_start - a.train_end).days


def test_embargo_never_leaves_a_degenerate_training_window():
    dates = pd.bdate_range("2015-01-01", periods=400)
    folds = make_folds(dates, 150, 100, embargo=120)
    for f in folds:
        assert f.train_start < f.train_end < f.test_start <= f.test_end
