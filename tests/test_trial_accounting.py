"""The trial count and the walk-forward grid: the two things the Deflated Sharpe
and the out-of-sample claim rest on.

Both failed quietly before. ``k`` was chosen with full-sample hindsight and then
pinned in the config, while the README said it was re-selected per fold. And the
headline grid was counted twice -- once from the ledger and once from the run
cache -- while the defaults, being a grid cell too, were run and counted again.
"""

import pandas as pd
import pytest

from quant_capstone.cli import QUICK_GRID
from quant_capstone.config import Config
from quant_capstone.validation import (
    DEFAULT_GRID,
    WALKFORWARD_KEY,
    RunCache,
    TrialLedger,
    make_folds,
    walk_forward,
)


def _default(key: str):
    section, attr = key.split("__")
    return getattr(getattr(Config(), section), attr)


# ------------------------------------------------------------------ the grid
def test_k_is_reselected_per_fold_not_pinned():
    assert "signal__k" in DEFAULT_GRID
    assert DEFAULT_GRID["signal__k"] == [1, 2, 3, 5, 10], "the spec's S4 set, not one centred on 2"


@pytest.mark.parametrize("grid", [DEFAULT_GRID, QUICK_GRID], ids=["default", "quick"])
def test_every_grid_contains_the_defaults(grid):
    """Otherwise the 'fixed defaults' comparison is against a config the
    selection could never have picked."""
    for key, values in grid.items():
        assert _default(key) in values, f"default {key}={_default(key)} missing from grid"


def test_embargo_covers_the_longest_hold_in_the_grid(dataset, cfg):
    """A fold that selects rebalance_every=15 holds for fifteen days, so ten
    days of embargo would let its positions straddle the boundary."""
    grid = {"execution__rebalance_every": [5, 25]}
    result = walk_forward(cfg, dataset, grid)
    first = result.folds.iloc[0]
    dates = result.cache.returns().index
    expected = make_folds(dates, cfg.validation.train_days, cfg.validation.test_days,
                          cfg.validation.expanding, embargo=25)[0]
    assert pd.Timestamp(first["train_end"]) == expected.train_end


# --------------------------------------------------------------- run cache
def test_identical_configs_share_one_run_and_one_trial(dataset, cfg):
    cache = RunCache(dataset)
    cache.evaluate(cfg, {"signal__k": [cfg.signal.k, 10]})
    cache.evaluate(cfg, {})  # the walk-forward's 'base' entry: same config as a cell

    assert cache.n_trials == 2, "the defaults are one configuration, not two"
    assert cache.runs["base"] is cache.runs[f"k={cfg.signal.k}"]
    assert cache.returns().shape[1] == 2, "aliases must not enter PBO twice"


# ------------------------------------------------------------------ ledger
def test_superseded_sweeps_stay_on_record_but_stop_counting(tmp_path):
    ledger = TrialLedger(tmp_path / "l.json")
    ledger.record("old_grid", 27, "27-cell grid").record("probe", 5, "probe")
    ledger.record(WALKFORWARD_KEY, 135, "the 135-cell grid containing old_grid")
    ledger.supersede(["old_grid"], by=WALKFORWARD_KEY)

    reloaded = TrialLedger(tmp_path / "l.json")
    assert "old_grid" in reloaded.entries, "history is kept"
    assert reloaded.total == 140
    assert reloaded.total_excluding(WALKFORWARD_KEY) == 5


def test_supersede_rejects_unknown_keys(tmp_path):
    with pytest.raises(KeyError):
        TrialLedger(tmp_path / "l.json").supersede(["nope"], by=WALKFORWARD_KEY)


def test_the_committed_ledger_does_not_double_count_the_headline_grid():
    ledger = TrialLedger()  # results/trial_ledger.json, version-controlled
    assert WALKFORWARD_KEY in ledger.entries
    grid_cells = 1
    for values in DEFAULT_GRID.values():
        grid_cells *= len(values)
    assert ledger.entries[WALKFORWARD_KEY].n_trials == grid_cells
    assert ledger.total == ledger.total_excluding(WALKFORWARD_KEY) + grid_cells
