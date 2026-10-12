"""How much each sleeve family adds, measured with one harness.

The project's history reads single sleeve -> blend -> blend + PEAD, but each of
those numbers was recorded under the harness of its day: different grids, no
embargo, a trial count that later turned out to double-count. They are not
comparable with each other or regenerable now. This runs all three stages
through *today's* validation -- same walk-forward, same embargo, same grid axes
(minus ``pead_weight`` where there is no PEAD sleeve), same trial ledger -- so
the increments mean what they claim.

The reversal-only stage is a fresh look at the data (its cells are not in the
headline grid) and is recorded in the trial ledger as ``ablation_reversal_only``.
The blend stage is the ``pead_weight = 0`` slice of the headline grid, so it
adds nothing to the count.
"""

from __future__ import annotations

import tempfile

import pandas as pd

from .. import validation as v
from ..config import Config
from ..data import load_dataset
from ..study import run_study
from . import eras

DESCRIPTION = "Ablation: reversal only -> blend -> blend + PEAD, one harness"
ABLATION_KEY = "ablation_reversal_only"
PRICE_GRID = {k: g for k, g in v.DEFAULT_GRID.items() if k != "signal__pead_weight"}
#: stage -> (config, grid, the ledger entry that already counts this grid)
STAGES: dict[str, tuple[Config, dict, str]] = {
    "reversal only": (eras.FINAL.override(signal__alphas=("reversal",), signal__pead_weight=0.0),
                      PRICE_GRID, ABLATION_KEY),
    "blend": (eras.FINAL.override(signal__pead_weight=0.0), PRICE_GRID, v.WALKFORWARD_KEY),
    "blend + PEAD": (eras.FINAL, v.DEFAULT_GRID, v.WALKFORWARD_KEY),
}


def _metrics(study) -> dict[str, float]:
    perf = study.performance.loc["sharpe"]
    stab = study.walk_forward.stability()
    return {
        "oos_net_sharpe": perf["OOS_net_selected"],
        "oos_net_default": perf["OOS_net_default"],
        "full_net_sharpe": perf["full_net"],
        "full_gross_sharpe": perf["full_gross"],
        "deflated_sharpe": study.deflated["deflated_sharpe_ratio"],
        "n_trials": study.n_trials,
        "pbo": study.pbo["pbo"],
        "ci_low": study.bootstrap["ci_low"],
        "ci_high": study.bootstrap["ci_high"],
        "prob_sharpe_below_zero": study.bootstrap["prob_sharpe_below_zero"],
        "folds_positive": stab["folds_positive"],
        "train_minus_test": stab["mean_train_minus_test"],
        "break_even_cost_multiplier": study.break_even,
    }


def run() -> dict[str, pd.DataFrame]:
    ledger = v.TrialLedger()
    rows = {}
    with tempfile.TemporaryDirectory() as scratch:
        for stage, (cfg, grid, own_key) in STAGES.items():
            # Every stage deflates against the whole project's count, with its
            # own grid counted once -- by the run, not also by the ledger.
            prior = ledger.total_excluding(v.WALKFORWARD_KEY, own_key)
            study = run_study(cfg, scratch, grid=grid, dataset=load_dataset(cfg),
                              make_plots=False, prior_trials=prior)
            rows[stage] = _metrics(study)
    return {"progression": pd.DataFrame(rows)}
