"""The headline numbers of a study, as a small version-controlled JSON file.

``results.md`` is regenerated into the git-ignored ``output/``, so on its own it
leaves no record of what the last real run actually said. This file is that
record. The README and the write-up quote these numbers, and
``tests/test_docs.py`` fails if either drifts from them -- which is how a README
ends up announcing one Sharpe in its heading and a different one in its table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..study import Study

DEFAULT_PATH = Path("results/headline.json")


def headline_numbers(study: Study) -> dict[str, Any]:
    perf = study.performance.loc["sharpe"]
    stability = study.walk_forward.stability()
    folds = study.walk_forward.folds
    return {
        "oos_net_sharpe": round(float(perf["OOS_net_selected"]), 4),
        "oos_net_default": round(float(perf["OOS_net_default"]), 4),
        "full_net_sharpe": round(float(perf["full_net"]), 4),
        "full_gross_sharpe": round(float(perf["full_gross"]), 4),
        "deflated_sharpe": round(float(study.deflated["deflated_sharpe_ratio"]), 4),
        "psr_vs_zero": round(float(study.deflated["psr_vs_zero"]), 4),
        "min_track_record_days": round(float(study.deflated["min_track_record_length_days"])),
        "n_trials": int(study.n_trials),
        "grid_trials": int(study.grid_trials),
        "prior_trials": int(study.prior_trials),
        "pbo": round(float(study.pbo["pbo"]), 4),
        "bootstrap_ci": [round(float(study.bootstrap["ci_low"]), 2),
                         round(float(study.bootstrap["ci_high"]), 2)],
        "prob_sharpe_below_zero": round(float(study.bootstrap["prob_sharpe_below_zero"]), 4),
        "n_folds": int(stability["n_folds"]),
        "folds_positive": int(round(stability["folds_positive"] * stability["n_folds"])),
        "folds_selection_beat_default": int((folds["test_sharpe"] > folds["test_sharpe_default"]).sum()),
        "train_minus_test": round(float(stability["mean_train_minus_test"]), 4),
        "break_even_cost_multiplier": round(float(study.break_even), 2),
        "annual_cost_bps": round(float(study.attribution["total"]), 1),
        "selected_per_fold": folds["selected"].value_counts().to_dict(),
        "grid": {k: list(v) for k, v in _grid(study).items()},
    }


def _grid(study: Study) -> dict[str, list]:
    params = study.sensitivity.drop(
        columns=[c for c in study.sensitivity.columns if "__" not in c]
    )
    return {c: sorted(params[c].unique().tolist()) for c in params.columns}


def write_headline(study: Study, path: Path | str = DEFAULT_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(headline_numbers(study), indent=2) + "\n", encoding="utf-8")
    return path


def read_headline(path: Path | str = DEFAULT_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text())
