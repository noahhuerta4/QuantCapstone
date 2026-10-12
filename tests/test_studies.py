"""The study registry and its reproducibility plumbing.

The studies themselves need the real data cache and minutes of compute, so they
are verified by ``quant-capstone study --check`` rather than here. What is cheap
to pin down is the wiring: every registered study is importable and runnable,
each era really does pin what changed, and the comparison flags a real diff.
"""

import pandas as pd
import pytest

from quant_capstone.config import Config
from quant_capstone.studies import ORDER, compare_dirs, eras
from quant_capstone.studies.runner import _writes_index, module


@pytest.mark.parametrize("name", ORDER)
def test_every_study_is_registered_and_runnable(name):
    mod = module(name)
    assert mod.DESCRIPTION and callable(mod.run)


def test_figures_runs_after_the_studies_it_reads():
    figures = ORDER.index("figures")
    for need in module("figures").NEEDS:
        assert ORDER.index(need) < figures


def test_unknown_study_is_an_error():
    with pytest.raises(KeyError):
        module("nope")


def test_historical_eras_pin_what_has_changed_since():
    final = Config()
    for era in (eras.FLAT_SPREAD, eras.BLEND, eras.PEAD):
        assert era.execution.no_trade_band == 0.003 != final.execution.no_trade_band
        assert era.costs.enable_financing is False, "financing post-dates every earlier era"
    assert eras.FLAT_SPREAD.costs.use_estimated_spread is False
    assert eras.BLEND.signal.pead_weight == 0.0
    assert eras.FINAL == final


def test_labelled_indexes_are_written_and_integer_ones_are_not():
    assert _writes_index(pd.DataFrame({"a": [1]}, index=["Q1"]))
    assert _writes_index(pd.DataFrame({"a": [1]}, index=pd.Index([0.5], name="adverse")))
    assert not _writes_index(pd.DataFrame({"a": [1, 2]}, index=[1, 0]))  # sorted, unlabelled


def test_compare_flags_a_changed_number_and_passes_an_identical_one(tmp_path):
    ref, new = tmp_path / "ref", tmp_path / "new"
    ref.mkdir(), new.mkdir()
    frame = pd.DataFrame({"x": ["a", "b"], "net": [0.5, 0.25]})
    frame.to_csv(ref / "same.csv", index=False)
    frame.to_csv(new / "same.csv", index=False)
    frame.to_csv(ref / "moved.csv", index=False)
    frame.assign(net=[0.5, 0.26]).to_csv(new / "moved.csv", index=False)

    report = compare_dirs(new, ref).set_index("table")["status"]
    assert report["same"] == "match"
    assert report["moved"].startswith("differs")
