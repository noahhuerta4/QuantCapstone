"""The README and the write-up must quote the numbers the last real run produced.

Both drifted before: a README heading announced one Sharpe while the table under
it showed another, the trial count was quoted three different ways, and the test
count was two phases out of date. Prose cannot be regenerated, but it can be
checked. ``results/headline.json`` is written by every full ``quant-capstone
run``; each number below must appear, formatted as the documents quote it.
"""

from pathlib import Path

import pytest

from quant_capstone.reporting import book
from quant_capstone.reporting.headline import read_headline

README = Path("README.md").read_text(encoding="utf-8")
BOOK = book.text()
H = read_headline()


def _expected() -> dict[str, list[str]]:
    lo, hi = H["bootstrap_ci"]
    common = [f"{H['n_trials']} configurations", f"[{lo:.2f}, {hi:.2f}]"]
    return {
        "README": common + [
            f"Net out-of-sample Sharpe: {H['oos_net_sharpe']:.2f}",
            f"**{H['oos_net_sharpe']:.2f}** |",
            f"**{H['deflated_sharpe']:.2f}** ({H['n_trials']} trials)",
            f"**{H['pbo']:.2f}** |",
            f"**{H['folds_positive'] / H['n_folds']:.0%}** |",
            f"**{H['break_even_cost_multiplier']:.1f}x** |",
            f"{H['oos_net_default']:.2f} against {H['oos_net_sharpe']:.2f}",
            f"selection wins only {H['folds_selection_beat_default']} of {H['n_folds']} folds",
        ],
        "book": common + [
            f"{H['oos_net_sharpe']:.2f} Sharpe, net OOS",
            f"Walk-forward OOS Sharpe {H['oos_net_sharpe']:.3f}",
            f"Folds positive {H['folds_positive']} of {H['n_folds']}",
            f"Fixed-default OOS Sharpe {H['oos_net_default']:.3f}",
            f"Deflated Sharpe ({H['n_trials']} trials) {H['deflated_sharpe']:.3f}",
            f"PBO {H['pbo']:.3f}",
            f"Cost break-even {H['break_even_cost_multiplier']:.2f}×",
            f"Min. track record length {H['min_track_record_days']} days",
        ],
    }


DOCS = {"README": README, "book": BOOK}
CASES = [(doc, phrase) for doc, phrases in _expected().items() for phrase in phrases]


@pytest.mark.parametrize("doc,phrase", CASES, ids=[f"{d}:{p[:40]}" for d, p in CASES])
def test_documents_quote_the_headline(doc, phrase):
    assert phrase in DOCS[doc], f"{doc} does not say {phrase!r} -- out of date with results/headline.json"


STALE = ["327 configurations", "327 trials", "179 tests", "0.338 and 0.047", "output/trial_ledger.json"]


@pytest.mark.parametrize("doc", DOCS)
@pytest.mark.parametrize("phrase", STALE)
def test_documents_drop_superseded_numbers(doc, phrase):
    assert phrase not in DOCS[doc]


def test_quoted_test_count_is_current(request):
    """Only checkable when the whole suite is collected."""
    if any(not str(a).startswith("-") for a in request.config.invocation_params.args):
        pytest.skip("partial run; the collected count is not the suite size")
    n = request.session.testscollected
    for name, text in DOCS.items():
        assert f"{n} tests" in text, f"{name} should say '{n} tests'"
