"""A durable, honest count of every configuration ever examined.

The Deflated Sharpe Ratio discounts a result for how many things were tried, and
it is only meaningful if that count is real. Holding the number in someone's head
across a multi-week project does not work: exploratory sweeps get forgotten,
rejected candidates get dropped, and the count quietly drifts downward in exactly
the direction that flatters the result.

So the count lives on disk. Every sweep appends an entry with what was tried and
why, and the total is read back whenever a Sharpe is deflated.

**What counts as a trial.** Any configuration whose *result was looked at*. That
includes candidates that failed and were rejected — a sleeve tested and dropped
still consumed a look at the data, and pretending otherwise is the error this
module exists to prevent. It includes exploratory sweeps run outside the final
script. It does not include re-running an identical configuration, which is why
entries carry a key and are deduplicated on it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

DEFAULT_PATH = Path("output/trial_ledger.json")


@dataclass
class TrialEntry:
    key: str
    """Stable identifier, so re-running the same sweep does not double-count."""
    n_trials: int
    description: str
    outcome: str = ""
    """What came of it -- 'adopted', 'rejected', 'exploratory'. Kept because a
    ledger of bare numbers is unauditable six weeks later."""
    recorded: str = field(default_factory=lambda: date.today().isoformat())


class TrialLedger:
    """Append-only record of examined configurations."""

    def __init__(self, path: Path | str = DEFAULT_PATH):
        self.path = Path(path)
        self.entries: dict[str, TrialEntry] = {}
        if self.path.exists():
            payload = json.loads(self.path.read_text())
            self.entries = {e["key"]: TrialEntry(**e) for e in payload.get("entries", [])}

    def record(self, key: str, n_trials: int, description: str, outcome: str = "") -> "TrialLedger":
        """Add or update one sweep. Re-recording the same key overwrites it."""
        if n_trials < 0:
            raise ValueError("n_trials cannot be negative")
        self.entries[key] = TrialEntry(key, int(n_trials), description, outcome)
        self.save()
        return self

    @property
    def total(self) -> int:
        return sum(e.n_trials for e in self.entries.values())

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "total": self.total,
            "entries": [asdict(e) for e in sorted(self.entries.values(), key=lambda e: e.recorded)],
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def to_frame(self):
        import pandas as pd

        if not self.entries:
            return pd.DataFrame(columns=["key", "n_trials", "description", "outcome", "recorded"])
        frame = pd.DataFrame([asdict(e) for e in self.entries.values()])
        return frame.sort_values("recorded").reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        return f"TrialLedger({len(self.entries)} sweeps, {self.total} trials, {self.path})"


#: Everything examined in Phase 1, reconstructed from the project record so the
#: ledger starts from the real number rather than from zero.
PHASE1_TRIALS = [
    ("p1_turnover_sweep_a", 18, "First turnover tuning: band x rebalance x universe", "exploratory"),
    ("p1_turnover_sweep_b", 24, "Turnover tuning re-run after the universe-rebuild fix", "exploratory"),
    ("p1_ic_by_lookback", 10, "IC for raw and residual reversal at k in {1,2,3,5,10}", "adopted k=2"),
    ("p1_joint_k_rebalance", 60, "k x rebalance_every x no_trade_band joint search", "adopted"),
    ("p1_interior_optimum", 20, "Confirming rebalance=10 is interior, not a grid edge", "adopted"),
    ("p1_cov_estimator", 8, "ledoit_wolf / diagonal / pca / sample", "rejected: kept ledoit_wolf"),
    ("p1_cov_scope", 4, "Covariance over universe vs whole panel", "adopted universe"),
    ("p1_sleeve_survey", 7, "Seven price-based alpha sleeves measured standalone", "adopted 2"),
    ("p1_blend_combinations", 6, "Which sleeves to blend", "adopted vsr+overnight"),
    ("p1_retune_after_spread", 18, "Rebalance re-tuned once spreads were calibrated", "adopted"),
    ("p1_breadth_universe", 12, "Universe size 100/200/400/800 with realistic spreads", "rejected"),
    ("p1_liquidity_deciles", 10, "Reversal IC across ADV deciles on S&P 1500", "rejected"),
    ("p1_fundamental_sleeves", 8, "Six static value/quality sleeves + market cap", "rejected"),
    ("p1_fundamental_blend", 6, "Static fundamentals at several blend weights", "rejected"),
    ("p1_pead_weight", 7, "Earnings-surprise blend weight sweep", "adopted 10%"),
    ("p1_transfer_coefficient", 10, "Caps, neutrality and MVO variants to raise TC", "rejected"),
    ("p1_intraday_horizons", 5, "Hourly reversal IC at 1/2/3/6/13-hour lookbacks", "deferred"),
    ("p1_final_grid", 9, "Walk-forward grid: pead_weight x rebalance_every", "adopted"),
]


def seed_phase1(path: Path | str = DEFAULT_PATH) -> TrialLedger:
    """Initialise the ledger with the Phase-1 record if it is empty."""
    ledger = TrialLedger(path)
    if len(ledger):
        return ledger
    for key, n, description, outcome in PHASE1_TRIALS:
        ledger.entries[key] = TrialEntry(key, n, description, outcome)
    ledger.save()
    return ledger
