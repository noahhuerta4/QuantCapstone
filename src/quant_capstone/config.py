"""All tunable defaults in one place (spec Section 13).

Every config object is frozen. Sweeps and walk-forward folds build variants with
``cfg.override(signal__k=10, risk__vol_target=0.08)`` rather than mutating state,
so a config can never be silently changed underneath a completed backtest.

Where these defaults came from
------------------------------
Spec Section 13 gives starting defaults; four were changed after measuring, and
the original is recorded beside each so the comparison is not lost. None of this
was tuned toward the Sharpe *target* -- the changes follow from the measured IC
and from the turnover/holding-period interaction -- but they were still chosen by
looking at results, so every configuration examined is counted into the Deflated
Sharpe trial count via ``--prior-trials``.

===============  ========  ===========================================
parameter        S13 -> now  why
===============  ========  ===========================================
signal.k             5 -> 2  k=2 carries ~25% more IC than k=5 (0.0139 vs
                             0.0111; t = 2.9 vs 2.3) and retains enough
                             autocorrelation to survive a multi-day hold.
                             k=1 has the highest IC of all (0.0172, t = 3.7)
                             but ~zero autocorrelation, so a held k=1 signal
                             is stale within a day and nets less.
data.universe_size 200 -> 100 Halves cost (108 vs 238 bps/yr) for a small
                             breadth give-up: net 0.36 vs 0.33.
execution.
  rebalance_every    1 -> 10 The dominant lever. Daily rebalancing runs 264x
                             annual turnover costing 859 bps, more than the
                             gross edge is worth. Confirmed an interior
                             optimum rather than a grid edge -- net Sharpe
                             falls away on both sides (rebal 5 -> 0.20,
                             10 -> 0.36, 15 -> 0.23, 21 -> -0.17).
  no_trade_band  0.0015 -> 0.006  Re-tuned in Phase 2 once the sleeve set
                             changed. Confirmed an interior optimum: net Sharpe
                             runs 0.845 / 0.864 / 0.938 / 0.975 / 0.857 across
                             bands 0.000 / 0.003 / 0.006 / 0.010 / 0.013, and a
                             band above the 1.5% per-name cap stops trading
                             entirely. Default sits mid-plateau at 0.006 rather
                             than on the 0.010 peak; the walk-forward re-selects
                             it per fold.
signal.alphas   reversal -> blend  One sleeve caps out near 0.6 gross Sharpe.
                             Blending volume-shock reversal with overnight
                             reversal (correlated only 0.29) lifts gross to 0.91
                             and net to 0.74. See SignalConfig.alphas.
costs.
  use_estimated_   off -> on  The flat 2 bps was conservative, not realistic:
  spread                     the calibrated model puts this universe at 0.44 bps
                             median. Off is now the stress case.
===============  ========  ===========================================

The unchanged S13 defaults (vol target 10%, 1.5% per-name cap, 1% ADV liquidity
cap, IBKR tiered commissions, half-spread + impact slippage, t+1 open execution)
are all still in force.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class DataConfig:
    source: Literal["yfinance", "synthetic"] = "yfinance"
    start: str = "2015-01-01"
    end: str = "2025-01-01"
    cache_dir: str = "data_cache"
    universe_size: int = 100  # spec S13 says 200; see module docstring
    adv_lookback: int = 63
    """Trailing window (days) for the dollar-volume ranking that picks the universe."""
    universe_refresh: int = 21
    """Rebalance universe membership every N trading days (~monthly)."""
    min_price: float = 5.0
    min_history: int = 252
    """Bars of history a name needs before it is eligible."""


@dataclass(frozen=True)
class SignalConfig:
    alphas: tuple[str, ...] = ("volume_shock_reversal", "overnight_reversal")
    """Which alpha sleeves to blend (see ``signals/alphas.py``).

    Measured, not assumed. The seven implemented sleeves collapse into three
    families -- within each, variants correlate above 0.93 and are redundant:

        reversal <-> volume_shock_reversal      0.97
        momentum <-> residual_momentum          0.95
        low_volatility <-> idiosyncratic_vol    0.94

    Across families the correlations are ~0.00, which is the diversification
    worth having. Of the families:

    * **Reversal** is the workhorse. ``volume_shock_reversal`` beats plain
      reversal (gross 0.68 vs 0.59) for the same turnover, because scaling by
      volume tilts toward liquidity-driven moves that actually revert.
    * **Overnight reversal** correlates only 0.29 with the intraday-inclusive
      version, so it is a genuinely separate bet -- adding it lifts gross Sharpe
      from 0.68 to 0.91, the single largest gain in this project.
    * **Momentum** has the best IC of any sleeve (0.018, t = 3.1) and the worst
      Sharpe (0.12), because autocorrelation of 0.995 leaves under one
      independent bet per year. Equal-weighting it into the blend *lowers* net
      Sharpe (0.74 -> 0.52), so it is excluded.
    * **Volatility** sleeves are outright negative here (net -0.39, -0.48) in a
      book that is already beta-neutral and vol-targeted. Excluded; flipping the
      sign would be data mining, not a finding."""

    combine: Literal["equal", "fixed", "ic_weighted"] = "equal"
    """Equal weighting is the robust default; see ``signals/combine.py``."""
    pead_weight: float = 0.10
    """Weight given to ``earnings_surprise``; the rest is split evenly among
    ``alphas``. Setting this above zero *adds* the sleeve automatically, so it is
    a single knob and ``pead_weight = 0`` needs no fundamental data at all.

    A single scalar rather than a general weight vector so the walk-forward can
    *sweep* it: a blend weight chosen by looking at full-sample results is
    hindsight like any other parameter, and pinning it in the config would push
    that hindsight into every out-of-sample fold."""
    sleeve_weights: tuple[tuple[str, float], ...] = ()
    """Weights for ``combine="fixed"``, as (name, weight) pairs."""

    k: int = 2  # spec S13 says 5; see module docstring
    """Lookback for the trailing return that is reverted. Test k in {1,2,3,5,10}."""
    momentum_lookback: int = 252
    momentum_skip: int = 21
    """Skip the most recent month so momentum does not fight the reversal sleeve."""
    vol_lookback: int = 126
    overnight_k: int = 5
    volume_baseline: int = 63

    # Event-driven sleeves (Phase 2). Fixed in advance rather than swept: every
    # configuration examined is charged to the Deflated Sharpe trial count.
    insider_halflife: float = 20.0
    """Decay half-life in trading days for insider flow. A filing is a point
    event; without decay the sleeve becomes a step function that never moves,
    which is the static-ratio failure all over again."""
    announcement_window: int = 3
    """Days of price move treated as the reaction to an earnings report."""
    announcement_decay: int = 5
    """Reaction fades over a week -- overreaction unwinds fast, unlike the
    quarter-long drift the PEAD sleeve rides."""

    winsorize: bool = True
    winsor_mad: float = 3.0
    standardize: Literal["zscore", "rank"] = "zscore"
    orthogonalize_beta: bool = False
    """Regress the combined signal on beta cross-sectionally and keep the residual."""
    beta_lookback: int = 126


@dataclass(frozen=True)
class RiskConfig:
    cov_estimator: Literal["ledoit_wolf", "pca", "sample", "diagonal"] = "ledoit_wolf"
    cov_scope: Literal["universe", "panel"] = "universe"
    """Which names Sigma is estimated over.

    ``universe`` (correct): only the names tradable on the refresh day. Shrinkage
    is then calibrated to the T/N of the book we actually hold.

    ``panel``: every name in the downloaded price panel, then subset. This was an
    accidental behaviour before it was a setting, and it is kept only so the
    comparison is reproducible -- it is not point-in-time honest when the panel
    is assembled with hindsight, and its shrinkage depends on how many unrelated
    tickers happen to have been downloaded, which is not a modelling choice
    anyone would make on purpose."""
    cov_lookback: int = 252
    cov_refresh: int = 21
    """Recompute Sigma every N days; it is far too slow to redo daily and barely moves."""
    pca_factors: int = 10
    weight_mode: Literal["risk_weighted", "mvo"] = "risk_weighted"
    mvo_lambda: float = 10.0
    neutralize_dollar: bool = True
    neutralize_beta: bool = True
    neutralize_sector: bool = True
    vol_target: float = 0.10
    max_weight: float = 0.015
    max_gross: float = 3.0
    max_adv_participation: float = 0.01
    regime_scaling: Literal["off", "linear", "binary"] = "off"
    """Scale finished weights by a trailing cross-sectional dispersion regime.

    Off by default: it is a Phase-2 overlay and the unconditional book stays the
    reference. Applied to the finished weights rather than to the signal, because
    vol-targeting would scale a signal-level change straight back out."""
    regime_floor: float = 0.4
    """Smallest multiplier. The overlay only ever scales down -- scaling up would
    hit the per-name cap that already binds."""
    regime_lookback: int = 504

    constraint_iters: int = 8
    """Clipping breaks neutrality, re-projecting breaks the caps; iterate to a fixed point."""


@dataclass(frozen=True)
class CostConfig:
    # Each component is individually switchable so Section 10 can sweep them.
    enable_commission: bool = True
    enable_reg_fees: bool = True
    enable_borrow: bool = True
    enable_slippage: bool = True

    # IBKR US equities. VERIFY AT BUILD TIME -- these schedules drift.
    commission_model: Literal["tiered", "fixed"] = "tiered"
    per_share_tiered: float = 0.0035
    per_share_fixed: float = 0.005
    min_per_order_tiered: float = 0.35
    min_per_order_fixed: float = 1.00
    max_pct_trade_value: float = 0.01

    # Regulatory, charged on sales only.
    sec_fee_rate: float = 0.0000278  # $27.80 per $1M of sales
    finra_taf_per_share: float = 0.000166
    finra_taf_cap: float = 8.30

    borrow_rate: float = 0.005
    """Annualized general-collateral borrow on the short leg."""

    enable_financing: bool = True
    financing_rate: float = 0.03
    """Annualized margin rate on the portion of the long leg funded with debt.

    Charged on ``max(long exposure - NAV, 0)``, so it is **exactly zero at the
    book's current ~0.5x long exposure** and every unlevered result in this
    project is unchanged by its introduction. It exists because leverage changes
    the arithmetic: at 3x gross the long leg is 1.5x NAV and half of NAV is
    borrowed; at 5x gross, one-and-a-half times NAV is.

    A single rate across 2015-2024 is crude -- benchmark rates went from near
    zero to above 5% inside the sample -- so 3% is a mid-period compromise and
    the number to stress rather than trust."""

    slippage_model: Literal["half_spread", "impact"] = "impact"
    half_spread_bps: float = 2.0
    use_estimated_spread: bool = True
    """Charge each name its own half-spread rather than a flat rate.

    ON by default, because the flat 2 bps turned out to be the *conservative*
    assumption, not the realistic one: the calibrated model puts the top-100-by-
    ADV universe at a median half-spread of 0.44 bps, roughly 4.5x cheaper. A
    flat rate is also wrong in the other direction for a wide universe, where
    small-caps genuinely cost tens of bps.

    Setting this to False recovers the flat ``half_spread_bps`` and is the natural
    stress case -- it is ~4.5x the calibrated cost for this universe."""
    spread_window: int = 21
    spread_estimator: Literal["liquidity_scaled", "abdi_ranaldo", "corwin_schultz"] = (
        "liquidity_scaled"
    )
    """How the per-name half-spread is set when ``use_estimated_spread`` is on.

    ``liquidity_scaled`` is a calibrated function of dollar volume. The two
    bar-based estimators are available but do NOT work for liquid US equities --
    they return 0 bps and 20 bps for names whose true half-spread is ~0.4 bps.
    See ``costs/spread.py``."""
    impact_coef: float = 0.1
    """c in impact = c * sigma * sqrt(shares / ADV). See costs/slippage.py for calibration."""

    cost_multiplier: float = 1.0
    """Global scale on every cost. The break-even value is a Section 10 deliverable."""


@dataclass(frozen=True)
class ExecConfig:
    timing: Literal["next_open", "same_close"] = "next_open"
    rebalance_every: int = 10  # spec S13 says daily; see module docstring
    """Trade every N days. The single biggest determinant of net Sharpe here."""
    no_trade_band: float = 0.006  # S13 says 0.0015; Phase 2 re-tuned, see module docstring
    """Skip a name unless |target - current| exceeds this fraction of NAV."""
    min_trade_dollars: float = 200.0
    """Skip trades too small to clear the per-order commission minimum."""
    initial_nav: float = 10_000_000.0


@dataclass(frozen=True)
class ValidationConfig:
    train_days: int = 756
    test_days: int = 126
    """~6-month OOS windows, rolled forward."""
    expanding: bool = True
    embargo_days: int = 10
    """Days purged from the end of each training window so a position opened in
    training cannot still be open in the test window. Raised to the rebalance
    interval automatically if that is longer."""
    bootstrap_samples: int = 2000
    bootstrap_block: int = 21
    pbo_splits: int = 14
    seed: int = 0


@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    execution: ExecConfig = field(default_factory=ExecConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    def override(self, **kwargs: Any) -> "Config":
        """Return a copy with ``section__field=value`` overrides applied.

        >>> cfg.override(signal__k=10, costs__cost_multiplier=2.0)
        """
        sections: dict[str, dict[str, Any]] = {}
        top: dict[str, Any] = {}
        for key, value in kwargs.items():
            section, sep, attr = key.partition("__")
            if not sep:
                top[section] = value
                continue
            if not hasattr(self, section):
                raise KeyError(f"unknown config section {section!r}")
            sections.setdefault(section, {})[attr] = value

        for section, updates in sections.items():
            current = getattr(self, section)
            for attr in updates:
                if not hasattr(current, attr):
                    raise KeyError(f"unknown field {section}.{attr}")
            top[section] = dataclasses.replace(current, **updates)
        return dataclasses.replace(self, **top)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


DEFAULT = Config()

TRADING_DAYS = 252
