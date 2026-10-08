"""Cost accounting (spec S0: a priority test target).

Cost bugs are the ones that inflate a Sharpe without ever looking wrong: a fee
charged on the wrong side, a per-order minimum silently skipped, a toggle that
does not actually disable anything. Each of those gets a test.
"""

import numpy as np
import pandas as pd
import pytest

from quant_capstone.config import CostConfig
from quant_capstone.costs import (
    BorrowModel,
    CommissionModel,
    CostModel,
    RegulatoryFees,
    SlippageModel,
    attribution_bps,
)


# ------------------------------------------------------------------ commissions
def test_per_share_rate_applies_above_the_minimum():
    model = CommissionModel(per_share=0.0035, min_per_order=0.35, max_pct_trade_value=0.01)
    # 1,000 shares at $50 -> $3.50, comfortably above both the floor and the cap.
    assert model.charge(np.array([1000.0]), np.array([50_000.0]))[0] == pytest.approx(3.50)


def test_per_order_minimum_binds_on_small_orders():
    model = CommissionModel(0.0035, 0.35, 0.01)
    # 10 shares would be $0.035; the $0.35 minimum is 10x that.
    assert model.charge(np.array([10.0]), np.array([2_000.0]))[0] == pytest.approx(0.35)


def test_one_percent_cap_overrides_the_minimum():
    model = CommissionModel(0.0035, 0.35, 0.01)
    # A $10 trade caps at $0.10, below the $0.35 floor -- the cap wins.
    assert model.charge(np.array([3.0]), np.array([10.0]))[0] == pytest.approx(0.10)


def test_untraded_names_pay_nothing():
    model = CommissionModel(0.0035, 0.35, 0.01)
    assert model.charge(np.array([0.0, 100.0]), np.array([0.0, 5_000.0]))[0] == 0.0


def test_minimum_makes_small_orders_expensive_in_bps():
    """The economic point of the minimum: it penalises fragmentation."""
    model = CommissionModel(0.0035, 0.35, 0.01)
    small = model.effective_bps(np.array([10.0]), np.array([2_000.0]))[0]
    large = model.effective_bps(np.array([10_000.0]), np.array([2_000_000.0]))[0]
    assert small > 5 * large


def test_fixed_and_tiered_schedules_differ():
    cfg = CostConfig()
    tiered = CommissionModel.from_config(cfg)
    fixed = CommissionModel.from_config(CostConfig(commission_model="fixed"))
    assert fixed.per_share > tiered.per_share
    assert fixed.min_per_order > tiered.min_per_order


def test_unknown_commission_model_is_rejected():
    with pytest.raises(ValueError):
        CommissionModel.from_config(CostConfig(commission_model="nonsense"))


# ---------------------------------------------------------------- regulatory
def test_regulatory_fees_are_charged_on_sales_only():
    fees = RegulatoryFees(sec_fee_rate=0.0000278, finra_taf_per_share=0.000166, finra_taf_cap=8.30)
    buy = fees.charge(np.array([100.0]), np.array([10_000.0]))[0]
    sell = fees.charge(np.array([-100.0]), np.array([-10_000.0]))[0]
    assert buy == 0.0
    assert sell > 0.0


def test_sec_fee_scales_with_sale_value():
    fees = RegulatoryFees(0.0000278, 0.0, 8.30)
    assert fees.charge(np.array([-1.0]), np.array([-1_000_000.0]))[0] == pytest.approx(27.80)


def test_finra_taf_is_capped_per_trade():
    fees = RegulatoryFees(0.0, 0.000166, 8.30)
    assert fees.charge(np.array([-10_000_000.0]), np.array([-1e9]))[0] == pytest.approx(8.30)


# ------------------------------------------------------------------- borrow
def test_borrow_is_charged_on_shorts_only():
    model = BorrowModel(annual_rate=0.005)
    charged = model.charge_per_name(np.array([1_000_000.0, -1_000_000.0]))
    assert charged[0] == 0.0
    assert charged[1] == pytest.approx(0.005 / 252 * 1_000_000)


def test_borrow_accrues_to_the_annual_rate_over_a_trading_year():
    model = BorrowModel(0.005)
    assert model.charge(1_000_000.0) * 252 == pytest.approx(5_000.0)


# ----------------------------------------------------------------- slippage
def test_half_spread_is_linear_in_notional_and_ignores_size():
    model = SlippageModel(half_spread_bps=2.0, impact_coef=0.1, use_impact=False)
    rate = model.rate(np.array([1.0, 1e9]), np.array([1e6, 1e6]), np.array([0.02, 0.02]))
    assert rate == pytest.approx([2e-4, 2e-4])


def test_impact_grows_with_the_square_root_of_participation():
    model = SlippageModel(half_spread_bps=0.0, impact_coef=1.0, use_impact=True)
    adv, sigma = np.array([1e6, 1e6]), np.array([0.02, 0.02])
    rate = model.rate(np.array([10_000.0, 40_000.0]), adv, sigma)
    # Quadrupling the order doubles the impact.
    assert rate[1] / rate[0] == pytest.approx(2.0)


def test_impact_scales_with_volatility():
    model = SlippageModel(0.0, 1.0, use_impact=True)
    rate = model.rate(np.array([1e4, 1e4]), np.array([1e6, 1e6]), np.array([0.01, 0.02]))
    assert rate[1] / rate[0] == pytest.approx(2.0)


def test_slippage_survives_a_missing_adv():
    model = SlippageModel(2.0, 0.1, use_impact=True)
    rate = model.rate(np.array([1000.0]), np.array([0.0]), np.array([np.nan]))
    assert np.isfinite(rate).all()
    assert rate[0] == pytest.approx(2e-4)


# ---------------------------------------------------------------- aggregate
def _charge(cfg: CostConfig):
    return CostModel(cfg).charge(
        trade_shares=np.array([1000.0, -1000.0]),
        trade_notional=np.array([50_000.0, -50_000.0]),
        position_notional=np.array([50_000.0, -50_000.0]),
        adv_shares=np.array([1e6, 1e6]),
        sigma=np.array([0.02, 0.02]),
    )


def test_every_component_contributes_by_default():
    costs = _charge(CostConfig())
    assert costs.commission > 0 and costs.reg_fees > 0
    assert costs.slippage > 0 and costs.borrow > 0
    assert costs.total == pytest.approx(
        costs.commission + costs.reg_fees + costs.slippage + costs.borrow
    )


@pytest.mark.parametrize("component", ["commission", "reg_fees", "slippage", "borrow"])
def test_each_toggle_switches_off_exactly_one_component(component):
    baseline = _charge(CostConfig())
    off = _charge(CostConfig(**{f"enable_{component}": False}))

    assert getattr(off, component) == 0.0
    assert off.total < baseline.total
    for other in ("commission", "reg_fees", "slippage", "borrow"):
        if other != component:
            assert getattr(off, other) == pytest.approx(getattr(baseline, other))


def test_cost_multiplier_scales_everything_proportionally():
    baseline = _charge(CostConfig())
    doubled = _charge(CostConfig(cost_multiplier=2.0))
    assert doubled.total == pytest.approx(2.0 * baseline.total)
    assert _charge(CostConfig(cost_multiplier=0.0)).total == 0.0


def test_slippage_dominates_commission_on_a_liquid_trade():
    """The spec's central claim about where the Sharpe goes."""
    costs = _charge(CostConfig())
    assert costs.slippage > costs.commission


def test_attribution_converts_to_annual_bps():
    index = pd.date_range("2020-01-01", periods=252, freq="B")
    costs = pd.DataFrame({"commission": 100.0, "slippage": 300.0}, index=index)
    nav = pd.Series(1_000_000.0, index=index)

    annual = attribution_bps(costs, nav)
    # $100/day on $1m NAV for 252 days = 2.52% = 252 bps.
    assert annual["commission"] == pytest.approx(252.0)
    assert annual["slippage"] == pytest.approx(756.0)
    assert annual["total"] == pytest.approx(1008.0)
