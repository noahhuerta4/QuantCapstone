from .metrics import (
    annualized_return,
    annualized_volatility,
    calmar,
    conditional_var,
    drawdown_duration,
    drawdown_series,
    equity_curve,
    hit_rate,
    max_drawdown,
    rolling_sharpe,
    sharpe,
    sortino,
    summarize,
    value_at_risk,
)

# NOTE: ``report`` and ``plots`` are deliberately not re-exported here. They
# depend on ``study``, which depends on ``validation``, which depends back on
# ``strategy`` -- and ``strategy`` imports ``reporting.metrics``. Importing them
# eagerly closes that loop. Import them by module path instead:
#     from quant_capstone.reporting.report import write_report

__all__ = [
    "sharpe", "sortino", "calmar", "summarize", "annualized_return",
    "annualized_volatility", "max_drawdown", "drawdown_series", "drawdown_duration",
    "equity_curve", "hit_rate", "value_at_risk", "conditional_var", "rolling_sharpe",
]
