"""The configurations each side study was originally run under.

The side studies were run at different points in the project, against whatever
``Config()`` defaulted to at the time -- and those defaults moved as results came
in. A study that calls ``Config()`` today would silently run against the *final*
defaults and stop reproducing the numbers the README quotes. So every study
names its era explicitly, and the era pins every default that has changed since.

Defaults that changed, and when (reconstructed from the project record):

====================  =========  =========  ==========  ==========
era                   alphas     spread     pead_weight  no_trade_band
====================  =========  =========  ==========  ==========
FLAT_SPREAD           reversal   flat 2bps  --          0.003
BLEND                 vsr+ovn    estimated  0.0         0.003
PEAD                  vsr+ovn    estimated  0.10        0.003
FINAL                 vsr+ovn    estimated  0.10        0.006
====================  =========  =========  ==========  ==========

``k = 2``, ``rebalance_every = 10`` and ``universe_size = 100`` hold in every era
here. The one study that predates even those (``turnover``) pins them itself.

Margin financing on the long leg entered the cost model only at the close of
Phase 2, so every earlier era switches it off. It is exactly zero for an
unlevered book, but the studies that loosen caps or widen the universe push the
long leg past NAV, and there it would charge a cost those results never paid.
"""

from __future__ import annotations

from ..config import Config

FLAT_SPREAD = Config().override(
    signal__alphas=("reversal",),
    signal__pead_weight=0.0,
    costs__use_estimated_spread=False,
    execution__no_trade_band=0.003,
    costs__enable_financing=False,
)
"""Phase 1, single reversal sleeve, flat 2 bps half-spread (to 2026-09-13 07:55)."""

BLEND = Config().override(signal__pead_weight=0.0, execution__no_trade_band=0.003,
                          costs__enable_financing=False)
"""Phase 1, volume-shock + overnight blend on calibrated spreads (to 2026-09-14 02:14)."""

PEAD = Config().override(execution__no_trade_band=0.003, costs__enable_financing=False)
"""Phase 1 close and Phase 2 directions A-C: blend + 10% earnings surprise (to 2026-09-15 00:40)."""

FINAL = Config()
"""Phase 2 close: the no-trade band re-tuned to 0.006. The current defaults."""
