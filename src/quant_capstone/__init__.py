"""Cross-sectional, market-neutral, short-term reversal equity strategy.

Build order mirrors the spec: data -> signal -> costs -> backtest -> risk
pipeline -> validation -> fundamental law -> reporting.
"""

from .config import DEFAULT, Config

__all__ = ["Config", "DEFAULT"]
__version__ = "0.1.0"
