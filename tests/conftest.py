import pytest

from quant_capstone.config import Config
from quant_capstone.data import load_dataset
from quant_capstone.signals import build_signal


@pytest.fixture(scope="session")
def cfg() -> Config:
    """Synthetic-data config sized so the whole suite runs in seconds."""
    return Config().override(
        data__source="synthetic",
        data__universe_size=30,
        data__min_history=300,
        validation__train_days=500,
        validation__test_days=126,
        validation__bootstrap_samples=200,
        # The synthetic panel carries no SEC filings, so the PEAD sleeve has no
        # data to build from. Price sleeves only here; the fundamental path is
        # covered directly in tests/test_fundamentals.py.
        signal__pead_weight=0.0,
    )


@pytest.fixture(scope="session")
def dataset(cfg):
    return load_dataset(cfg)


@pytest.fixture(scope="session")
def signal(dataset, cfg):
    return build_signal(dataset, cfg)


@pytest.fixture(scope="session")
def run(dataset, cfg):
    from quant_capstone.strategy import run_strategy

    return run_strategy(cfg, dataset)
