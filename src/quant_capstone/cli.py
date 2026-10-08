"""Command line entry point: ``uv run quant-capstone <command>``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from .config import Config


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quant-capstone", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--source", choices=["yfinance", "synthetic"], default="yfinance")
        p.add_argument("--start", default=Config().data.start)
        p.add_argument("--end", default=Config().data.end)
        p.add_argument("--universe-size", type=int, default=Config().data.universe_size)
        p.add_argument("--k", type=int, default=Config().signal.k)
        p.add_argument("--vol-target", type=float, default=Config().risk.vol_target)
        p.add_argument("--timing", choices=["next_open", "same_close"], default="next_open")
        p.add_argument(
            "--rebalance-every", type=int, default=Config().execution.rebalance_every,
            help="trade every N days; the main lever on turnover and therefore on net Sharpe",
        )
        p.add_argument(
            "--no-trade-band", type=float, default=Config().execution.no_trade_band,
            help="skip a name unless its target weight moves by more than this fraction of NAV",
        )
        p.add_argument("--outdir", default="output")
        p.add_argument("-v", "--verbose", action="store_true")

    run = sub.add_parser("run", help="full study: backtest, law diagnostic, validation, report")
    add_common(run)
    run.add_argument("--quick", action="store_true", help="smaller grid and fewer bootstrap draws")
    run.add_argument("--no-plots", action="store_true")
    run.add_argument(
        "--prior-trials", type=int, default=None,
        help="configurations examined before this run, added to the Deflated Sharpe trial "
             "count. Defaults to the recorded total in output/trial_ledger.json; pass 0 to "
             "ignore the ledger.",
    )

    backtest = sub.add_parser("backtest", help="single backtest, print the gross/net table only")
    add_common(backtest)

    data = sub.add_parser("data", help="download and inspect the price panel")
    add_common(data)

    cross = sub.add_parser("crosscheck", help="re-run the baseline through vectorbt")
    add_common(cross)

    return parser


def _config_from_args(args: argparse.Namespace) -> Config:
    cfg = Config().override(
        data__source=args.source,
        data__start=args.start,
        data__end=args.end,
        data__universe_size=args.universe_size,
        signal__k=args.k,
        risk__vol_target=args.vol_target,
        execution__timing=args.timing,
    )
    cfg = cfg.override(
        execution__rebalance_every=args.rebalance_every,
        execution__no_trade_band=args.no_trade_band,
    )
    if args.source == "synthetic":
        # The synthetic panel is 60 names over ~6 years; scale the screens to it.
        cfg = cfg.override(
            data__universe_size=min(args.universe_size, 40),
            data__min_history=300,
            validation__train_days=500,
            validation__test_days=126,
        )
    if getattr(args, "quick", False):
        cfg = cfg.override(validation__bootstrap_samples=500)
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = _base_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)

    cfg = _config_from_args(args)
    outdir = Path(args.outdir)

    if args.command == "data":
        return _cmd_data(cfg)
    if args.command == "backtest":
        return _cmd_backtest(cfg)
    if args.command == "crosscheck":
        return _cmd_crosscheck(cfg)
    return _cmd_run(
        cfg, outdir,
        quick=args.quick, make_plots=not args.no_plots, prior_trials=args.prior_trials,
    )


def _cmd_data(cfg: Config) -> int:
    from .data import data_quality_report, load_dataset, universe_stats

    dataset = load_dataset(cfg)
    print("\n== data quality ==")
    print(data_quality_report(dataset.prices).to_string())
    print("\n== universe ==")
    print(universe_stats(dataset.members).describe().loc[["mean", "min", "max"]].to_string())
    return 0


def _cmd_backtest(cfg: Config) -> int:
    from .costs.model import attribution_bps
    from .strategy import activity_table, performance_table, run_strategy

    run = run_strategy(cfg)
    print("\n== performance (gross vs net) ==")
    print(performance_table(run.result).round(4).to_string())
    print("\n== activity ==")
    print(activity_table(run.result).round(4).to_string())
    print("\n== cost attribution (bps of NAV per year) ==")
    print(attribution_bps(run.result.costs, run.result.nav).round(2).to_string())
    print(f"\nrealized beta: {run.realized_beta():.4f}")
    return 0


def _cmd_crosscheck(cfg: Config) -> int:
    from .backtest.crosscheck import crosscheck
    from .strategy import run_strategy

    run = run_strategy(cfg)
    report = crosscheck(run)
    print("\n== engine cross-check ==")
    print(report.to_string())
    return 0


def _cmd_run(cfg: Config, outdir: Path, quick: bool, make_plots: bool,
             prior_trials: int | None = None) -> int:
    from .reporting.report import write_report
    from .study import run_study
    from .validation import DEFAULT_GRID, TrialLedger

    if prior_trials is None:
        # Read the count from the ledger rather than trusting anyone to remember
        # it. An under-stated trial count is exactly the error the Deflated
        # Sharpe exists to prevent.
        ledger = TrialLedger()
        prior_trials = ledger.total
        print(f"trial ledger: {len(ledger)} recorded sweeps, {prior_trials} configurations")

    grid = {"signal__k": [3, 5, 10]} if quick else DEFAULT_GRID
    study = run_study(cfg, outdir, grid=grid, make_plots=make_plots, prior_trials=prior_trials)
    path = write_report(study, outdir)

    print("\n" + "=" * 72)
    print(f"net OOS Sharpe: {study.oos_net_sharpe:.2f}   target {2.0:.1f}   "
          f"{'CLEARED' if study.clears_target else 'NOT CLEARED'}")
    print("=" * 72)
    print(study.law.verdict)
    print(f"\nreport: {path}")
    if study.plots:
        print(f"figures: {len(study.plots)} written to {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
