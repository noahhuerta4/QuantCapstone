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
             "count. Defaults to the ledger total in results/trial_ledger.json (less the "
             "walk-forward grid, which the run counts itself); pass 0 to ignore the ledger.",
    )

    backtest = sub.add_parser("backtest", help="single backtest, print the gross/net table only")
    add_common(backtest)

    data = sub.add_parser("data", help="download and inspect the price panel")
    add_common(data)
    data.add_argument("--write-manifest", action="store_true",
                      help="record the cached inputs' hashes in results/data_manifest.json")

    study = sub.add_parser("study", help="regenerate the side studies behind the README")
    study.add_argument("names", nargs="*", default=["all"],
                       help="study names, or 'all' (default); see quant_capstone/studies")
    study.add_argument("--outdir", default="results/studies")
    study.add_argument("--check", action="store_true",
                       help="regenerate into a scratch directory and diff against --outdir")
    study.add_argument("-v", "--verbose", action="store_true")

    book = sub.add_parser("book", help="assemble Report/Market-Neutral-Reversal-Book.html")
    book.add_argument("-v", "--verbose", action="store_true")

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

    if args.command == "study":
        return _cmd_study(args.names, Path(args.outdir), args.check)
    if args.command == "book":
        from .reporting.book import build

        print(f"write-up: {build()}")
        return 0

    cfg = _config_from_args(args)
    outdir = Path(args.outdir)

    if args.command == "data":
        return _cmd_data(cfg, args.write_manifest)
    if args.command == "backtest":
        return _cmd_backtest(cfg)
    if args.command == "crosscheck":
        return _cmd_crosscheck(cfg)
    return _cmd_run(
        cfg, outdir,
        quick=args.quick, make_plots=not args.no_plots, prior_trials=args.prior_trials,
    )


def _cmd_data(cfg: Config, write_manifest: bool = False) -> int:
    from .data import data_quality_report, load_dataset, manifest, universe_stats

    dataset = load_dataset(cfg)
    if write_manifest:
        print(f"manifest: {manifest.write()}")
    print("\n== data quality ==")
    print(data_quality_report(dataset.prices).to_string())
    print("\n== universe ==")
    print(universe_stats(dataset.members).describe().loc[["mean", "min", "max"]].to_string())
    return 0


def _verify_inputs() -> bool:
    """Warn, loudly, when the cached data is not what the committed results used."""
    from .data import manifest

    if not manifest.MANIFEST.exists():
        print("no results/data_manifest.json -- cannot confirm these are the recorded inputs")
        return False
    report = manifest.verify()
    bad = report[report["status"] != "ok"]
    if bad.empty:
        print(f"inputs: all {len(report)} cached files match results/data_manifest.json")
        return True
    print("WARNING: cached inputs differ from the ones the committed results used.\n"
          "Numbers may move for reasons no code change explains.")
    print(bad.to_string(index=False))
    return False


def _cmd_study(names: list[str], outdir: Path, check: bool) -> int:
    import tempfile

    from .studies import ORDER, compare_dirs, run

    names = list(ORDER) if names == ["all"] else names
    _verify_inputs()
    if not check:
        for path in run(names, outdir):
            print(path)
        return 0

    # Figures are images, not tables; --check compares the tables.
    names = [n for n in names if n != "figures"]
    with tempfile.TemporaryDirectory() as scratch:
        run(names, Path(scratch))
        report = compare_dirs(Path(scratch), outdir)
    print(report.to_string(index=False))
    return 0 if (report["status"] == "match").all() else 1


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


QUICK_GRID = {"signal__k": [1, 2, 5]}
"""A smoke-test grid. It must contain the default k, or the walk-forward has no
cell matching the defaults it is compared against."""


def _cmd_run(cfg: Config, outdir: Path, quick: bool, make_plots: bool,
             prior_trials: int | None = None) -> int:
    from .reporting.headline import write_headline
    from .reporting.report import write_report
    from .study import run_study
    from .validation import DEFAULT_GRID, WALKFORWARD_KEY, TrialLedger

    grid = QUICK_GRID if quick else DEFAULT_GRID
    ledger = TrialLedger()
    # Only a full real-data run is the headline and gets written to the ledger.
    # Synthetic data is not a look at the market, and --quick is a smoke test.
    records = prior_trials is None and not quick and cfg.data.source == "yfinance"
    if prior_trials is None:
        # Read the count from the ledger rather than trusting anyone to remember
        # it. The walk-forward grid is left out because this run counts it itself
        # -- including it here as well was double-counting every cell.
        prior_trials = ledger.total_excluding(WALKFORWARD_KEY)
        print(f"trial ledger: {len(ledger)} recorded sweeps, "
              f"{prior_trials} configurations before this grid")

    study = run_study(cfg, outdir, grid=grid, make_plots=make_plots, prior_trials=prior_trials)
    path = write_report(study, outdir)
    if records:
        axes = " x ".join(f"{k.split('__')[-1]} {list(v)}" for k, v in grid.items())
        ledger.record(WALKFORWARD_KEY, study.grid_trials,
                      f"Headline walk-forward grid: {axes}", "headline")
        print(f"headline numbers: {write_headline(study)}")

    print("\n" + "=" * 72)
    print(f"net OOS Sharpe: {study.oos_net_sharpe:.2f}   target {2.0:.1f}   "
          f"{'CLEARED' if study.clears_target else 'NOT CLEARED'}")
    print(f"deflated against {study.n_trials} configurations "
          f"({prior_trials} earlier + {study.grid_trials} in this grid)")
    print("=" * 72)
    print(study.law.verdict)
    print(f"\nreport: {path}")
    if study.plots:
        print(f"figures: {len(study.plots)} written to {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
