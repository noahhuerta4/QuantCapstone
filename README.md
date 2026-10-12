# Quant Capstone — Short-Term Cross-Sectional Reversal

A dollar-neutral, beta-neutral, sector-neutral short-term reversal strategy on liquid
US equities, backtested with a full IBKR cost model and a validation harness built to
catch the ways this kind of backtest normally lies.

The deliverable is **not** a high Sharpe. It is an honest, net-of-cost, out-of-sample
Sharpe with a confidence interval, plus a diagnosis — via the Fundamental Law — of
which term is actually binding: forecasting skill (IC) or costs and constraints (TC).

## Results

S&P 500, 2015–2024. Headline is the **walk-forward out-of-sample** number: every parameter
that was ever chosen by looking at results — the reversal lookback `k`, the PEAD blend
weight, the rebalance interval and the no-trade band — is re-selected on each fold's
training window from a 135-cell grid, with the holding period embargoed off the end of
training. Nothing is pinned with hindsight.

### Net out-of-sample Sharpe: 0.91

Each column adds one sleeve family and runs through the identical harness — same grid
axes, same 13 folds, same embargo, same trial ledger — so the increments are comparable
(`quant-capstone study progression`).

| | reversal only | blend | **blend + PEAD** |
|---|---|---|---|
| **OOS net Sharpe** | 0.25 | 0.63 | **0.91** |
| OOS net, fixed defaults | 0.41 | 0.75 | 0.95 |
| Full-sample net Sharpe | 0.44 | 0.77 | 0.94 |
| Full-sample gross Sharpe | 0.60 | 0.94 | 1.11 |
| Deflated Sharpe | 0.28 | 0.41 | **0.67** (443 trials) |
| PBO | 0.86 | 0.37 | **0.24** |
| Bootstrap 95% CI | [-0.51, 1.02] | [-0.03, 1.33] | **[0.23, 1.60]** |
| P(true Sharpe ≤ 0) | 25% | 3% | **0.4%** |
| Folds positive (of 13) | 46% | 69% | **85%** |
| Train − test Sharpe | 0.58 | 0.31 | **0.17** |
| Break-even cost multiplier | 3.7x | 5.4x | **6.4x** |

"Reversal only" is the plain `k`-day reversal sleeve; "blend" adds volume-shock and
overnight reversal in its place. The deflated Sharpe is computed against every
configuration the project has examined (443 configurations, from
`results/trial_ledger.json`), not just this grid. The numbers recorded at the time each
sleeve was introduced (OOS 0.05 → 0.80 → 0.84, then 0.90 after Phase 2) came from
different grids and an earlier, flawed trial count, so they are history rather than
results — see [Validation fixes](#validation-fixes).

**Per-fold selection does not beat the fixed defaults.** Through the same windows with no
re-selection, the defaults score 0.95 against 0.91, and selection wins only 3 of 13 folds.
That is not a reason to prefer the defaults: they were themselves chosen by looking at
full-sample results — the band was set at 0.006 just below the
full-sample peak at 0.010 — so the default column carries exactly the hindsight the
walk-forward removes. It is an optimistic reference, not a rival estimate. What it does
say is that choosing among 135 cells per fold adds nothing measurable out of sample; at the
margin the search fits noise, and the 0.17 train-minus-test gap is its price. The headline
is the selected number because it is the only one that could actually have been traded.
`results.md` states this automatically whenever it happens.

What the folds chose is stable, which is the reassuring half: `k = 2` in 11 of 13 folds
(`k = 1` in the other two), a 10-day rebalance and a 10% PEAD weight in all 13, and the
0.010 band in 11 (the 0.006 default in the other two). The per-fold table is in
`output/results.md`.

### Three sleeves, three mechanisms

| sleeve | IC | t | autocorr | turnover | standalone net |
|---|---|---|---|---|---|
| volume-shock reversal | 0.0136 | 2.90 | 0.47 | 26.1x | 0.55 |
| overnight reversal | 0.0055 | 1.27 | 0.78 | 25.9x | 0.34 |
| earnings surprise (PEAD) | 0.0113 | 2.89 | 0.99 | 6.3x | 0.29 |

Standalone backtests use the final configuration (`quant-capstone study sleeve_survey`).
Cross-correlations are 0.30 between the two reversal sleeves and **−0.01** between either
and PEAD — three genuinely different bets. PEAD carries 10% of the blend, which is where a
fine sweep shows a plateau: 0 → 0.778, 5% → 0.788, 10% → 0.864, 15% → 0.861, then a cliff to
0.646 at 20% (`study pead`). The walk-forward picks 10% in every fold.

### What the cost number means

63.5 bps/yr is **0.64% of fund capital**, or about $64k/yr on the $10m book.
Equivalently **2.50 bps per dollar traded**, at 25.4x annual turnover. Against a 4.1% gross
return it consumes about a sixth, leaving 3.4% net at 3.7% volatility. The break-even
multiplier of 6.4x means costs would have to reach ~408 bps/yr before the edge dies.

## Validation fixes

A review of the finished Phase 2 project found three ways the headline was weaker than
the README claimed. All three are fixed, and the result held.

1. **`k` was pinned, not selected.** The README said the reversal lookback was re-chosen
   per fold; the walk-forward grid did not contain it. `k = 2` came from a full-sample IC
   sweep and every fold inherited it. It is now in the grid at the spec's pre-registered
   values {1, 2, 3, 5, 10} — not a range centred on the number already picked — and the
   folds choose `k = 2` on their own in 11 of 13. Net OOS moved 0.90 → 0.91, together
   with the embargo fix below.
2. **The fixed defaults beat selection, unremarked.** The comparison column was always
   printed and never discussed. It now is, above and in every generated `results.md`.
3. **The trial count was double-counted.** The quoted 327 was the ledger's 299 *plus* the
   27-cell headline grid that was already in it, *plus* one more for the defaults, which
   were run twice because they are also a grid cell. The run cache now counts distinct
   configurations, and the ledger marks the two earlier grids that the new one contains
   as superseded rather than deleting them. The count is now 443:
   308 earlier configurations, including the 45-cell reversal-only ablation above,
   plus the 135-cell grid.

Two smaller fixes came with them: the embargo now covers the longest holding period
anywhere in the grid (15 days) instead of the default's (10), and the trial ledger moved
from the git-ignored `output/` into version-controlled `results/`. `tests/test_docs.py`
now fails if this README or the write-up quotes a headline number that disagrees with
`results/headline.json`.

## Phase 2: three directions, one worked

Net OOS Sharpe moved **0.84 → 0.90** under the harness of the time (0.91 after the
[validation fixes](#validation-fixes)). The gain did not come from where it was
expected, and two of the three directions produced clean negative results.

| | direction | intended lever | outcome |
|---|---|---|---|
| **A** | Event-driven sleeves | breadth | **rejected** — all six candidates |
| **B** | Passive-execution intraday | cost inversion | **rejected** at the feasibility gate |
| **C** | Regime conditioning | forecasting skill | **rejected**, but exposed the real lever |

### A — insider and earnings sleeves (rejected)

Six pre-registered candidates built on two new data sources: SEC Form 4 bulk insider
filings (217,517 open-market trades, 489 tickers, median filing lag of 2 days exactly as
the statute requires) and further cuts of the XBRL panel.

Every one failed. The insider sleeves have *significantly negative* IC — net insider
buying predicts −0.0133 at t = −3.50 — which survived two checks for artefacts: only 1.7%
of scored days are zero-filled, and restricting to names that actually filed within 60
days gives −0.0145 at t = −3.66. **The sign was not flipped.** The literature's prior is
positive, there is no mechanism for a negative sign, and reversing it because this sample
says so is precisely the curve-fit the whole project is built to detect.

Revenue surprise had the best IC of any sleeve ever measured here (0.0175, t = 3.05) and
was still rejected: it correlates 0.46 with the existing earnings-surprise sleeve, so it
is a redundant variant rather than a new bet, and swapping or adding it both made things
worse (0.806 and 0.791 against 0.864).

### B — passive execution (rejected at the gate, in hours not weeks)

The thesis was cost *inversion*: a resting limit order earns the half-spread instead of
paying it, so the dominant turnover-linked cost changes sign and 20x breadth becomes
affordable. A parametric (fill ratio × adverse selection) map killed it before any
microstructure work, on one line of arithmetic:

```
commission + fees per $ traded : 0.700 bps   (unavoidable, passive or aggressive)
max spread CAPTURE at 100% fill: 0.440 bps
net cost per $ traded          : 0.260 bps   → turnover never becomes free
```

Because Phase 1 established the calibrated half-spread is only 0.44 bps, the spread is no
longer the dominant per-dollar cost — commission is. Earning the spread perfectly cannot
flip the sign of a cost that commission dominates. The thesis was built on a cost
structure the project had already disproved.

### C — regime conditioning (rejected, but it found the lever)

Blend IC across dispersion quartiles is monotone (−0.0028, +0.0078, +0.0181, +0.0242) and
the mechanism fits: reversal is paid for providing liquidity, and both reversal sleeves
strengthen with dispersion while the earnings sleeve shows no pattern. Market volatility
was tested the same way, came out jagged, and was discarded as noise.

The overlay improved the number — and the control showed it was for the wrong reason. A
**constant** multiplier at the same average size beat the regime-varying one (0.907 vs
0.878), and the diagnostic found no alpha timing at all (return gap −0.000004). The entire
gain was reduced turnover.

Chasing that directly found the real lever: the **no-trade band**, re-tuned now that the
sleeve set has changed. It is a genuine interior optimum, and the cliff is mechanically
explained — a band wider than the 1.5% per-name cap means no trade can ever clear it:

| band | 0.000 | 0.003 | 0.006 | 0.010 | 0.013 | 0.016+ |
|---|---|---|---|---|---|---|
| net Sharpe | 0.845 | 0.864 | 0.938 | **0.975** | 0.857 | *nothing trades* |

### Where it lands

Phase 2 took net OOS from 0.84 to 0.90 under the harness of the time; the strategy as it
stands after the validation fixes is the headline table above. Two methodological
upgrades came with it. The walk-forward now **embargoes** the holding period off the end
of each training window, so a position opened in training cannot still be open in the
test window. And the trial count lives in `results/trial_ledger.json` rather than in
anyone's memory — every sweep with what it tried and what came of it, rejected candidates
included, read automatically by the CLI when deflating.

_Reproduce: `quant-capstone study events passive regime band`._

## The fundamentals route

Price and volume give two sleeves correlated at 0.30 — two views of one microstructure
effect. Genuinely *independent* families were the main remaining route to more IR, so the
project carries a real point-in-time fundamentals layer.

**Data.** IB's fundamentals need a paid Reuters subscription (error 10358); yfinance
returns only 5–7 quarters. The SEC's XBRL API is free, official, reaches back to ~2009,
and stamps every fact with its **filing date** — so `data/sec_fundamentals.py` makes a
value visible the day it was actually filed, rather than applying the usual blanket
six-month lag. Restatements fall out correctly: refiling 2019 numbers in 2021 creates a
later-filed fact, so a 2019 query still returns what the market had in 2019. 138,678
facts across 503 tickers.

**Static ratios do not work, for a breadth reason.** Book-to-price, earnings yield, gross
profitability, ROE, asset growth and accruals are all near-perfectly independent of the
price sleeves (correlations 0.00–0.03) — exactly the diversification wanted. And none of
them helps at any weight. Every one has autocorrelation between **0.997 and 0.9995**:
filings arrive four times a year and the ratio then sits still, so `IR = IC·√BR` caps the
contribution near zero regardless of IC. Earnings yield has genuinely significant IC
(0.0057, t = 1.95) and a negative standalone Sharpe. **Independence without breadth buys
nothing.**

**PEAD does work, because it has an event clock.** An earnings *surprise* is not a level —
it arrives, decays over about a quarter, and fiscal calendars are staggered, so the
cross-section keeps turning over. Standardized unexpected earnings (Bernard & Thomas 1989,
seasonal random-walk form, faded linearly over 60 days) gives IC 0.0113 at t = 2.89 —
the best of any fundamental sleeve and close to the best price sleeve — with a positive
standalone Sharpe and only 6.3x turnover. In the ablation above it lifts OOS net Sharpe
from 0.63 to 0.91.

That contrast is the cleanest illustration in the project of what the Fundamental Law is
for: two signals built from the *same filings*, one useless and one useful, and the
difference is entirely breadth.

_Reproduce: `quant-capstone study fundamentals pead`._

## The intraday route

Breadth is the binding constraint everywhere else in this project, and hourly bars are
where breadth actually lives. Tested on 309,573 hourly bars (59 names, 3 years) pulled
from Interactive Brokers via `ibapi` — IB refuses fundamentals without a Reuters
subscription but serves historical bars fine.

**Hourly reversal has almost no autocorrelation, which is the whole point:**

| horizon | IC | t | autocorrelation |
|---|---|---|---|
| **1 hour** | **0.0095** | **2.93** | **−0.001** |
| 2 hours | 0.0065 | 1.99 | 0.45 |
| 3 hours | 0.0077 | 2.34 | 0.61 |
| 6 hours | 0.0052 | 1.53 | 0.80 |

At an autocorrelation of −0.001 every bar is a genuinely fresh bet, so effective breadth
is **26,059 per year against 1,309** for the daily strategy — twenty times more. Implied
gross Sharpe is 1.54 at TC = 1, or **0.97 at our measured TC of 0.63**.

**But harvesting it costs more than it yields.** Only the turnover-linked part of our cost
scales with frequency (1.51 bps per dollar traded; borrow stays flat at ~25 bps/yr):

| trades per bar | turnover/yr | cost bps/yr | implied net |
|---|---|---|---|
| 25% of book | 441x | 691 | −0.90 |
| 15% | 265x | 425 | −0.18 |
| 10% | 176x | 291 | 0.18 |
| 5% | 88x | 158 | 0.54 |
| 2% | 35x | 78 | 0.76 |

Break-even needs under **12.5% of the book traded per bar**, and even at 2% the implied
0.76 does not beat the 0.91 the daily strategy already delivers. So the twentyfold breadth
gain is real and the cost of collecting it is larger — which is the same trade-off the
whole project keeps running into, now measured at a second frequency.

This is a diagnostic, not a backtest: it assumes our daily cost-per-dollar carries over to
hourly execution, which is optimistic (we cross the spread far more often), and there is
no intraday engine here to verify it. The honest conclusion is that intraday is the most
promising *remaining* lead but only for someone who can execute passively — a real
intraday book earns the spread with limit orders instead of paying it, which changes the
sign of the dominant cost term. That is a different project.

_Reproduce: `quant-capstone study intraday`._

## The liquidity question, tested

Reversal alpha is supposed to live in less liquid names (Avramov-Chordia-Goyal 2006),
which would make a liquid large-cap universe the worst place to look. Tested on S&P 1500
(1,448 names after cleaning).

**Alpha is flat across liquidity; cost is not.**

| ADV decile | median ADV | IC | t-stat | IC per bp of spread |
|---|---|---|---|---|
| Q1 (least liquid) | $19m | 0.0144 | 3.9 | 0.0009 |
| Q5 | $68m | 0.0088 | 2.2 | 0.0008 |
| Q10 (most liquid) | $759m | 0.0144 | 3.0 | **0.0023** |
| **Q1 − Q10** | | **0.0001** | **0.02** | |

The alpha difference between the least and most liquid deciles is **0.0001, t = 0.02** —
indistinguishable from zero, while spreads rise steeply as liquidity falls. Per basis
point of spread paid, the **most** liquid decile is 2.5x the best place to trade this
signal. The full backtest agrees: net Sharpe peaks at a universe of ~100 names and decays
at 200, 400 and 800. Moving down-market is strictly worse, so the universe stays small
and liquid.

Caveat: S&P 1500 small-caps are not truly illiquid — Q1 still trades $19m/day — so this
tests "moderately less liquid", not the microcap regime AC-G describe.

_Reproduce: `quant-capstone study liquidity breadth`._

### A negative result on spread estimation

Estimating the bid-ask spread from daily bars **does not work for liquid US equities**,
and this is worth stating because it is a standard technique. Measured against mega-caps
whose true half-spreads are common knowledge:

| ticker | true | Corwin-Schultz (2012) | Abdi-Ranaldo (2017) |
|---|---|---|---|
| AAPL | ~0.3 bps | 0.00 | 4.91 |
| MSFT | ~0.5 bps | 4.73 | 19.90 |
| XOM | ~0.5 bps | 0.00 | 12.02 |

Off by an order of magnitude in both directions, with no usable ordering. Both estimators
infer the spread from the geometry of the daily range, and for a liquid US name the spread
is ~0.3 bps against ~150 bps of daily volatility — there is nothing to recover at that
ratio, so they measure volatility instead. They are designed for regimes where the spread
is a material fraction of daily movement (micro-caps, emerging markets, pre-decimalisation
data), and this universe is the opposite.

The cost model therefore uses a **calibrated function of dollar volume**
(`56 × ADV_$m^−0.735`, anchored to published spread levels), which is honest about being
an assumption. It puts the top-100-by-ADV universe at 0.44 bps median — meaning the
spec's flat 2 bps was ~4.5x *too expensive*, not too cheap. Setting
`costs.use_estimated_spread=False` recovers the flat rate as a stress case, and the
Phase 1 blend still nets 0.63 there (`study blends`). The break-even multiplier of 6.4x
is what carries this uncertainty.

## Quick start

```bash
uv sync                                         # create .venv and install
uv run pytest                                   # 262 tests, ~20s, no network
uv run quant-capstone run --source synthetic    # end-to-end offline, ~1 min
uv run quant-capstone run                       # the headline study on real data, ~5 min
```

The first real-data run downloads ~10 years of daily bars into `data_cache/` (about
50 MB, a few minutes); later runs read the cache. Output lands in `output/`: `results.md`
plus the figures it references. A full real-data run also writes `results/headline.json`
and records its grid in the trial ledger.

```bash
uv run quant-capstone study all       # regenerate every side study and the write-up figures
uv run quant-capstone study --check   # regenerate the study tables, diff against results/studies
uv run quant-capstone study pead      # or any single study by name
uv run quant-capstone book            # assemble Report/Market-Neutral-Reversal-Book.html
uv run quant-capstone data            # data-quality report (--write-manifest to re-pin inputs)
uv run quant-capstone backtest        # one backtest, gross/net table only
uv run quant-capstone crosscheck      # verify the engine against independent calculations
```

Flags available on `run`, `backtest`, `data` and `crosscheck`: `--source
{yfinance,synthetic}`, `--start`, `--end`, `--universe-size`, `--k`, `--vol-target`,
`--rebalance-every`, `--no-trade-band`, `--timing {next_open,same_close}`, `--outdir`.
`run` also takes `--prior-trials`, `--quick` and `--no-plots`.

## Reproducing every number

Every figure in this README and in the write-up regenerates from the repository.

**The headline** comes from `quant-capstone run`; its numbers are written to
`results/headline.json`, which `tests/test_docs.py` checks this README and the write-up
against.

**The side studies** — everything that is not the headline — are modules in
`src/quant_capstone/studies/`, one per study. Each pins the configuration it was
originally run under (`studies/eras.py`): the defaults moved as the project learned, so a
study that silently picked up today's `Config()` would stop reproducing what it found.
Their tables are committed in `results/studies/`; `study --check` regenerates them all into
a scratch directory and diffs the two.

| study | README section | era |
|---|---|---|
| `progression` | Results table | final |
| `sleeve_survey`, `sleeves`, `blends`, `pead` | Three sleeves; sleeve choice | flat-spread / blend / PEAD |
| `transfer` | Raising the transfer coefficient | blend |
| `fundamentals` | The fundamentals route | blend |
| `intraday` | The intraday route | (IB hourly bars) |
| `liquidity`, `breadth`, `retune`, `turnover` | The liquidity question | flat-spread |
| `events`, `passive`, `regime`, `band` | Phase 2 A, B, C | PEAD |
| `leverage`, `figures` | Write-up §6 and every figure | final |

Regenerating them against the cached data reproduced the archived tables from the
original exploratory scripts **exactly**, with four explained exceptions: the two
walk-forward fold tables (training windows now end five days earlier because of the
embargo fix — wherever the same configuration is selected the test Sharpes agree to the
last digit); the $10m leverage curves (now stored in dollars rather than growth of
$10,000); and `turnover`, the very first exploratory sweep, which ran before the cache and
universe-rebuild fixes and now lands within 0.035 Sharpe of what was recorded. Nothing in
this README quotes that last one.

**The data** cannot be pinned by code: yfinance revises adjusted prices and throttles
into silently empty panels, the IB hourly bars need a live TWS session, and `data_cache/`
is 180 MB and git-ignored. `results/data_manifest.json` records the SHA-256, size, shape
and density of every cached input; `study` checks it before running and warns when the
inputs differ, so a number that moves after a re-download is caught as a data change
before anyone starts debugging the model. To reproduce on another machine, copy
`data_cache/` across rather than re-downloading.

**The write-up** is built from `Report/src` (a stylesheet and one HTML fragment per
section) with the figures in `Report/figures`; `quant-capstone book` inlines them into a
single self-contained HTML file. The two project briefs are `quant_capstone_spec.md` and
`phase2_brief.md`.

## Layout

```
src/quant_capstone/
  config.py       all tunable defaults, and why four differ from the spec's
  linalg.py       projection primitive shared by neutralization and orthogonalization
  factors.py      market proxy, rolling betas, sector loadings
  data/           yfinance | synthetic sources, SEC fundamentals, Form 4, PIT universe, manifest
  signals/        7 price, 6 fundamental and 6 event sleeves, blending, standardization
  risk/           covariance (Ledoit-Wolf / PCA), neutralization, sizing, constraints
  costs/          commissions, reg fees, borrow, financing, slippage/impact — each toggleable
  execution/      passive-fill economics (Phase 2, direction B)
  backtest/       vectorized engine, independent cross-checks, optional vectorbt
  law/            IC, effective breadth, transfer coefficient, implied vs realized IR
  validation/     walk-forward, sensitivity, DSR/PBO, bootstrap, cost sensitivity, trial ledger
  reporting/      metrics, chart style, plots, results.md, headline.json, the write-up builder
  studies/        one module per side study, the eras they ran under, runner and diff
  strategy.py     config -> data -> signal -> portfolio -> backtest
  study.py        the full study: baseline + law + every Section 10 control
results/          committed: headline.json, trial_ledger.json, data_manifest.json, studies/
Report/           the write-up: src/ (sections, stylesheet), figures/, the built HTML
tests/            262 tests; signal, neutralization and cost accounting get priority
```

## The conventions that keep it honest

**Timing.** The signal on day `t` uses data only through the `t` close. Weights derived
from it are traded at the `t+1` open (default) and held to the `t+2` open; the P&L is
booked on `t+1`. Switch to close-to-close with `--timing same_close` — optimistic by
construction, and there only for comparison.

**Gross vs net, IS vs OOS.** Every result is reported all four ways. The gross-to-net gap
is the headline finding, and only the OOS column counts toward the objective.

**Costs.** Commission (per-share with a per-order minimum and a 1%-of-value cap), SEC
Section 31 and FINRA TAF on sales, borrow on the short leg, margin financing on any long
exposure above NAV (zero for the unlevered book), and slippage (half-spread plus a
`c·σ·√(shares/ADV)` impact term). Each has its own toggle, and `cost_multiplier`
scales all of them for the break-even sweep.

**Causality.** The pipeline is causal — trailing windows only, everywhere. That is what
lets the walk-forward run each configuration once over the full history and slice it per
fold instead of re-running it. `tests/test_no_lookahead.py` proves the property by
truncating the panel and asserting every earlier weight is unchanged; if that test
breaks, the walk-forward silently becomes in-sample.

**Everything selectable goes in the walk-forward grid.** `k`, the PEAD weight,
`rebalance_every` and `no_trade_band` are all swept per fold rather than fixed in the
config. Pinning a parameter that was chosen by looking at full-sample results makes every
fold inherit that hindsight and quietly turns the "out-of-sample" number back into an
in-sample one. This project broke the rule twice — first for the holding period, then for
`k` — see [Validation fixes](#validation-fixes).

**Survivorship.** yfinance serves only currently-listed names and we seed from today's
S&P 500, so delisted losers are missing. This biases a reversal backtest *optimistic* —
the absent names are disproportionately the ones whose "oversold, buy it" signal never
reverted. Treat the numbers as an upper bound.

## Three things worth knowing before you read a result

**Raising the transfer coefficient makes things worse.** The Fundamental Law writes
`IR = TC·IC·√BR`, which reads as "raise TC". Measured on the Phase 1 blend, the opposite
holds (`study transfer`):

| variant | TC | gross | net |
|---|---|---|---|
| baseline (1.5% cap) | 0.63 | 0.96 | **0.78** |
| cap 4% | 0.70 | 0.71 | 0.51 |
| cap 8% | 0.82 | 0.66 | 0.47 |
| cap 4% + no sector neutrality | 0.84 | 0.55 | 0.39 |

TC rises 0.63 → 0.84 and net Sharpe falls 0.78 → 0.39, with *gross* falling too, so it is
not merely the extra turnover. The constraints are regularization: the "ideal" weights come
from a noisy alpha estimate, and caps stop the book concentrating wherever the noise
happens to point. This is Jagannathan & Ma (2003), "imposing the wrong constraints helps".
The law's TC term is only a multiplier to maximize if alpha is estimated without error.

**The vol target and the per-name cap fight each other.** A dollar-neutral book of `N`
names capped at `w` each can carry at most about `w · σ_idio · √N` of annualized
volatility. At 1.5% and 100 names that ceiling is ~3.5%, so the 10% target is unreachable
and the book runs at a fraction of its intended size. Sharpe is scale-invariant so the
ratio is unaffected, but realized *return* is proportionally smaller and fixed per-order
commission minimums weigh more on the smaller book.

**Breadth is where the strategy's plausibility is decided.** Rebalancing 100 names daily
looks like 25,193 bets a year, which would make IR = 2 need an IC of only 0.013. It is
really about 1,300 once cross-sectional dependence (24.6 independent directions, not 100
names) and signal persistence are accounted for — a 19x reduction that raises the required
IC fourfold, to 0.104 against the 0.011 we have.

## Not built

The optional cointegration **pairs strategy** (spec §2 secondary) is not implemented.
The natural insertion point is a new `signals/` producer plus a `risk/` sizing mode — the
cost model, backtest engine, law diagnostics and validation harness are all
strategy-agnostic and would be reused unchanged.

Also not modelled, and named in the write-up's limitations section: hard-to-borrow rates
above general collateral, a single financing rate across a decade in which benchmark
rates went from zero to 5%, and fill uncertainty. Each would make the net number worse.
