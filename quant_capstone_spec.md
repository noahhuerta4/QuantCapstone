# Quantitative Finance Capstone — Project Spec

*A cross-sectional, market-neutral, short-term reversal equity strategy with a rigorous cost model, risk pipeline, and overfitting controls. This document is the build brief. Read Section 0 first.*

---

## 0. How to use this spec (read before writing code)

This is a **quant-research** project (backtest only — no live trading needed). The intellectual content is not "produce a high Sharpe"; it is "produce an *honest, net-of-cost, out-of-sample* Sharpe and explain what constrains it." Please follow these rules throughout:

- **Report every performance number twice: gross and net-of-cost.** The gap between them is the headline finding, not a footnote.
- **Report every performance number twice again: in-sample (IS) and out-of-sample (OOS).** OOS is the only number that counts toward the goal.
- **Do not tune parameters to hit the target Sharpe.** If a config only looks good IS, say so. We want the *true* achievable number, plus a diagnosis of the binding constraint (forecasting skill vs. costs). A defensible Sharpe of 1.1 beats a curve-fit 2.5.
- **No look-ahead / no survivorship shortcuts.** Signals at day *t* may use data only through *t* (or the *t* close if execution is next-day open). Universe membership must be point-in-time where feasible.
- **Make every cost explicit and switchable.** Commissions, regulatory fees, borrow, and slippage each get their own toggle so we can run cost-sensitivity sweeps.
- Prefer clear, tested, vectorized code over cleverness. Write unit tests for the signal, the neutralization, and the cost accounting specifically — those are where silent bugs inflate Sharpe.

---

## 1. Objective & definition of done

**Design target:** net-of-cost Sharpe ratio ≥ 2.0, sustained across multiple consecutive multi-month OOS windows (not a single lucky period).

**Definition of done:**
1. A reproducible backtest producing gross+net, IS+OOS performance over a multi-year history.
2. A risk/portfolio-construction pipeline (Section 5) that makes the book market-neutral and vol-controlled.
3. A full IBKR-baseline cost model (Section 6) with slippage, applied to every trade.
4. A Fundamental Law diagnostic (Section 7) decomposing realized IR into IC, breadth, and transfer coefficient.
5. Overfitting controls (Section 10): walk-forward OOS, parameter-sensitivity maps, Deflated Sharpe / PBO, bootstrap CIs, cost-sensitivity.
6. A written results section stating the achieved net OOS Sharpe, whether it clears 2.0, and — regardless — *why* (which term in the Fundamental Law is binding).

If net OOS Sharpe lands below 2.0, that is an acceptable and expected outcome to document; the capstone is graded on rigor, not on the number.

---

## 2. Strategy

**Primary:** Cross-sectional, dollar-neutral, short-term **reversal** on liquid US equities.
- Each rebalance, rank the universe by trailing *k*-day return.
- Go **long recent underperformers**, **short recent outperformers**.
- Cross-sectionally demean the signal so long and short notional net to ~zero (dollar-neutral by construction).
- Hold ~1–3 days; the strategy naturally has high turnover, which is what puts us in the 5–50 trades/day band (throttled per Section 8).

**Rationale / known risk:** short-term reversal alpha is real but concentrated in less-liquid names and largely eaten by transaction costs (Lehmann 1990; Lo–MacKinlay 1990; Avramov–Chordia–Goyal 2006). The central research question is exactly the tension between "liquid enough to trade cheaply" and "enough edge to survive costs." That tension is the project.

**Secondary (optional, for comparison/robustness):** cointegration-based **statistical-arbitrage pairs** (Engle–Granger or Johansen selection; trade each spread's z-score with entry/exit bands). Market-neutral by construction, each pair ≈ one bet (clean breadth story), trade count tunable via pair count and thresholds. Its failure mode (relationships breaking OOS + pair-selection multiple-testing) is good material for Section 10. Build this only after the primary strategy is complete.

---

## 3. Universe & data

- **Prototype data:** `yfinance` daily bars (OHLCV, adjusted). Free and sufficient for daily cross-sectional work.
- **Upgrade path:** IBKR via `ibapi` (user already has an account and experience) for cleaner/longer history and a real point-in-time universe. Keep the data layer behind an interface so the source is swappable.
- **Universe:** ~100–300 liquid US common stocks. Starting proxy: current S&P 500 constituents, then filter to the top ~200 by trailing dollar-volume (ADV). Rebalance membership monthly.
- **Survivorship bias — must address explicitly.** `yfinance` returns only currently-listed names, which inflates cross-sectional backtests (delisted losers vanish). Options, in order of preference: (a) reconstruct point-in-time membership from a historical constituent source; (b) pull a broader delisting-inclusive history via IBKR; (c) at minimum, document the bias, bound its likely direction/magnitude, and treat results as optimistic. Put this in the validity discussion regardless.
- Handle bad prints, splits/dividends adjustments, and missing bars defensively (forward-fill only where legitimate; never fill forward across a listing gap into the future).

---

## 4. Signal construction

- Compute trailing ***k*-day simple return** per name (default `k = 5`; test `k ∈ {1,2,3,5,10}`).
- Optionally orthogonalize the raw reversal signal against short-term market/beta and known factors before ranking (reduces incidental factor bets; see Section 5 neutralization).
- **Winsorize** cross-sectionally (e.g., clip at ±3 MAD or 1st/99th pct) to tame outliers.
- **Cross-sectionally standardize** (z-score or rank-normal) each day so the signal is comparable across dates and sums to ~zero.
- Signal at day *t* uses data through the *t* close; execute at *t+1* open (or *t* close with an explicit assumption). Make the execution-timing assumption a single config flag.

Deliver the signal as a tidy `date × ticker` matrix of standardized scores.

---

## 5. Portfolio construction & risk pipeline (the graded core)

This is where most student projects lose their Sharpe. Build it as an explicit signal → weights pipeline, each stage individually testable.

1. **Covariance estimation Σ.** Sample covariance is ill-conditioned when name count ≈ lookback. Use **Ledoit–Wolf shrinkage** (default) or a **PCA statistical-factor** covariance. Make this a swappable estimator — it materially changes realized risk.
2. **Alpha → target weights.** Two modes, both implemented:
   - *Risk-weighted (default, robust):* size positions inversely to volatility / by risk contribution, direction from signal sign, magnitude from signal strength.
   - *Mean-variance:* maximize `wᵀα − λ·wᵀΣw` subject to the constraints below. Keep λ configurable; expect MVO to be sensitive to Σ noise (good discussion point).
3. **Neutralization (what actually makes it market-neutral):**
   - Dollar-neutral: `Σ wᵢ = 0`.
   - Beta-neutral: `Σ wᵢ βᵢ = 0`.
   - Sector-neutral and, ideally, factor-neutral: trade the **residual** after regressing the signal on market/size/value/PCA factors.
4. **Sizing:** scale gross exposure to target **constant ex-ante annualized volatility** (default 10%). Document fractional-Kelly and equal-risk-contribution as alternatives and why full Kelly's drawdowns are unacceptable.
5. **Hard constraints:**
   - Max weight per name: default **1.5% NAV** (test 1–3%).
   - Max gross leverage: cap (e.g., 2–3× gross), scaled down to meet the vol target.
   - Max net sector exposure.
   - **Liquidity cap:** position ≤ ~1% of the name's ADV (this doubles as market-impact control).
   - Optional: drawdown-triggered de-leveraging.

Output: a `date × ticker` weights matrix that respects all constraints, feeding the backtester.

---

## 6. Transaction costs — IBKR baseline (where the Sharpe actually goes)

Model with IBKR as the baseline. **Verify the current schedule at build time — these rates drift.** Approximate figures:

- **Commissions:** IBKR US equities — fixed ≈ `$0.005/share` (`$1.00` min per order, capped at 1% of trade value), or tiered ≈ `$0.0035/share` plus exchange/clearing pass-throughs (~`$0.35` min). The **per-order minimum** is critical: fragmenting into many tiny orders makes them uneconomic — this is a direct argument for fewer, larger trades and shapes the 5–50/day target.
- **Regulatory fees (sells):** SEC Section 31 fee and FINRA TAF on sales — small but nonzero; include them (rates change).
- **Borrow cost (short leg):** apply a borrow rate to shorts. General-collateral ≈ 0.25–0.5%/yr; hard-to-borrow far higher. Don't ignore it — it's a persistent drag on a market-neutral book.
- **Slippage / market impact — the dominant term, usually dwarfs commissions.** Implement two models behind a flag:
  - *Baseline:* charge **half the bid-ask spread per side** (assume ~1–3 bps half-spread for liquid names; make it a parameter). A simple all-in "N bps per side" (e.g., 5 bps) is an acceptable first pass.
  - *Better:* half-spread **+ an impact term** scaling with order size, e.g. `impact ≈ c · σ · √(order/ADV)`. Calibrate/​assume `c`, and justify.

**Every trade pays commission + fees + slippage (+ borrow on shorts).** Report all results **gross and net side by side**; the delta is the headline. A backtest that models commissions but silently ignores slippage reports a fantasy Sharpe — do not do this.

---

## 7. Fundamental Law of Active Management (diagnostic module)

Use the law as a *measurement tool*, not decoration. For a dollar-neutral book, IR ≈ Sharpe, so this directly models the target.

- Law: `IR = TC · IC · √BR` (Clarke–de Silva–Thorley 2002), where **IC** = cross-sectional correlation of forecast vs. realized forward return, **BR** = number of independent bets/year, **TC** = transfer coefficient (fraction of ideal alpha surviving constraints + costs).
- **The seductive lie to expose:** naïvely, rebalancing *N* names daily gives `BR ≈ 252N`, implying IR = 2 needs only `IC ≈ 2/√(252N) ≈ 0.009` for N=200 — a trivially small correlation. Show why this is wrong: (a) cross-sectional return correlation makes bets non-independent, so **effective breadth ≪ 252N**; (b) signal autocorrelation means holding the same tilt for days is not a fresh daily bet; (c) TC then knocks IR down further via caps and — dominant — transaction costs.
- **Implement:**
  - Measure realized **IC** per period (mean and IR of the IC series — "IC breadth").
  - Estimate **effective breadth** (e.g., from the eigenstructure of the return correlation matrix, or via signal autocorrelation adjustment).
  - Estimate **TC** (correlation between ideal unconstrained weights and actual post-constraint, post-cost weights).
  - Back out **implied IR** and compare to the **realized** net Sharpe. The residual tells us whether **IC (forecasting)** or **TC (costs/constraints)** is the binding constraint. That comparison is a paper-worthy result.

---

## 8. Trade-frequency control (hitting 5–50 trades/day)

Turnover, not signal, sets trade count. Tune it with:
- **Universe size** (fewer names → fewer trades).
- **No-trade bands / rebalance threshold:** only trade a name when its target weight moves beyond a band (cuts churn dramatically, improves net Sharpe).
- **Rebalance frequency** (daily vs. every 2–3 days).
- **Min-trade-size filter** (skip trades too small to clear the per-order commission minimum — ties back to Section 6).

Log realized trades/day and confirm it sits in the 5–50 band; treat the band as a constraint the tuning must satisfy.

---

## 9. Backtest engine

- **Primary:** a **custom vectorized backtester** in `numpy`/`pandas`. For a daily cross-sectional "rank N names, rebalance the whole book" strategy this is usually the *best* fit — event-driven frameworks are awkward here, and rolling our own makes cost/turnover accounting fully explicit and defensible.
- **Cross-check:** re-run key results in **`vectorbt`** to catch look-ahead and accounting bugs. Discrepancies between the two engines must be understood before trusting results.
- Engine requirements: point-in-time discipline (no future data), explicit `t → t+1` execution, per-trade cost application, daily P&L, turnover, gross/net exposure, and full position/trade logs.
- (If the secondary pairs strategy or any intraday extension is pursued, `backtrader` suits event-driven/few-instrument work — but daily cross-sectional stays in the vectorized engine.)

---

## 10. Validation & overfitting controls (this is also risk)

- **Walk-forward / expanding-window OOS:** fit any parameters on a train window, evaluate on the next untouched window, roll forward. Only OOS results count.
- **Parameter-sensitivity heatmaps:** sweep `k`, holding period, caps, vol target, no-trade band. A Sharpe that **cliffs** on one parameter is fitted; a broad plateau is robust. Show the surfaces.
- **Deflated Sharpe Ratio & Probability of Backtest Overfitting** (Bailey & López de Prado): discount the reported Sharpe for the number of configurations tried. Track the trial count honestly and feed it in.
- **Block bootstrap** of the return series for Sharpe confidence intervals (report the CI, not just a point estimate).
- **Cost-sensitivity sweep:** raise slippage/commissions and find the level at which the edge dies. The margin between baseline and break-even cost is the strategy's robustness margin — a key deliverable.

---

## 11. Metrics & reporting

Compute (gross and net, IS and OOS):
- Sharpe, Sortino, Calmar.
- Max drawdown **and its duration**.
- Annualized return and volatility.
- Turnover (annualized and per-day trade count).
- Gross and net exposure over time; realized beta (confirm ~0).
- Hit rate; average win/loss.
- VaR and CVaR (historical + parametric).
- **Rolling 6-month Sharpe** plot (exposes regime fragility) and the equity curve.
- A cost-attribution breakdown: bps lost to commissions vs. fees vs. borrow vs. slippage.

---

## 12. Suggested repo layout & build order

```
quant_capstone/
  data/            # loaders: yfinance + ibkr behind a common interface; universe builder
  signals/         # k-day reversal, winsorize, standardize, (optional) factor orthogonalization
  risk/            # covariance (Ledoit-Wolf / PCA), neutralization, sizing, constraints
  costs/           # commissions, reg fees, borrow, slippage/impact models (each toggleable)
  backtest/        # vectorized engine; vectorbt cross-check harness
  law/             # IC, effective breadth, TC, implied-vs-realized IR
  validation/      # walk-forward, sensitivity sweeps, DSR/PBO, bootstrap, cost-sensitivity
  reporting/       # metrics, plots, cost attribution, final results notebook
  tests/           # unit tests: signal, neutralization, cost accounting (priority)
  config.py        # all tunable defaults in one place
```

**Build order:** (1) data + universe → (2) signal → (3) *cost model early* → (4) minimal vectorized backtest producing gross+net → (5) risk/portfolio pipeline → (6) validation harness → (7) Fundamental Law module → (8) reporting → (9) optional pairs strategy. Getting a crude end-to-end net-of-cost number early (steps 1–4) prevents building an elaborate pipeline on top of a fantasy result.

## 13. Starting defaults (all tunable via `config.py`)

`k=5` · universe ≈ top 200 by ADV · daily rebalance · vol target 10%/yr · per-name cap 1.5% NAV · gross cap ~2–3× · liquidity cap 1% ADV · commission `$0.0035/share` tiered (or `$0.005` fixed) with per-order min · slippage = half-spread (~2 bps) + `c·σ·√(size/ADV)` · borrow 0.5%/yr GC · execute `t+1` open.

## 14. Dependencies

`numpy`, `pandas`, `scipy`, `scikit-learn` (Ledoit–Wolf), `yfinance`, `ibapi`, `vectorbt`, `matplotlib`/`plotly`, `statsmodels` (cointegration, regressions). Pin versions.

---

*Note for the report: backtested performance is not live performance — which is precisely why Sections 6 and 10 (costs and overfitting controls) carry most of the weight. Treat the honest OOS net Sharpe and its confidence interval as the true deliverable.*
