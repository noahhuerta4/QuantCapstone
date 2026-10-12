# Capstone Phase 2 — Pushing Net OOS Sharpe Past 1.0

*Extension brief for the existing Market-Neutral Reversal Book. Read Section 0 first. This assumes the Phase-1 package already exists: the three-sleeve blend, Ledoit–Wolf risk model, orthogonal-projection neutralization, IBKR-baseline cost model, walk-forward harness, engine cross-check, and the Fundamental-Law decomposition. **Extend that package; do not rebuild it.***

---

## 0. Context and how to use this (read before writing code)

**Where the project stands:** net OOS Sharpe 0.84 (gross 1.04), walk-forward over 13 folds, 100 most-liquid US large-caps, 2015–2024. The goal is a *legitimate* net OOS Sharpe above 1.0.

**The diagnosis this brief is built on.** The Phase-1 report already localized the constraint. Three findings converge on one root cause — the book has too few *independent* sources of risk/alpha:
- Ex-ante vol settles at 3.34% against a 10% target because the per-name cap binds — the book physically cannot express more risk.
- Widening the universe doesn't help (net Sharpe peaks at 100 names, decays at 200/400/800).
- Loosening constraints *lowers* Sharpe (the Jagannathan–Ma regularization result), so concentrating doesn't help either.

Through `IR = TC · IC · √BR`: **TC is already at its noisy-alpha optimum, IC (0.0113) is small but real, so BR is the lever.** Two ways to raise BR — more uncorrelated sleeves, or faster turnover — plus a bounded IC lever. The three directions below are exactly those levers.

**The integrity rules from Phase 1 still bind, and one is now a hard constraint:**
- Report every number gross **and** net, in-sample **and** out-of-sample. Only net OOS counts toward the goal.
- No look-ahead; every new signal goes through the same causal / point-in-time discipline and must pass the existing causality (weight-invariance-under-truncation) test.
- **Do not tune to hit 1.0.** State the true achievable number and its confidence interval.
- **The validation budget is now the binding constraint, not compute.** The Deflated Sharpe is already 0.718 at **249 trials**. Every configuration examined from here — including exploratory ones — must be counted into the DSR. See Section 1.1: this changes *which kind* of search is worth doing.

---

## 1. The frame, and the success test

Each direction must move a *specific* term of the Fundamental Law, and success is measured by re-running the Phase-1 decomposition afterward to confirm the intended term actually moved:

| Direction | Intended lever | What should move |
|---|---|---|
| A — Event-driven sleeves | BR (independent bets) + escape the vol ceiling | effective breadth rises above 1,309; realized vol rises toward target |
| B — Passive-execution intraday | BR (via turnover) + TC_costs | effective breadth ≫ daily; net/gross ratio rises above 0.83 |
| C — Regime-conditional combination | IC (forecasting skill) | measured IC rises above 0.0113 |

**Definition of done:** net OOS Sharpe > 1.0 sustained across the walk-forward folds, with the gain attributable to a term that demonstrably moved — *and* a Deflated Sharpe that still clears significance at the new (honestly counted) trial total.

### 1.1 Why the order of directions matters — the DSR trap

There are two ways to raise a backtested Sharpe: bring **new information**, or **search parameter space harder**. Your own DSR punishes the second — every extra config inflates the trial count and discounts the result. So the DSR-efficient way to buy Sharpe is to add genuinely new, independent *information* (new data → real, non-search edge), not to sweep more parameters on the data you already have.

This directly orders the work: Direction A and Direction B bring new information (new data sources / a new frequency), so a sleeve that works there is a real edge rather than a search artifact. Direction C is largely parameter/regime search on existing data — the most trial-hungry and the most DSR-taxed. Hence A → B → C.

---

## 2. Direction A (do this first): a genuine multi-sleeve event-driven book

**Principle, straight from your PEAD finding.** Your sharpest Phase-1 result was *why* earnings surprise earns a place and static value doesn't: an **event clock** gives breadth — staggered fiscal calendars keep the cross-section turning — where a static ratio (autocorr 0.997) delivers under one independent bet a year. So the search is not "better forecasts," it's *more event clocks*, each roughly uncorrelated with the price sleeves and with each other. Every one that works raises effective breadth **and** lifts the 3.34% vol ceiling, hitting the root cause directly.

### 2.1 The inclusion gate (apply to every candidate sleeve before it joins the blend)

A sleeve joins only if it passes all of:
1. **Standalone IC** significant (report IC, t-stat, IC hit rate), computed on next-day tradable returns, point-in-time.
2. **Autocorrelation low enough for breadth** — this is the gate the value sleeves failed. Compute signal autocorrelation and effective breadth; reject anything near the 0.997–0.9995 band regardless of IC.
3. **Low correlation to existing sleeves** (target < ~0.3, like overnight-to-reversal at 0.30 and PEAD at −0.01). Correlated variants are redundant.
4. **Positive *marginal* contribution to net OOS Sharpe** in the walk-forward, not just standalone — and the blend weight goes into the walk-forward grid, re-selected per fold, never pinned.

Every candidate tried counts as trials toward the DSR — including ones that fail the gate. Budget accordingly; prefer a short list of theory-motivated sleeves over a wide sweep.

### 2.2 Candidate sleeves — with honest data notes

Ranked by (free + high-breadth + reuses your pipeline). I flagged loosely-named ones last conversation; here is the accurate version.

**A1 — Insider transactions (Form 4). Strongest candidate.**
- *Source:* SEC EDGAR Form 4 ownership filings — structured XML (`ownershipDocument`), free. Filed within 2 business days of the trade, so genuinely fresh; continuous arrival across names → good breadth.
- *Signal:* net insider buying (buys − sells), dollar- or shares-weighted, normalized by market cap or shares outstanding, decayed over ~20–60 days.
- *Sophisticated refinement:* separate **opportunistic** from **routine** (calendar-regular) trades — the opportunistic subset carries the signal (Cohen–Malloy–Pomorski 2012). Weighting by insider role (CEO/CFO more informative) is a second refinement.
- *Why it fits:* new information source, event-clocked, ~independent of price and earnings sleeves.

**A2 — Earnings refinements on the XBRL you already pull. Cheapest to build.**
- *Source:* the SEC XBRL facts already in the pipeline — no new ingestion.
- *Two distinct, roughly-independent sleeves from the same events:*
  - **Earnings-announcement short-term reversal** — the 1–3 day *reversal* of the announcement-day move (overreaction), fast-decaying → high breadth. This is a reversal on the same event where your existing PEAD is drift; the two are near-orthogonal by construction.
  - **Revenue SUE** — surprise on revenue, a separate dimension from EPS surprise; adds breadth on the earnings clock.
- *Skip:* accruals and other annual-cadence fundamentals — same low-breadth failure as value.

**A3 — 8-K information-arrival drift. Good breadth, moderate effort.**
- *Source:* EDGAR filing index / full-text search — 8-Ks carry item codes, free. Continuous arrival.
- *Honest scope:* item codes *alone* are a weak signal. The workable version is **8-K-conditioned reaction drift** — an unscheduled 8-K (exclude routine Item 2.02 earnings, already captured by A2) as an information-arrival event, using the initial price/volume reaction as the conditioning variable and fading over days. Build it as "reaction to information arrival," not "item code → return."

**A4 — Realized buyback intensity. Secondary.**
- *Source:* `PaymentsForRepurchaseOfCommonStock` from the XBRL cash-flow statement — free, already reachable. Quarterly cadence, so moderate breadth (weaker event clock than A1–A3). The *announcement* version (8-K/press release) has a better clock but needs text parsing — defer that.

**Flagged / likely to fail the gate — test only if cheap, expect rejection:**
- **13F institutional holdings changes** — quarterly with a 45-day lag → low freshness and quarterly cadence. Expect it to fail gate #2 for the same reason the value sleeves did; running it and showing the rejection is itself a nice callback, but don't invest heavily.
- **Management guidance revisions** — *not* cleanly in us-gaap XBRL (it's narrative text in 8-K exhibits, inconsistently tagged). Requires NLP on filings — a real project of its own. Out of scope unless you specifically want an NLP lift.

### 2.3 Integration
Each new sleeve is computed to a `date × ticker` score, clipped at ±3 MAD, standardized cross-sectionally, then blended — reusing the *existing* standardize/blend/neutralize/cost stages unchanged. No change to the risk pipeline. The only new code is ingestion + per-sleeve scoring + the gate diagnostics.

**Expected payoff:** the most reliable path to 1.0 with data you already have or can get free. Partial success still counts — each independent sleeve pays off on its own, so this de-risks the whole phase.

---

## 3. Direction B (high-ceiling swing): passive-execution intraday reversal

**The thesis — cost inversion, not cost reduction.** Your intraday study already showed hourly reversal at IC 0.0095, autocorrelation ≈ 0, **26,059 effective bets/yr (≈20× the daily book)**, implied gross IR ~0.97. You closed it because you'd pay the spread. But your cost attribution shows borrow (38.8%) is fixed rent independent of turnover, and slippage+commission (35 bps) is the *only* turnover-linked cost. If execution is **passive** — resting limit orders that *earn* the half-spread instead of paying it — that dominant term flips sign and higher turnover becomes nearly free. That inversion is what unlocks the 20× breadth. This is the single change most likely to push well past 1.0.

**Build it in two stages — cheap feasibility gate before the expensive sim.**

**Stage 1 — parametric feasibility (do this first; hours, not weeks).**
Reuse the hourly reversal signal you already measured. Model passive execution with two parameters:
- `p` = fraction of intended volume that fills passively (earns `+half_spread`); the unfilled remainder is either missed alpha or crossed at the end of the bar (`−half_spread`).
- `a` = adverse-selection haircut: filled orders realize a forward return worse than intended by `a · σ`, because you get filled precisely when the market is moving against you.
Sweep `(p, a)` and map the region where net Sharpe beats the daily book. This tells you the *required* fill quality before building anything. If no plausible `(p, a)` clears the daily book, stop here and document it.

**Stage 2 — microstructure fill model (only if Stage 1 is encouraging).**
- *Data caveat (important):* hourly bars **cannot** model within-bar fills — Stage 2 needs finer data (1-minute or tick via IBKR `reqHistoricalTicks` / `reqTickByTickData`). Your existing 59-ticker hourly set is a feasibility sample, not enough for a fill model. Scope the data pull explicitly.
- Model queue position, fill probability as a function of order aggressiveness and contemporaneous volume, and realistic adverse selection estimated from the tick data rather than assumed.

**Definition of done / honest framing:** this reframes the project from alpha research to *execution* research, and the deliverable is the fill model plus the `(p, a)` feasibility map — not just a Sharpe number. Report the intraday result only with the fill assumptions stated as prominently as the Sharpe.

---

## 4. Direction C (finishing pass, not a foundation): regime-conditional combination

**The IC lever.** Your rolling six-month Sharpe shows real regime dependence, and your blend is a *static* 45/45/10. Conditioning targets IC directly.
- **First, the interpretable version:** make sleeve weights a function of a regime variable — cross-sectional return dispersion (reversal pays more when dispersion is high), and/or market realized volatility. This is a handful of parameters, theory-motivated, and legible.
- **Optionally, a nonlinear combiner:** gradient-boosted trees over sleeve scores + conditioning variables + interactions, producing one forecast.

**The hard constraint (from Section 1.1):** you are at 249 trials with DSR 0.718. An ML sweep adds dozens of hyperparameter configs and the DSR will eat the gains — you have a *validation budget*, not a compute budget. Therefore:
- Use **very few, pre-registered** configurations, chosen on theory before seeing results.
- Mandatory **purged and embargoed cross-validation** (López de Prado) so overlapping 10-day holds don't leak across the train/test boundary.
- Every config counts into the DSR trial total, honestly.

Treat this as a finishing pass over a book already improved by A and B — a static blend on a stronger sleeve set may already clear 1.0, in which case C's marginal value (net of its DSR cost) may be negative. Check that explicitly.

---

## 5. How to proceed — sequencing and rationale

1. **Direction A first.** Lowest risk, reuses the pipeline, brings new information (DSR-efficient), and each sleeve pays off independently so partial success still moves the number. Start with **A1 (Form 4)** and **A2 (earnings refinements)** — highest breadth per unit of effort — then **A3**, then **A4**. Re-run the Fundamental-Law decomposition after each: confirm effective breadth rose and the vol ceiling lifted.
2. **Direction B second** — the high-ceiling swing. Do **Stage 1 only** initially; commit to the Stage-2 data pull and microstructure model only if the feasibility map is encouraging. A working passive-fill model is the single thing most likely to carry you well past 1.0.
3. **Direction C last**, and only if A+B haven't already cleared 1.0 comfortably — and even then, weigh its DSR cost against its gain.

**After every direction**, re-run the *full* Phase-1 validation harness (walk-forward, block bootstrap CI, cost-sensitivity, DSR/PBO with the updated trial count) and the IR decomposition. A Sharpe that rises while the DSR falls below significance is not progress — say so.

---

## 6. Cross-cutting requirements

- **Reuse, don't duplicate:** the cost model, neutralization algebra, sizing, walk-forward harness, and engine cross-check stay exactly as they are. New code is limited to data ingestion, per-sleeve scoring, the fill model (B), the combiner (C), and their diagnostics.
- **Point-in-time discipline** applies to every new source: Form 4 visible on/after its filing date, 8-K on/after its filing timestamp, XBRL facts on/after filing date (already handled). Each must pass the existing causality test.
- **Trial accounting is now first-class:** maintain the running trial count as a tracked artifact and feed it to the DSR on every reported result.
- **Same limitations carry forward** (survivorship bias, calibrated-not-measured spreads, single 10-year regime) and any new source adds its own — Form 4 coverage gaps, 8-K parsing errors, tick-data survivorship. Document each where it changes how a number should be read.

## 7. Suggested additions to the existing repo

```
quant_capstone/
  data/
    edgar_form4.py       # A1: Form 4 ownership XML → insider-flow panel
    edgar_8k.py          # A3: 8-K filing index + item codes → event panel
    ibkr_intraday.py     # B Stage 2: minute/tick pull via ibapi
  signals/
    insider.py           # A1 sleeve (opportunistic vs routine split)
    earnings_extra.py    # A2: announcement reversal + revenue SUE (reuses XBRL)
    event_8k.py          # A3: 8-K-conditioned reaction drift
    buyback.py           # A4: realized repurchase intensity
  execution/
    passive_fill.py      # B: (p, a) parametric model, then microstructure sim
  signals/
    combiner.py          # C: regime-conditional weights; optional GBM
  validation/
    trial_ledger.py      # running, honest trial count feeding the DSR
  # risk/, costs/, backtest/ — unchanged
```

---

*The honest north star for this phase: the extra ~0.16 of Sharpe must come from new information (Directions A and B), because your own Deflated Sharpe will correctly refuse to credit it if it comes from searching harder. Report the net OOS Sharpe, its bootstrap CI, and the DSR at the final trial count as one triple — that triple, not the headline, is the claim.*
