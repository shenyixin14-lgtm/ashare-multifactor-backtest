# ashare-multifactor-backtest

A cross-sectional multi-factor backtesting framework for the CSI 300, built from scratch in Python on a real 12-year daily panel (~288 constituents, ~1.2M rows). The framework is organized into clean, swappable layers and ships with a test suite that guards the invariants that matter most in factor research — chiefly, the absence of look-ahead bias.

The headline finding is deliberately unglamorous, and that is the point: the factors carry a real but fragile edge that does not survive realistic transaction costs. The project is built around honest evaluation rather than an inflated Sharpe.

## Why this project

Most backtests fail quietly. A filter applied one day too early, a merge that duplicates rows, a normalization that silently reverts to equal weight — none of these throw an error, and all of them flatter the results. This framework was built by repeatedly finding such failures on real data, understanding the mechanism behind each, and encoding the fix as a structural guarantee or a test. The Backtest Pitfalls section below documents the ones worth knowing.

## Architecture

Data flows top to bottom through five core layers, plus a single isolated zone where future returns are touched.

```
DataLayer              load prices, dedupe, derive today_chg / next_ret (kept separate)
   |
   +-- [Controlled Zone]  measure_ic   the ONLY place that touches next_ret
   |
SignalLayer            compute factors, align direction, z-score  (never sees next_ret)
   |
PortfolioConstructor   scores -> explicit dollar-neutral weights
   |
ExecutionEngine        tradability filter + renormalize to dollar-neutral
   |
Evaluator              Sharpe, max drawdown, turnover-based cost, IC / ICIR
```

The central design principle is **physical isolation of look-ahead bias**. Future returns (`next_ret`) are split out at the data layer and never enter the weight-generation pipeline. They appear in exactly two places: once in `measure_ic` (to decide factor direction, on the training split only) and once at settlement (to compute realized returns). The `build_signals` function has no `next_ret` in its signature or body — selection and weighting cannot peek at the future even by mistake.

## Key results (CSI 300, out-of-sample 2019 onward)

| Metric | Value | Reading |
|---|---|---|
| OOS Sharpe (zero cost) | **0.65** | the factors carry a genuine edge |
| Average daily turnover | **1.30** | high-turnover, short-horizon signal |
| Break-even one-way cost | **~3-4 bps** | edge is wiped out above this |

The story these three numbers tell: there is a real signal, but it is high-turnover and thin. Once realistic A-share transaction costs (typically well above 4 bps one-way) are applied, the strategy turns negative. The factor is statistically significant but not tradable as-is — a conclusion worth far more than a frictionless Sharpe of 2.

Single-factor and composite comparison (OOS ICIR):

| Signal | ICIR |
|---|---|
| 5-day reversal | 1.90 |
| IC-weighted composite | 1.81 |
| 20-day "momentum" | 0.99 |

The composite does **not** beat the strongest single factor. This is explained in the findings below.

## A finding worth highlighting: the "momentum" factor is actually reversal

The 20-day momentum factor has a **negative** IC on the A-share training set, which prompted the question of whether it is a momentum signal at all. Measuring IC across horizons confirmed it is not:

| Window | IC |
|---|---|
| 20d | -0.022 |
| 60d | -0.012 |
| 120d | -0.007 |
| 250d | -0.003 |
| 400d | -0.006 |
| 600d | -0.007 |

IC is negative across **every** horizon tested (20-600 days). A-shares show reversal at all these scales — the classic medium-term momentum effect documented in US equities (positive IC at 60-250 days) simply does not appear here. The reversal effect is weakest near 250 days and strengthens on either side.

The implication: this so-called "momentum + reversal" model is really *two reversal signals at different horizons*. That explains their elevated correlation (~0.46) and why combining them fails to diversify — the composite averages two versions of the same effect rather than blending independent signals, so it lands between the two and underperforms the stronger one.

(Caveat: the 400/600-day windows rest on fewer, older stocks due to the lookback requirement, so the long end is less reliable than the short end. The core conclusion — no momentum regime in A-shares over these horizons — is robust.)

## Backtest pitfalls (each found on real data, fixed at the source)

1. **Look-ahead via the tradability filter.** Filtering on `next_ret` (next-day return) instead of `today_chg` inflated the OOS Sharpe roughly 2x. The filter variable was the same as the return being earned, so it systematically removed the positions that would have lost money. Fix: tradability uses only information known at order time.

2. **Selection bias from filter placement.** Applying the price-limit filter *before* computing IC removed extreme-move stocks from the cross-section, inflating ICIR by ~0.4 (2.3 -> 1.9). Verified by control: removing the early filter made an independent implementation's ICIR match to the third decimal. Fix: IC is computed on the full cross-section; tradability lives in the execution layer.

3. **Time-series factors must be computed before the train/test split.** Splitting first leaves the test set without its lookback window. Factor computation (which only uses past prices) runs on the full panel; only fitting (IC, direction) is restricted to train.

4. **`groupby.apply` drops the group key.** Depending on the pandas version, `groupby('date', group_keys=False).apply(...)` silently removes the `date` column. Fix: update only the target column rather than replacing the whole frame, so `date` never participates in reassembly.

5. **Assigning a column from `groupby.apply` can silently misalign.** `df['x'] = grp.apply(...)` matches on index; a `.copy()` inside the function breaks alignment and leaves stale values with no error. Caught by the dollar-neutral test.

6. **`merge` one-to-many row inflation.** Duplicate `(date, code)` keys in `next_ret` caused the settlement merge to duplicate rows, blowing up net exposure to 0.45. Fix: dedupe at the data layer; a test asserts the merge does not change row count.

7. **A-share panel quirks: duplicate records and stripped leading zeros.** ~78k duplicate `(date, code)` rows and codes read as integers (`1` instead of `000001`) corrupted `pct_change` and broke merges. Fix: zero-pad, dedupe, and derive returns once, in order, at the data layer.

8. **Re-normalization vs. equal-weight reassignment.** After zeroing untradable names, scaling survivors by their group sum preserves the relative weights of an IC-weighted book; reassigning `1/N` would silently overwrite the upstream weighting. Only the former is correct.

A theme runs through these: each more rigorous fix moves an inflated metric down to a lower but honest value (look-ahead 2x Sharpe, selection bias 0.4 ICIR, costs turning the edge negative).

## Tests

Five groups of invariant assertions guard the framework:

- **Weights** — dollar-neutral net exposure, legs sum to +/-1, no NaN
- **Look-ahead** — no `next_ret` in the signal pipeline, alignment correct, `aligned_ic == |raw_ic|`
- **Data quality** — no duplicate keys, merge does not inflate rows, codes are 6-digit
- **Determinism** — identical IC across repeated runs
- **Edge cases** — a too-small cross-section flattens the day rather than crashing

## Usage

```python
features, next_ret  = load_data()                              # DataLayer
raw_ic, ic_stats    = measure_ic(features, next_ret, cols, SPLIT)   # controlled zone
signals, aligned_ic = build_signals(features, raw_ic)          # clean zone

# IC-weighted composite, then split out-of-sample
w = {c: aligned_ic[c] / sum(aligned_ic.values()) for c in cols}
signals['composite_icw'] = sum(w[c] * signals[c] for c in cols)
test_signals = signals[signals['date'] >= SPLIT].copy()

# portfolio -> execution -> settlement
weights, backtest = run_backtest(test_signals, 'composite_icw', next_ret)
print(calc_sharpe_max_dd(backtest))
```

Adding a new factor requires writing one function and registering it in `FACTORS`; the rest of the pipeline is unchanged.

## Data

Daily CSI 300 constituent prices via `akshare` (Tencent endpoint), cached locally. ~288 stocks, ~1.2M rows, ~12 years. Train/test split at 2019-01-01.

## Limitations

- Early years have few constituents, so cross-sections can be concentrated; results are most reliable on the later, stable period.
- Costs are modeled as `cost_rate x turnover`; market impact and borrow costs for shorts are not separately modeled.
- The universe is point-in-time approximate, not survivorship-bias-free at the membership level.
