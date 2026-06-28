#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 11:50:50 2026

@author: shenyixin
"""

import os
import time
import numpy as np
import pandas as pd
import akshare as ak
SYMBOL = "000300"                 # CSI 300 index code
MOMENTUM_WINDOW = 20          # momentum look-back (trading days)
REVERSAL_WINDOW = 5              # reversal look-back (trading days)
Q = 0.2                          # long/short quantile (top & bottom 20%)
COST_RATE = 0.001                # one-way transaction cost per unit turnover
SPLIT = "2019-01-01"            # train / test split date
MIN_STOCKS = 10                  # min names per leg before a day is skipped
PRICE_LIMIT = 0.098              # +/- price-limit threshold (A-share ~10%)
TRADING_DAYS = 252               # annualization factor
RETRIES = 2                      # retries per stock on data fetch
SLEEP = 0.5                      # delay between stock fetches (rate limiting)
CACHE = "multifactor_raw.csv"
COST_GRID = [0, 0.0002, 0.0005, 0.0008, 0.001, 0.0015, 0.002]
# ============================================================
# 1. DATA LAYER — fetch clean prices, isolate next_ret
# ============================================================

def get_one(code, retries=RETRIES):
    '''Fetch raw daily close prices for a single stock.
    return: a DataFrame [date, close, code], or None if all retries fail.'''
    for attempt in range(retries):
        try:
            name = ('sh'+code) if code.startswith('6') else ('sz'+code)
            raw = ak.stock_zh_a_hist_tx(symbol=name)
            result = raw[['date','close']].copy()
            result['code'] = code
            return result
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {type(e).__name__}, retrying...")
            time.sleep(3)
    return None

def get_more(codes):
    '''Fetch and concatenate price panels for many stocks.'''
    frame = []
    total = len(codes)
    success = 0
    for i, code in enumerate(codes, 1):
        one = get_one(code)
        if one is not None:
            frame.append(one)
            success += 1
        print(f"progress {i}/{total} | success {success} | current {code}")
        time.sleep(SLEEP)
    panel = pd.concat(frame, ignore_index=True)
    return panel

def load_data(cache=CACHE):
    '''Load price panel, compute derived columns, return features / next_ret.
    features: [date, code, close, today_chg]   (decision-time, no look-ahead)
    next_ret: [date, code, next_ret]           (held out for settlement only)'''
    if os.path.exists(cache):
        panel = pd.read_csv(cache)
    else:
        cons = ak.index_stock_cons(symbol=SYMBOL)
        codes = cons['品种代码'].astype(str).str.zfill(6).tolist()
        panel = get_more(codes)
        panel.to_csv(cache, index=False)
    panel['code'] = panel['code'].astype(str).str.zfill(6)
    panel['date'] = pd.to_datetime(panel['date'])
    panel = panel.drop_duplicates(subset=['code','date'])
    panel = panel.sort_values(['code','date'])
    panel['today_chg'] = panel.groupby('code')['close'].pct_change()
    panel['next_ret']  = panel.groupby('code')['today_chg'].shift(-1)
    features = panel[['date', 'code', 'close', 'today_chg']].copy()
    next_ret = panel[['date', 'code', 'next_ret']].copy()
    return features, next_ret

# ============================================================
# 2. FACTOR DEFINITIONS — registry + dispatch
# ============================================================

def momentum(panel, window=MOMENTUM_WINDOW):
    '''N-day momentum factor, computed per stock.'''
    return panel.groupby('code')['close'].pct_change(window)

def reversal(panel, window=REVERSAL_WINDOW):
    '''N-day reversal factor, computed per stock.'''
    return panel.groupby('code')['close'].pct_change(window)

FACTORS = {'momentum': momentum, 'reversal': reversal}

# ============================================================
# 3. SIGNAL LAYER — factors, IC, alignment, standardization
# ============================================================

def compute_factors(panel, factors=FACTORS):
    '''Compute all registered factors and add each as a column.'''
    panel = panel.copy()
    for name, func in factors.items():
        panel[name] = func(panel)
    return panel

def compute_ic(panel, cols):
    '''Cross-sectional Rank IC (Spearman) per factor.
    return: {col: {'ic_mean', 'icir', 'ic_win'}}.
    Skips days where the cross-section is too small or constant.'''
    panel = panel.copy()
    def cross_ic(x, col, min_stocks=MIN_STOCKS):
        x = x.dropna(subset=[col, "next_ret"])
        if len(x) < min_stocks or x[col].nunique() <= 1 or x['next_ret'].nunique() <= 1:
            return np.nan
        return x[col].corr(x["next_ret"], method="spearman")
    ic_sum = {}
    for col in cols:
        daily_ic = panel.groupby("date").apply(cross_ic, col=col).dropna()
        ic_mean = daily_ic.mean()
        icir    = ic_mean / daily_ic.std() * np.sqrt(TRADING_DAYS)
        ic_win  = (daily_ic > 0).mean()
        ic_sum[col] = {'ic_mean': ic_mean, 'icir': icir, 'ic_win': ic_win}
    return ic_sum

def align_direction(panel, cols, train_ic):
    '''Flip factors with negative train IC so all become positively-oriented.
    Direction decided on TRAIN ic only; test reuses it (no look-ahead).'''
    panel = panel.copy()
    for col in cols:
        if train_ic[col] < 0:
            panel[col] = -panel[col]
    return panel

def standardize(panel, cols):
    '''Cross-sectional z-score per day. Per-day op — safe on full panel.'''
    panel = panel.copy()
    panel[cols] = panel.groupby('date')[cols].transform(lambda x: (x - x.mean()) / x.std())
    return panel

def measure_ic(features, next_ret, cols, split_date=SPLIT):
    '''[CONTROLLED ZONE] The ONLY place that touches next_ret.
    Computes signed Rank IC on the TRAIN split only, used to decide factor
    direction. next_ret is merged in temporarily and never leaves this function.
    Returns: (raw_ic {col: signed ic_mean}, ic_stats full dict).'''
    panel = compute_factors(features, factors=FACTORS)
    train = panel[panel['date'] < split_date].copy()
    panel = pd.merge(train, next_ret, on=['date', 'code'])
    ic_stats = compute_ic(panel, cols)
    raw_ic = {col: ic_stats[col]['ic_mean'] for col in cols}
    return raw_ic, ic_stats

def build_signals(features, raw_ic, factors=FACTORS):
    '''[CLEAN ZONE] Build standardized, direction-aligned signals.
    Receives only features + raw_ic; next_ret never enters this function.
    Returns: (signals with aligned+standardized factors, aligned_ic dict).'''
    features = features.drop(columns=['next_ret'], errors='ignore').copy()
    cols = list(factors.keys())
    signals = compute_factors(features, factors)
    signals = align_direction(signals, cols, raw_ic)
    aligned_ic = {col: abs(raw_ic[col]) for col in raw_ic}
    signals = standardize(signals, cols)
    return signals, aligned_ic

# ============================================================
# 4. PORTFOLIO CONSTRUCTOR — scores -> dollar-neutral weights
# ============================================================

def assign_weights(x, col, q=Q, min_stocks=MIN_STOCKS):
    '''One day's cross-section -> dollar-neutral weights.
    Long top (1-q) at +1/N_long, short bottom q at -1/N_short (net zero).
    Returns: weight Series.'''
    x = x.copy()
    if len(x) < min_stocks:
        x['weight'] = 0.0
        return x['weight']
    is_short = x[col] < x[col].quantile(q)
    is_long  = x[col] > x[col].quantile(1 - q)
    n_long  = is_long.sum()
    n_short = is_short.sum()
    x['weight'] = 0.0
    x.loc[is_long, 'weight']  = 1 / n_long
    x.loc[is_short, 'weight'] = -1 / n_short
    return x['weight']

def build_weights(panel, col, q=Q, min_stocks=MIN_STOCKS):
    '''Add a 'weight' column to the panel, assigned per day.'''
    panel = panel.copy()
    panel['weight'] = panel.groupby('date', group_keys=False).apply(
        assign_weights, col=col, q=q, min_stocks=min_stocks)
    return panel

# ============================================================
# 5. EXECUTION ENGINE — tradability filter + renormalize
# ============================================================

def apply_tradability(x, min_stocks=MIN_STOCKS, price_limit=PRICE_LIMIT):
    '''Zero out untradable names (price-limit / missing today_chg), then
    renormalize survivors back to dollar-neutral (+1 / -1). If either leg
    has too few survivors, the whole day goes flat.
    Uses only today_chg (decision-time info) — no look-ahead.'''
    x = x.copy()
    not_tradeable = (x['today_chg'].abs() > price_limit) | (x['today_chg'].isna())
    x.loc[not_tradeable, 'weight'] = 0.0
    is_long  = x['weight'] > 0
    is_short = x['weight'] < 0
    if is_long.sum() < min_stocks or is_short.sum() < min_stocks:
        x['weight'] = 0.0
        return x
    x.loc[is_long, 'weight']  /= x.loc[is_long, 'weight'].sum()
    x.loc[is_short, 'weight'] /= abs(x.loc[is_short, 'weight'].sum())
    return x
    
# ============================================================
# 6. EVALUATOR — backtest metrics
# ============================================================

def calc_sharpe_max_dd(backtest, cost_rate=COST_RATE, trading_days=TRADING_DAYS):
    '''Backtest metrics on a settled panel (must contain 'weight' and 'next_ret').
    Gross daily return = sum(weight * next_ret) per day.
    Cost = cost_rate * turnover, turnover[t] = sum|w[t] - w[t-1]| per stock.
    Returns: annualized Sharpe and max drawdown.'''
    backtest = backtest.copy()
    backtest = backtest.sort_values(['code', 'date'])
    daily_ret = backtest.groupby('date').apply(lambda x: (x['weight'] * x['next_ret']).sum())
    backtest['prev_weight'] = backtest.groupby('code')['weight'].shift(1).fillna(0)
    backtest['weight_chg']  = (backtest['weight'] - backtest['prev_weight']).abs()
    turnover = backtest.groupby('date')['weight_chg'].sum()
    daily_net_ret = daily_ret - turnover * cost_rate
    sharpe = daily_net_ret.mean() / daily_net_ret.std() * np.sqrt(trading_days)
    acc = (1 + daily_net_ret).cumprod()
    max_dd = (acc / acc.cummax() - 1).min()
    return {'sharpe': sharpe, 'max_dd': max_dd}
    
# ============================================================
# 7. TEST
# ============================================================

def test_weights(weights):
    '''Weight invariants: dollar-neutral, legs sum to +/-1, no NaN.'''
    net = weights.groupby('date')['weight'].sum().abs().max()
    assert net < 1e-10, f"net exposure too large: {net}"
    long_sum  = weights[weights['weight'] > 0].groupby('date')['weight'].sum()
    short_sum = weights[weights['weight'] < 0].groupby('date')['weight'].sum()
    assert np.allclose(long_sum, 1),  "long leg should sum to +1"
    assert np.allclose(short_sum, -1), "short leg should sum to -1"
    assert weights['weight'].isna().sum() == 0, "weight has NaN"
    print("[1/5] weight invariants passed")


def test_no_lookahead(signals, raw_ic, aligned_ic):
    '''Look-ahead guards: no next_ret leak, alignment correct, aligned == |raw|.'''
    assert 'next_ret' not in signals.columns, "next_ret leaked into signal pipeline"
    assert all(v >= 0 for v in aligned_ic.values()), "aligned IC must be non-negative"
    assert all(np.isclose(aligned_ic[c], abs(raw_ic[c])) for c in aligned_ic), \
        "aligned IC must equal |raw IC|"
    print("[2/5] look-ahead guards passed")


def test_data_quality(features, next_ret, weights, backtest):
    '''Data quality: no duplicate keys, merge does not inflate rows, code is 6-digit.'''
    assert features.duplicated(subset=['date', 'code']).sum() == 0, "duplicate (date, code)"
    assert len(weights) == len(backtest), "merge inflated row count (one-to-many)"
    assert features['code'].str.len().eq(6).all(), "code is not 6-digit"
    print("[3/5] data quality passed")


def test_consistency(features, next_ret, cols, split_date=SPLIT):
    '''Determinism: same input yields identical IC across runs.'''
    raw_ic_1 = measure_ic(features, next_ret, cols, split_date)
    raw_ic_2 = measure_ic(features, next_ret, cols, split_date)
    assert raw_ic_1 == raw_ic_2, "measure_ic is not deterministic"
    print("[4/5] determinism passed")


def test_edge_cases():
    '''Degenerate cross-section: too few names should flatten the day, not crash.'''
    tiny = pd.DataFrame({
        'date': pd.to_datetime(['2020-01-01'] * 3),
        'code': ['000001', '000002', '000003'],
        'weight': [0.5, -0.3, -0.2],
        'today_chg': [0.01, 0.01, 0.01],
    })
    out = apply_tradability(tiny, min_stocks=10)
    assert (out['weight'] == 0).all(), "too-few-names day should be flat"
    print("[5/5] edge cases passed")
    
# ============ Backtest ============
def run_backtest(test_signals, score_col, next_ret):
    """Portfolio + execution + settlement. Returns (weights, backtest)."""
    weights = build_weights(test_signals, col=score_col)
    weights['weight'] = weights.groupby('date', group_keys=False).apply(
        lambda g: apply_tradability(g)['weight'])
    backtest = weights.merge(next_ret, on=['date', 'code'])
    return weights, backtest

# ============ Tests ============
def run_all_tests(features, next_ret, cols, signals, raw_ic, aligned_ic, weights, backtest):
    """Run the full test suite."""
    test_weights(weights)
    test_no_lookahead(signals, raw_ic, aligned_ic)
    test_data_quality(features, next_ret, weights, backtest)
    test_consistency(features, next_ret, cols)
    test_edge_cases()
    print("All tests passed ✓")

# ============ Main ============
if __name__ == "__main__":
    cols = list(FACTORS.keys())
    features, next_ret = load_data()
    raw_ic, ic_stats = measure_ic(features, next_ret, cols, SPLIT) 
    print("raw_ic:", raw_ic)
    print("ic_stats:", ic_stats)   
    signals, aligned_ic = build_signals(features, raw_ic)
    print("aligned_ic (|ic|):", aligned_ic)

    w = {c: aligned_ic[c] / sum(aligned_ic.values()) for c in cols}
    signals['composite_icw'] = sum(w[c] * signals[c] for c in cols)
    signals = signals.dropna(subset=['composite_icw'])
    test_signals = signals[signals['date'] >= SPLIT].copy()

    weights, backtest = run_backtest(test_signals, 'composite_icw', next_ret)
    print("OOS result:", calc_sharpe_max_dd(backtest))

    for cost in COST_GRID:
        r = calc_sharpe_max_dd(backtest, cost_rate=cost)
        print(f"cost={cost}: sharpe={r['sharpe']:.4f}, max_dd={r['max_dd']:.2%}")
    windows = [20, 60, 120, 250, 400, 600]
    results = {}
    for w in windows:
        f = features.copy()
        f['mom'] = f.groupby('code')['close'].pct_change(w)
        train = f[f['date'] < SPLIT]
        merged = train.merge(next_ret, on=['date','code'])
        ic = compute_ic(merged, ['mom'])['mom']['ic_mean']
        results[w] = ic
        print(f"window={w}: IC={ic:.4f}")
    run_all_tests(features, next_ret, cols, signals, raw_ic, aligned_ic, weights, backtest)
    