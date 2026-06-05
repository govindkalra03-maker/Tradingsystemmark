"""
VCP (Volatility Contraction Pattern) Scanner — Mark Minervini

A VCP is a series of progressively tighter price contractions within an
uptrending base.  Each swing from high to low must be smaller than the
previous, volume must dry up, and the pivot (last swing high) is the
breakout entry point.
"""

import numpy as np
import pandas as pd

from indicators import calculate_indicators
from risk import calculate_position

# ── Parameters ────────────────────────────────────────────────────────────────
LOOKBACK_DAYS     = 90    # trading days used for pattern detection
SWING_WINDOW      = 5     # bars each side needed to qualify as swing point
MIN_CONTRACTIONS  = 2     # minimum valid contractions required
CONTRACTION_TIGHT = 0.82  # each contraction must be ≤ 82 % of the previous
MAX_FINAL_RANGE   = 0.25  # final contraction high-to-low ≤ 25 %
MAX_PIVOT_DIST    = 0.10  # price must sit within 10 % below the pivot
MIN_SCORE         = 40    # minimum quality score to report a signal


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sq(obj):
    return obj.squeeze() if isinstance(obj, pd.DataFrame) else obj


# ── Swing detection ───────────────────────────────────────────────────────────

def find_swing_points(high_arr: np.ndarray, low_arr: np.ndarray, n: int = 5) -> list:
    """
    Identify alternating swing highs and lows.

    A swing high at bar i: high[i] is the maximum in high[i-n : i+n+1].
    A swing low  at bar i: low[i]  is the minimum in  low[i-n : i+n+1].

    Returns list of ('H'|'L', bar_index, price), deduplicated so types
    strictly alternate, keeping the more extreme price when deduplicating.
    """
    raw = []
    size = len(high_arr)
    for i in range(n, size - n):
        wh = high_arr[i - n: i + n + 1]
        wl = low_arr[i - n: i + n + 1]
        if high_arr[i] >= wh.max() - 1e-9:
            raw.append(('H', i, float(high_arr[i])))
        if low_arr[i] <= wl.min() + 1e-9:
            raw.append(('L', i, float(low_arr[i])))

    # Sort by bar index (handles edge case where H and L share same index)
    raw.sort(key=lambda x: x[1])

    # Deduplicate: keep alternating H/L, preserving the more extreme price
    alt = []
    for pt in raw:
        if not alt:
            alt.append(pt)
        elif pt[0] == alt[-1][0]:
            if (pt[0] == 'H' and pt[2] >= alt[-1][2]) or \
               (pt[0] == 'L' and pt[2] <= alt[-1][2]):
                alt[-1] = pt
        else:
            alt.append(pt)

    return alt


# ── Contraction extraction ────────────────────────────────────────────────────

def extract_contractions(swings: list) -> list:
    """
    From the alternating swing sequence extract consecutive H→L contractions.

    Each contraction is the % decline from a swing high to the very next
    swing low.
    """
    contractions = []
    for i in range(len(swings) - 1):
        if swings[i][0] == 'H' and swings[i + 1][0] == 'L':
            h_p = swings[i][2]
            l_p = swings[i + 1][2]
            if h_p > l_p:
                contractions.append({
                    'h_idx':     swings[i][1],
                    'h_price':   h_p,
                    'l_idx':     swings[i + 1][1],
                    'l_price':   l_p,
                    'range_pct': (h_p - l_p) / h_p,
                })
    return contractions


# ── VCP sequence validation ───────────────────────────────────────────────────

def find_valid_vcp_sequence(contractions: list) -> list:
    """
    Find the most recent valid VCP sequence where each contraction is
    progressively tighter (≤ CONTRACTION_TIGHT × previous range) and the
    final contraction is within MAX_FINAL_RANGE.

    Returns the best valid sequence (list of contraction dicts) or [].
    """
    if len(contractions) < MIN_CONTRACTIONS:
        return []

    n = len(contractions)
    valid_seqs = []

    for start in range(n):
        seq = [contractions[start]]
        for k in range(start + 1, n):
            if contractions[k]['range_pct'] <= seq[-1]['range_pct'] * CONTRACTION_TIGHT:
                seq.append(contractions[k])
        if (len(seq) >= MIN_CONTRACTIONS and
                seq[-1]['range_pct'] <= MAX_FINAL_RANGE):
            valid_seqs.append(seq)

    if not valid_seqs:
        return []

    # Prefer: most recent ending bar, then longest sequence
    valid_seqs.sort(key=lambda s: (s[-1]['l_idx'], len(s)), reverse=True)
    return valid_seqs[0]


# ── Volume analysis ───────────────────────────────────────────────────────────

def vol_is_declining(df_slice: pd.DataFrame) -> bool:
    """True if average volume of the last 10 bars < 88 % of prior 10 bars."""
    vol = _sq(df_slice['Volume']).astype(float)
    if len(vol) < 20:
        return False
    recent = float(vol.iloc[-10:].mean())
    prior  = float(vol.iloc[-20:-10].mean())
    return recent < prior * 0.88 if prior > 0 else False


# ── Quality scoring ───────────────────────────────────────────────────────────

def vcp_score(seq: list, ind: dict, vol_declining: bool) -> int:
    """Score a VCP setup 0–100."""
    s = 0
    n = len(seq)

    # ① Contraction count (max 20 pts)
    s += min(n * 7, 20)

    # ② Overall tightening (max 25 pts): ratio = final / first range
    ratio = seq[-1]['range_pct'] / seq[0]['range_pct']
    s += (25 if ratio < 0.25 else
          20 if ratio < 0.40 else
          15 if ratio < 0.55 else
          10 if ratio < 0.70 else 5)

    # ③ Final contraction tightness (max 20 pts)
    fr = seq[-1]['range_pct']
    s += (20 if fr < 0.08 else
          16 if fr < 0.12 else
          11 if fr < 0.17 else
           6 if fr < 0.22 else 2)

    # ④ SMA stack (max 20 pts)
    p, s50, s150, s200 = (ind['Last_Close'], ind['SMA_50'],
                          ind['SMA_150'],    ind['SMA_200'])
    if   p > s50 > s150 > s200: s += 20
    elif p > s50 and p > s200:  s += 12
    elif p > s200:              s +=  6

    # ⑤ Volume dry-up (max 15 pts)
    if vol_declining:
        s += 15

    return min(s, 100)


# ── Per-ticker VCP scan ───────────────────────────────────────────────────────

def scan_ticker_vcp(ticker: str, df: pd.DataFrame, bench_ret: float) -> dict | None:
    """
    Scan one ticker for a VCP setup.

    Returns a result dict on success (includes '_seq' key for chart generation
    — strip before sending to the frontend) or None if no valid VCP found.
    """
    if df is None or len(df) < 260:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    ind = calculate_indicators(df)
    if ind is None:
        return None

    price = ind['Last_Close']

    # ── Uptrend filter ────────────────────────────────────────────────────────
    if price < ind['SMA_200']:
        return None
    if ind['SMA_200_21d_ago'] and ind['SMA_200'] < ind['SMA_200_21d_ago']:
        return None

    # ── VCP pattern detection ─────────────────────────────────────────────────
    df_sl    = df.tail(LOOKBACK_DAYS).copy()
    high_arr = _sq(df_sl['High']).astype(float).values
    low_arr  = _sq(df_sl['Low']).astype(float).values

    swings       = find_swing_points(high_arr, low_arr, n=SWING_WINDOW)
    contractions = extract_contractions(swings)
    seq          = find_valid_vcp_sequence(contractions)

    if not seq:
        return None

    # ── Score and thresholds ──────────────────────────────────────────────────
    vol_dec = vol_is_declining(df_sl)
    score   = vcp_score(seq, ind, vol_dec)

    if score < MIN_SCORE:
        return None

    pivot = seq[-1]['h_price']
    stop  = seq[-1]['l_price']

    # Price must be near pivot — not blown past it or already failed
    if price > pivot * (1 + MAX_PIVOT_DIST) or price < stop:
        return None

    rk = calculate_position(pivot, stop)
    if rk is None:
        return None

    tightening = round(
        (1 - seq[-1]['range_pct'] / seq[0]['range_pct']) * 100, 1
    )
    rs = (round((ind['Return_12m'] - bench_ret) * 100, 2)
          if ind.get('Return_12m') is not None else None)

    return {
        'Ticker':          ticker.replace('.NS', ''),
        'TickerFull':      ticker,
        'Last_Close':      round(price, 2),
        'VCP_Score':       score,
        'N_Contractions':  len(seq),
        'Tightening_Pct':  tightening,
        'Pivot':           round(pivot, 2),
        'Stop_Loss':       round(stop, 2),
        'Stop_Pct':        rk['stop_pct'],
        'RR_Ratio':        rk['rr_ratio'],
        'Target_1':        rk['target_1'],
        'Target_2':        rk['target_2'],
        'Target_3':        rk['target_3'],
        'Shares_5L':       rk['shares'],
        'Capital_Risk_5L': rk['capital_at_risk'],
        'Vol_Declining':   vol_dec,
        'SMA_50':          round(ind['SMA_50'], 2),
        'SMA_150':         round(ind['SMA_150'], 2),
        'SMA_200':         round(ind['SMA_200'], 2),
        'RS_vs_Nifty':     rs,
        '_seq':            seq,   # used only for chart generation — not sent to UI
    }
