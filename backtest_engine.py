"""
VCP Backtesting Engine — day-by-day simulation with no look-ahead bias.

Usage:
    bt = VCPBacktester(params)
    bt.load_data(tickers)
    bt.run(tickers)
    print(bt.all_trades)
"""

import math
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

from indicators import calculate_indicators
from vcp_scanner import find_swing_points, extract_contractions, vol_is_declining, vcp_score

BENCHMARK_TICKER  = "^CRSLDX"
CONTRACTION_TIGHT = 0.82
MAX_FINAL_RANGE   = 0.25
MAX_PIVOT_DIST    = 0.10

DEFAULT_PARAMS = {
    "start_date":          "2019-01-01",
    "end_date":            "2024-12-31",
    "initial_capital":     500_000.0,
    "risk_per_trade_pct":  0.01,
    "max_open_positions":  5,
    "slippage_pct":        0.001,
    "brokerage_per_trade": 40.0,
    "stop_loss_pct":       0.08,
    "target_1_pct":        0.20,
    "target_2_pct":        0.35,
    "target_3_pct":        0.50,
    "partial_exit_1_pct":  0.33,
    "partial_exit_2_pct":  0.33,
    "vcp_min_score":       40,
    "min_contractions":    2,
    "lookback_days":       90,
    "time_stop_days":      21,
    "use_market_filter":   True,
    "include_stt":         False,
}


def _sq(x):
    return x.squeeze() if isinstance(x, pd.DataFrame) else x


def _safe_float(v):
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except Exception:
        return None


class VCPBacktester:
    """
    Simulates the VCP strategy day by day over a historical date range.

    CRITICAL — no look-ahead bias:
      On any day D, only data up to and including D is ever used to make
      decisions.  Data slices are always df.loc[:D].
    """

    def __init__(self, params=None):
        self.p            = {**DEFAULT_PARAMS, **(params or {})}
        self.data         = {}    # ticker -> DataFrame (reindexed to benchmark days)
        self.benchmark    = None  # full benchmark DataFrame
        self.all_trades   = []
        self.daily_values = []    # list of {"date", "portfolio", "benchmark"}
        self.progress     = {"pct": 0, "msg": "Not started", "done": False, "error": None}

    # ─── Data loading ─────────────────────────────────────────────────────────

    def load_data(self, tickers):
        """
        Download OHLCV for tickers and the Nifty 500 benchmark.
        Fetches from (start_date − 2 years) so indicators have enough warm-up history.
        Reindexes every stock to the benchmark trading-day calendar (forward-fill)
        so index lookups with .at[day, col] never raise KeyError.
        Returns True on success.
        """
        p = self.p
        start = pd.Timestamp(p["start_date"])
        end   = pd.Timestamp(p["end_date"])
        # Two-year warm-up before the backtest window
        dl_start = (start - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
        dl_end   = end.strftime("%Y-%m-%d")

        self.progress["msg"] = "Downloading benchmark…"
        try:
            df_b = yf.download(
                BENCHMARK_TICKER, start=dl_start, end=dl_end,
                auto_adjust=True, progress=False,
            )
            if isinstance(df_b.columns, pd.MultiIndex):
                df_b.columns = df_b.columns.get_level_values(0)
            if df_b.empty:
                self.progress["error"] = "Benchmark download returned empty data"
                return False
            self.benchmark = df_b
        except Exception as e:
            self.progress["error"] = f"Benchmark download failed: {e}"
            return False

        bench_days = self.benchmark.index

        for i, ticker in enumerate(tickers):
            self.progress["pct"] = int((i + 1) / len(tickers) * 25)
            self.progress["msg"] = f"Downloading {ticker} ({i+1}/{len(tickers)})"
            try:
                df = yf.download(
                    ticker, start=dl_start, end=dl_end,
                    auto_adjust=True, progress=False,
                )
                if df is None or df.empty or len(df) < 100:
                    continue
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                # Align to benchmark calendar — ffill so holiday gaps are filled
                # with the last real price.  This does NOT create look-ahead bias
                # because we always only look at data up to day D.
                df = df.reindex(bench_days, method="ffill")
                self.data[ticker] = df
            except Exception:
                continue

        return bool(self.data)

    # ─── VCP detection (parameterised) ────────────────────────────────────────

    def _find_valid_seq(self, contractions):
        """Parameterised version of find_valid_vcp_sequence using instance p."""
        min_c = self.p["min_contractions"]
        n = len(contractions)
        if n < min_c:
            return []
        valid = []
        for start_i in range(n):
            seq = [contractions[start_i]]
            for k in range(start_i + 1, n):
                if contractions[k]["range_pct"] <= seq[-1]["range_pct"] * CONTRACTION_TIGHT:
                    seq.append(contractions[k])
            if len(seq) >= min_c and seq[-1]["range_pct"] <= MAX_FINAL_RANGE:
                valid.append(seq)
        if not valid:
            return []
        valid.sort(key=lambda s: (s[-1]["l_idx"], len(s)), reverse=True)
        return valid[0]

    def _detect_vcp(self, df_slice):
        """
        Run VCP detection on df_slice — which must already be sliced to day D
        so there is zero look-ahead bias.
        Returns (score, pivot, stop) or None.
        """
        if len(df_slice) < 260:
            return None

        ind = calculate_indicators(df_slice)
        if ind is None:
            return None

        price = ind["Last_Close"]

        # Uptrend filter: price above 200 SMA and SMA rising
        if price < ind["SMA_200"]:
            return None
        sma_ago = ind.get("SMA_200_21d_ago")
        if sma_ago and ind["SMA_200"] < sma_ago:
            return None

        lb = self.p["lookback_days"]
        df_lb = df_slice.tail(lb)

        try:
            high_arr = _sq(df_lb["High"]).astype(float).values
            low_arr  = _sq(df_lb["Low"]).astype(float).values
        except Exception:
            return None

        swings       = find_swing_points(high_arr, low_arr, n=5)
        contractions = extract_contractions(swings)
        seq          = self._find_valid_seq(contractions)

        if not seq:
            return None

        vol_dec = vol_is_declining(df_lb)
        score   = vcp_score(seq, ind, vol_dec)

        if score < self.p["vcp_min_score"]:
            return None

        pivot = seq[-1]["h_price"]
        stop  = seq[-1]["l_price"]

        if price > pivot * (1 + MAX_PIVOT_DIST) or price < stop:
            return None

        return score, pivot, stop

    # ─── Main simulation loop ─────────────────────────────────────────────────

    def run(self, tickers):
        """
        Day-by-day backtest simulation.

        Steps each trading day D:
          1. Update market filter (benchmark vs 200-SMA)
          2. Update open positions (stop, targets, time stop, trailing stop)
          3. Generate new signals (VCP breakouts)
          4. Record portfolio value
        """
        p = self.p

        if not self.data:
            if not self.load_data(tickers):
                self.progress.update({"done": True, "error": "No data downloaded"})
                return

        start_ts = pd.Timestamp(p["start_date"])
        end_ts   = pd.Timestamp(p["end_date"])

        bench_close  = _sq(self.benchmark["Close"]).astype(float)
        bench_sma200 = bench_close.rolling(200).mean()

        trading_days = bench_close.loc[start_ts:end_ts].index
        if len(trading_days) == 0:
            self.progress.update({"done": True, "error": "No trading days in date range"})
            return

        # Scale benchmark to same starting capital for equity curve display
        bench_start_raw = _safe_float(bench_close.loc[start_ts:].iloc[0]) or 1.0

        capital        = float(p["initial_capital"])
        open_positions = []
        all_trades     = []
        daily_values   = []
        total_days     = len(trading_days)

        for day_i, day in enumerate(trading_days):
            # Progress updates every 50 days to avoid lock contention
            if day_i % 50 == 0:
                self.progress["pct"] = 25 + int(day_i / total_days * 70)
                self.progress["msg"] = (
                    f"Day {day_i+1}/{total_days} ({day.date()}) "
                    f"— {len(open_positions)} open, {len(all_trades)} closed"
                )

            # ── STEP 1: Market filter ──────────────────────────────────────
            market_ok = True
            if p["use_market_filter"]:
                try:
                    bc = _safe_float(bench_close.at[day])
                    bs = _safe_float(bench_sma200.at[day])
                    if bc is not None and bs is not None:
                        market_ok = bc >= bs
                except Exception:
                    pass

            # ── STEP 2: Update open positions ──────────────────────────────
            still_open = []
            for pos in open_positions:
                ticker = pos["ticker"]
                df_t   = self.data.get(ticker)

                if df_t is None or day not in df_t.index:
                    still_open.append(pos)
                    continue

                o  = _safe_float(df_t.at[day, "Open"])
                h  = _safe_float(df_t.at[day, "High"])
                lo = _safe_float(df_t.at[day, "Low"])
                c  = _safe_float(df_t.at[day, "Close"])

                if None in (o, h, lo, c):
                    still_open.append(pos)
                    continue

                pos["daily_lows"].append(lo)
                pos["highest_close"] = max(pos["highest_close"], c)

                closed = False

                # Stop loss — handle gap-down opens
                if lo <= pos["stop_loss"]:
                    exit_px = min(pos["stop_loss"], o) * (1 - p["slippage_pct"])
                    cost = self._exit_cost(pos["shares_remaining"], exit_px, p)
                    capital += pos["shares_remaining"] * exit_px - cost
                    all_trades.append(self._close_pos(pos, day, exit_px, "stopped_out", cost))
                    closed = True

                # Target 3 (only after T2 hit)
                elif pos["t2_hit"] and h >= pos["target_3"]:
                    exit_px = pos["target_3"] * (1 - p["slippage_pct"])
                    cost = self._exit_cost(pos["shares_remaining"], exit_px, p)
                    capital += pos["shares_remaining"] * exit_px - cost
                    all_trades.append(self._close_pos(pos, day, exit_px, "target_3", cost))
                    closed = True

                else:
                    # Target 2 (partial sell after T1 hit)
                    if pos["t1_hit"] and not pos["t2_hit"] and h >= pos["target_2"]:
                        exit_px = pos["target_2"] * (1 - p["slippage_pct"])
                        n_sell = max(1, min(
                            round(pos["shares_orig"] * p["partial_exit_2_pct"]),
                            pos["shares_remaining"] - 1,
                        ))
                        cost = self._exit_cost(n_sell, exit_px, p)
                        capital += n_sell * exit_px - cost
                        pos["shares_remaining"] -= n_sell
                        pos["partial_exits"].append((n_sell, exit_px, "target_2"))
                        pos["t2_hit"]    = True
                        pos["stop_loss"] = pos["target_1"]  # trail stop to T1

                    # Target 1 (partial sell)
                    if not pos["t1_hit"] and h >= pos["target_1"]:
                        exit_px = pos["target_1"] * (1 - p["slippage_pct"])
                        n_sell = max(1, min(
                            round(pos["shares_orig"] * p["partial_exit_1_pct"]),
                            pos["shares_remaining"] - 1,
                        ))
                        cost = self._exit_cost(n_sell, exit_px, p)
                        capital += n_sell * exit_px - cost
                        pos["shares_remaining"] -= n_sell
                        pos["partial_exits"].append((n_sell, exit_px, "target_1"))
                        pos["t1_hit"]    = True
                        pos["stop_loss"] = pos["entry_price"]  # trail to breakeven

                    # Time stop: held > time_stop_days calendar days with no progress
                    days_cal = (day - pos["entry_date"]).days
                    if days_cal > int(p["time_stop_days"] * 1.5):
                        if pos["highest_close"] <= pos["entry_price"] * 1.02:
                            exit_px = c * (1 - p["slippage_pct"])
                            cost = self._exit_cost(pos["shares_remaining"], exit_px, p)
                            capital += pos["shares_remaining"] * exit_px - cost
                            all_trades.append(self._close_pos(pos, day, exit_px, "time_stop", cost))
                            closed = True

                    # Trailing stop on 21-EMA after T1 hit
                    if not closed and pos["t1_hit"]:
                        try:
                            close_s = _sq(df_t["Close"]).astype(float).loc[:day]
                            ema21 = float(close_s.ewm(span=21, adjust=False).mean().iloc[-1])
                            if c < ema21:
                                exit_px = c * (1 - p["slippage_pct"])
                                cost = self._exit_cost(pos["shares_remaining"], exit_px, p)
                                capital += pos["shares_remaining"] * exit_px - cost
                                all_trades.append(
                                    self._close_pos(pos, day, exit_px, "trailing_stop", cost)
                                )
                                closed = True
                        except Exception:
                            pass

                if not closed:
                    still_open.append(pos)

            open_positions = still_open

            # ── STEP 3: New signals (only when slots open and market is ok) ──
            if market_ok and len(open_positions) < p["max_open_positions"]:
                open_tickers = {pos["ticker"] for pos in open_positions}

                for ticker in tickers:
                    if len(open_positions) >= p["max_open_positions"]:
                        break
                    if ticker in open_tickers or ticker not in self.data:
                        continue

                    df_t = self.data[ticker]
                    if day not in df_t.index:
                        continue

                    try:
                        # Slice data strictly up to day D — the core no-look-ahead guard
                        df_slice = df_t.loc[:day]
                        if len(df_slice) < 260:
                            continue

                        result = self._detect_vcp(df_slice)
                        if result is None:
                            continue

                        score, pivot, stop = result
                        entry_px = pivot * 1.01  # entry slightly above pivot

                        day_h = _safe_float(df_t.at[day, "High"])
                        if day_h is None or day_h < entry_px:
                            continue

                        # Position sizing: risk pct of current equity
                        actual_entry = entry_px * (1 + p["slippage_pct"])
                        risk_per_sh  = actual_entry - stop
                        if risk_per_sh <= 0:
                            continue

                        shares = max(1, int((capital * p["risk_per_trade_pct"]) / risk_per_sh))

                        total_cost = shares * actual_entry + p["brokerage_per_trade"]
                        # Cap at 95% of available capital per position
                        if total_cost > capital * 0.95:
                            shares = max(1, int(
                                (capital * 0.95 - p["brokerage_per_trade"]) / actual_entry
                            ))
                            total_cost = shares * actual_entry + p["brokerage_per_trade"]

                        if total_cost > capital or shares <= 0:
                            continue

                        capital -= total_cost

                        open_positions.append({
                            "ticker":           ticker,
                            "entry_date":       day,
                            "entry_price":      actual_entry,
                            "shares_orig":      shares,
                            "shares_remaining": shares,
                            "stop_loss":        stop,
                            "target_1":         actual_entry * (1 + p["target_1_pct"]),
                            "target_2":         actual_entry * (1 + p["target_2_pct"]),
                            "target_3":         actual_entry * (1 + p["target_3_pct"]),
                            "t1_hit":           False,
                            "t2_hit":           False,
                            "vcp_score":        score,
                            "highest_close":    actual_entry,
                            "daily_lows":       [],
                            "partial_exits":    [],  # (shares, price, type)
                        })
                        open_tickers.add(ticker)

                    except Exception:
                        continue

            # ── STEP 4: Record daily portfolio value ───────────────────────
            open_val = 0.0
            for pos in open_positions:
                try:
                    px = _safe_float(self.data[pos["ticker"]].at[day, "Close"])
                    open_val += pos["shares_remaining"] * (px or pos["entry_price"])
                except Exception:
                    open_val += pos["shares_remaining"] * pos["entry_price"]

            bench_raw = _safe_float(bench_close.at[day]) or bench_start_raw
            bench_val = (bench_raw / bench_start_raw) * p["initial_capital"]

            daily_values.append({
                "date":      day.strftime("%Y-%m-%d"),
                "portfolio": round(capital + open_val, 2),
                "benchmark": round(bench_val, 2),
            })

        # Close any remaining positions at the last day of the backtest period
        last_day = trading_days[-1]
        for pos in open_positions:
            df_t = self.data.get(pos["ticker"])
            try:
                exit_px = (_safe_float(df_t.at[last_day, "Close"]) or pos["entry_price"])
                exit_px *= (1 - p["slippage_pct"])
            except Exception:
                exit_px = pos["entry_price"]

            cost = self._exit_cost(pos["shares_remaining"], exit_px, p)
            capital += pos["shares_remaining"] * exit_px - cost
            all_trades.append(self._close_pos(pos, last_day, exit_px, "end_of_backtest", cost))

        self.all_trades   = all_trades
        self.daily_values = daily_values

        final_val = daily_values[-1]["portfolio"] if daily_values else p["initial_capital"]
        self.progress.update({
            "pct": 100,
            "msg": (
                f"Complete — {len(all_trades)} trades "
                f"| Final ₹{final_val:,.0f}"
            ),
            "done": True,
        })

    # ─── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _exit_cost(shares, price, p):
        """Brokerage + optional STT on exit."""
        stt = price * shares * 0.001 if p["include_stt"] else 0.0
        return p["brokerage_per_trade"] + stt

    def _close_pos(self, pos, exit_date, exit_price, exit_type, costs):
        """Build a trade dict for a fully closed position (all partial exits included)."""
        total_proceeds = sum(n * px for n, px, _ in pos["partial_exits"])
        total_proceeds += pos["shares_remaining"] * exit_price

        gross_cost = pos["shares_orig"] * pos["entry_price"]
        gross_pl   = total_proceeds - gross_cost
        net_pl     = gross_pl - costs
        net_pl_pct = net_pl / gross_cost * 100 if gross_cost > 0 else 0.0

        mae_pct = 0.0
        if pos["daily_lows"]:
            worst = min(pos["daily_lows"])
            mae_pct = max(0.0, (pos["entry_price"] - worst) / pos["entry_price"] * 100)

        return {
            "Ticker":      pos["ticker"].replace(".NS", ""),
            "Entry_Date":  pos["entry_date"].strftime("%Y-%m-%d"),
            "Exit_Date":   exit_date.strftime("%Y-%m-%d"),
            "Hold_Days":   (exit_date - pos["entry_date"]).days,
            "Entry_Price": round(pos["entry_price"], 2),
            "Exit_Price":  round(exit_price, 2),
            "Shares":      pos["shares_orig"],
            "Gross_PL":    round(gross_pl, 2),
            "Costs":       round(costs, 2),
            "Net_PL":      round(net_pl, 2),
            "Net_PL_Pct":  round(net_pl_pct, 2),
            "Exit_Type":   exit_type,
            "VCP_Score":   pos["vcp_score"],
            "MAE_Pct":     round(mae_pct, 2),
        }
