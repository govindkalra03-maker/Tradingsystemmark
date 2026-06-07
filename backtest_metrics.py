"""
Backtest performance metrics.

calculate_metrics()  — full stats dict from trade list + equity curve
generate_trade_log() — clean DataFrame with one row per trade
"""

import math
from collections import Counter

import numpy as np
import pandas as pd

RISK_FREE_RATE = 0.065  # Indian repo rate approximation (6.5%)


def calculate_metrics(trades, daily_values, initial_capital):
    """
    Compute a comprehensive performance metrics dict.

    Args:
        trades:          list of trade dicts from VCPBacktester.all_trades
        daily_values:    list of {"date", "portfolio", "benchmark"} dicts
        initial_capital: float

    Returns dict with return, risk, trade, and monthly breakdown metrics.
    """
    if not daily_values:
        return _empty_metrics()

    # ── Build daily series ────────────────────────────────────────────────────
    dates      = [d["date"] for d in daily_values]
    portfolio  = pd.Series([d["portfolio"] for d in daily_values], index=dates)
    benchmark  = pd.Series([d["benchmark"] for d in daily_values], index=dates)

    final_val   = float(portfolio.iloc[-1])
    bench_final = float(benchmark.iloc[-1])

    # Number of years
    try:
        d0 = pd.Timestamp(dates[0])
        d1 = pd.Timestamp(dates[-1])
        years = max((d1 - d0).days / 365.25, 0.01)
    except Exception:
        years = 1.0

    # ── Return metrics ────────────────────────────────────────────────────────
    total_return   = (final_val - initial_capital) / initial_capital * 100
    cagr           = ((final_val / initial_capital) ** (1 / years) - 1) * 100
    bench_return   = (bench_final - initial_capital) / initial_capital * 100
    bench_cagr     = ((bench_final / initial_capital) ** (1 / years) - 1) * 100
    alpha          = cagr - bench_cagr

    # ── Risk metrics ─────────────────────────────────────────────────────────
    daily_ret = portfolio.pct_change().dropna()
    vol_annual = float(daily_ret.std() * math.sqrt(252) * 100) if len(daily_ret) > 1 else 0.0

    # Sharpe ratio
    rf_daily = RISK_FREE_RATE / 252
    excess   = daily_ret - rf_daily
    sharpe   = (
        float(excess.mean() / excess.std() * math.sqrt(252))
        if excess.std() > 0 else 0.0
    )

    # Maximum drawdown
    running_peak  = portfolio.cummax()
    drawdown_pct  = (portfolio - running_peak) / running_peak * 100
    max_dd        = float(drawdown_pct.min())

    # Calmar ratio
    calmar = abs(cagr / max_dd) if max_dd != 0 else 0.0

    # ── Trade metrics ─────────────────────────────────────────────────────────
    total_trades = len(trades)
    if total_trades == 0:
        return _empty_metrics() | {
            "total_return_pct":    round(total_return, 2),
            "cagr_pct":            round(cagr, 2),
            "benchmark_return_pct":round(bench_return, 2),
            "alpha_pct":           round(alpha, 2),
            "max_drawdown_pct":    round(max_dd, 2),
            "sharpe_ratio":        round(sharpe, 2),
            "calmar_ratio":        round(calmar, 2),
            "volatility_annual":   round(vol_annual, 2),
            "monthly_returns":     _monthly_returns(portfolio),
        }

    returns  = [t["Net_PL_Pct"] for t in trades]
    wins     = [r for r in returns if r > 0]
    losses   = [r for r in returns if r <= 0]

    win_rate      = len(wins) / total_trades * 100
    avg_win       = float(np.mean(wins))   if wins   else 0.0
    avg_loss      = float(np.mean(losses)) if losses else 0.0
    sum_wins      = sum(t["Net_PL"] for t in trades if t["Net_PL"] > 0)
    sum_losses    = abs(sum(t["Net_PL"] for t in trades if t["Net_PL"] <= 0))
    profit_factor = sum_wins / sum_losses if sum_losses > 0 else float("inf")

    loss_rate     = 100 - win_rate
    expectancy    = (win_rate / 100 * avg_win) - (loss_rate / 100 * abs(avg_loss))

    hold_days     = [t["Hold_Days"] for t in trades]
    avg_hold      = float(np.mean(hold_days)) if hold_days else 0.0

    best_trade  = max(returns) if returns else 0.0
    worst_trade = min(returns) if returns else 0.0

    # Consecutive win/loss streaks
    max_cons_wins, max_cons_losses = _streaks(returns)

    exit_counts = Counter(t["Exit_Type"] for t in trades)

    # ── Monthly returns ────────────────────────────────────────────────────────
    monthly = _monthly_returns(portfolio)
    monthly_vals = [v["return_pct"] for v in monthly]
    best_month    = max(monthly_vals) if monthly_vals else 0.0
    worst_month   = min(monthly_vals) if monthly_vals else 0.0
    pos_months    = sum(1 for v in monthly_vals if v > 0)
    pos_month_pct = pos_months / len(monthly_vals) * 100 if monthly_vals else 0.0

    return {
        # Returns
        "total_return_pct":       round(total_return, 2),
        "cagr_pct":               round(cagr, 2),
        "benchmark_return_pct":   round(bench_return, 2),
        "benchmark_cagr_pct":     round(bench_cagr, 2),
        "alpha_pct":              round(alpha, 2),
        # Risk
        "max_drawdown_pct":       round(max_dd, 2),
        "sharpe_ratio":           round(sharpe, 2),
        "calmar_ratio":           round(calmar, 2),
        "volatility_annual":      round(vol_annual, 2),
        # Trades
        "total_trades":           total_trades,
        "win_rate_pct":           round(win_rate, 1),
        "avg_win_pct":            round(avg_win, 2),
        "avg_loss_pct":           round(avg_loss, 2),
        "profit_factor":          round(profit_factor, 2),
        "expectancy_pct":         round(expectancy, 2),
        "avg_hold_days":          round(avg_hold, 1),
        "max_consecutive_wins":   max_cons_wins,
        "max_consecutive_losses": max_cons_losses,
        "best_trade_pct":         round(best_trade, 2),
        "worst_trade_pct":        round(worst_trade, 2),
        "exit_breakdown":         dict(exit_counts),
        # Monthly
        "monthly_returns":        monthly,
        "best_month_pct":         round(best_month, 2),
        "worst_month_pct":        round(worst_month, 2),
        "positive_months_pct":    round(pos_month_pct, 1),
    }


def generate_trade_log(trades):
    """
    Convert the all_trades list to a clean DataFrame.
    Returns empty DataFrame if trades is empty.
    """
    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)

    # Friendly column order
    cols = [
        "Ticker", "Entry_Date", "Exit_Date", "Hold_Days",
        "Entry_Price", "Exit_Price", "Shares",
        "Gross_PL", "Costs", "Net_PL", "Net_PL_Pct",
        "Exit_Type", "VCP_Score", "MAE_Pct",
    ]
    existing = [c for c in cols if c in df.columns]
    return df[existing].reset_index(drop=True)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _empty_metrics():
    return {
        "total_return_pct":       0.0,
        "cagr_pct":               0.0,
        "benchmark_return_pct":   0.0,
        "benchmark_cagr_pct":     0.0,
        "alpha_pct":              0.0,
        "max_drawdown_pct":       0.0,
        "sharpe_ratio":           0.0,
        "calmar_ratio":           0.0,
        "volatility_annual":      0.0,
        "total_trades":           0,
        "win_rate_pct":           0.0,
        "avg_win_pct":            0.0,
        "avg_loss_pct":           0.0,
        "profit_factor":          0.0,
        "expectancy_pct":         0.0,
        "avg_hold_days":          0.0,
        "max_consecutive_wins":   0,
        "max_consecutive_losses": 0,
        "best_trade_pct":         0.0,
        "worst_trade_pct":        0.0,
        "exit_breakdown":         {},
        "monthly_returns":        [],
        "best_month_pct":         0.0,
        "worst_month_pct":        0.0,
        "positive_months_pct":    0.0,
    }


def _streaks(returns):
    """Return (max_consecutive_wins, max_consecutive_losses)."""
    max_w = max_l = cur_w = cur_l = 0
    for r in returns:
        if r > 0:
            cur_w += 1; cur_l = 0
        else:
            cur_l += 1; cur_w = 0
        max_w = max(max_w, cur_w)
        max_l = max(max_l, cur_l)
    return max_w, max_l


def _monthly_returns(portfolio_series):
    """
    Build a list of {"year", "month", "return_pct"} for the heatmap.
    portfolio_series: pandas Series indexed by date strings or Timestamps.
    """
    try:
        s = portfolio_series.copy()
        s.index = pd.to_datetime(s.index)
        monthly = s.resample("ME").last()
        monthly_ret = monthly.pct_change().dropna()

        result = []
        for ts, ret in monthly_ret.items():
            result.append({
                "year":       int(ts.year),
                "month":      int(ts.month),
                "return_pct": round(float(ret) * 100, 2),
            })
        return result
    except Exception:
        return []
