"""
Monte Carlo simulation for the VCP backtest.

run_monte_carlo() shuffles the historical trade returns 1000 times and
builds a distribution of outcomes — giving a robust view of the strategy's
range of possible results beyond the single historical path.
"""

import numpy as np


def run_monte_carlo(trades, initial_capital, n_simulations=1000):
    """
    Simulate n_simulations random orderings of the historical trade returns.

    Args:
        trades:          list of trade dicts with "Net_PL_Pct" field
        initial_capital: float — starting capital
        n_simulations:   int — number of Monte Carlo paths (default 1000)

    Returns dict with:
        paths           — list of n_simulations final portfolio values
        final_values    — sorted list of final portfolio values
        max_drawdowns   — list of worst drawdowns per simulation
        pct_5th … 95th  — percentile final values
        prob_profit     — % of sims ending above initial_capital
        prob_double     — % of sims ending above 2× initial_capital
        median_max_dd   — median worst drawdown across simulations
        worst_case_dd   — 95th percentile worst drawdown
        fan_data        — [{trade_num, p5, p25, p50, p75, p95}] for fan chart
    """
    if not trades:
        return _empty_mc(initial_capital)

    returns = [t["Net_PL_Pct"] / 100 for t in trades]  # fractional returns

    if len(returns) < 2:
        return _empty_mc(initial_capital)

    n_trades       = len(returns)
    final_values   = []
    max_drawdowns  = []
    all_paths      = np.zeros((n_simulations, n_trades))

    rng = np.random.default_rng(seed=42)

    for sim_i in range(n_simulations):
        shuffled = rng.permutation(returns)
        equity   = initial_capital
        path     = []
        peak     = equity
        worst_dd = 0.0

        for ret in shuffled:
            equity *= (1 + ret)
            path.append(equity)
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            if dd > worst_dd:
                worst_dd = dd

        final_values.append(equity)
        max_drawdowns.append(worst_dd)
        all_paths[sim_i] = path

    final_arr = np.array(final_values)
    dd_arr    = np.array(max_drawdowns)

    # Percentile paths for the fan chart (one value per trade number)
    fan_data = []
    for t_idx in range(n_trades):
        col = all_paths[:, t_idx]
        fan_data.append({
            "trade_num": t_idx + 1,
            "p5":        round(float(np.percentile(col, 5)), 2),
            "p25":       round(float(np.percentile(col, 25)), 2),
            "p50":       round(float(np.percentile(col, 50)), 2),
            "p75":       round(float(np.percentile(col, 75)), 2),
            "p95":       round(float(np.percentile(col, 95)), 2),
        })

    prob_profit = float(np.mean(final_arr > initial_capital) * 100)
    prob_double = float(np.mean(final_arr > 2 * initial_capital) * 100)

    return {
        "n_simulations":      n_simulations,
        "n_trades":           n_trades,
        "pct_5th":            round(float(np.percentile(final_arr, 5)), 2),
        "pct_25th":           round(float(np.percentile(final_arr, 25)), 2),
        "pct_50th":           round(float(np.percentile(final_arr, 50)), 2),
        "pct_75th":           round(float(np.percentile(final_arr, 75)), 2),
        "pct_95th":           round(float(np.percentile(final_arr, 95)), 2),
        "prob_profit":        round(prob_profit, 1),
        "prob_double":        round(prob_double, 1),
        "median_max_drawdown":round(float(np.median(dd_arr)), 2),
        "worst_case_drawdown":round(float(np.percentile(dd_arr, 95)), 2),
        "fan_data":           fan_data,
    }


def _empty_mc(initial_capital):
    return {
        "n_simulations":      0,
        "n_trades":           0,
        "pct_5th":            initial_capital,
        "pct_25th":           initial_capital,
        "pct_50th":           initial_capital,
        "pct_75th":           initial_capital,
        "pct_95th":           initial_capital,
        "prob_profit":        0.0,
        "prob_double":        0.0,
        "median_max_drawdown":0.0,
        "worst_case_drawdown":0.0,
        "fan_data":           [],
    }
