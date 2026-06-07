"""
VCP parameter optimiser — grid search with ProcessPoolExecutor.

optimise_parameters() tests every combination in param_grid, runs a full
backtest for each, and ranks by a chosen metric (default: sharpe_ratio).

NOTE: Each worker process downloads its own data.  For large ticker lists or
large grids this can take tens of minutes.  Cap max_workers to avoid hitting
yfinance rate limits.
"""

import itertools
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from backtest_engine import DEFAULT_PARAMS, VCPBacktester
from backtest_metrics import calculate_metrics

warnings.filterwarnings("ignore")


# ── Worker (module-level so it is picklable on macOS/Windows spawn) ───────────

def _run_combo(args):
    """
    Run one backtest parameter combination.
    Called in a worker process — must be picklable.

    Args:
        args: (tickers, params_dict)

    Returns dict: all params + all metric values, or None on failure.
    """
    tickers, params = args
    try:
        bt = VCPBacktester(params)
        bt.load_data(tickers)
        bt.run(tickers)

        metrics = calculate_metrics(
            bt.all_trades, bt.daily_values, params["initial_capital"]
        )

        # Strip non-serialisable keys (monthly_returns list etc.)
        row = {k: v for k, v in params.items()}
        scalar_metrics = {
            k: v for k, v in metrics.items()
            if isinstance(v, (int, float)) and k != "exit_breakdown"
        }
        row.update(scalar_metrics)
        return row
    except Exception as e:
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def optimise_parameters(
    tickers,
    param_grid,
    base_params=None,
    metric="sharpe_ratio",
    max_workers=4,
    progress_callback=None,
):
    """
    Grid-search parameter combinations and rank by chosen metric.

    Args:
        tickers:           list of ticker strings
        param_grid:        dict of {param_name: [value1, value2, ...]}
                           Example:
                             {
                               "stop_loss_pct":   [0.06, 0.08, 0.10],
                               "vcp_min_score":   [40, 55],
                               "min_contractions":[2, 3],
                               "time_stop_days":  [14, 21],
                             }
        base_params:       base parameter dict (defaults from DEFAULT_PARAMS)
        metric:            metric key to optimise (default "sharpe_ratio")
        max_workers:       max parallel processes (default 4)
        progress_callback: optional callable(done, total) for progress updates

    Returns:
        results_df:  DataFrame — one row per param combo, sorted by metric desc
        best_params: dict of best parameter combination
        best_score:  float — best metric value
    """
    base = {**DEFAULT_PARAMS, **(base_params or {})}

    keys   = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))

    if not combos:
        return pd.DataFrame(), {}, 0.0

    # Build full param dict for each combo
    work_items = []
    for combo in combos:
        params = {**base}
        for k, v in zip(keys, combo):
            params[k] = v
        work_items.append((tickers, params))

    results  = []
    done     = 0
    total    = len(work_items)

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_combo, item): item for item in work_items}
        for future in as_completed(futures):
            done += 1
            if progress_callback:
                progress_callback(done, total)
            try:
                row = future.result()
                if row is not None:
                    results.append(row)
            except Exception:
                pass

    if not results:
        return pd.DataFrame(), {}, 0.0

    df = pd.DataFrame(results)
    if metric not in df.columns:
        metric = "sharpe_ratio" if "sharpe_ratio" in df.columns else df.columns[-1]

    df = df.sort_values(metric, ascending=False).reset_index(drop=True)

    best_row   = df.iloc[0]
    best_score = float(best_row.get(metric, 0))

    # Extract only the param_grid keys for best_params
    best_params = {}
    for k in keys:
        if k in best_row:
            best_params[k] = best_row[k]

    return df, best_params, best_score
