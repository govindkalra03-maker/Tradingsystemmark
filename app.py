"""
SEPA + VCP Scanner Web Platform
Run:  python app.py
Then open http://localhost:5001
"""

import csv
import io
import os
import threading
import warnings
from datetime import datetime

import pandas as pd
import yfinance as yf
from flask import Flask, Response, jsonify, render_template, request, send_file

warnings.filterwarnings("ignore")

from indicators import calculate_indicators
from plot import generate_chart
from sepa_filter import check_sepa_conditions
from tickers import NIFTY500
from vcp_scanner import scan_ticker_vcp
from vcp_plot import generate_vcp_chart
from backtest_engine import VCPBacktester
from backtest_metrics import calculate_metrics
from backtest_monte_carlo import run_monte_carlo
from backtest_optimiser import optimise_parameters

BENCHMARK_TICKER  = "^CRSLDX"
OUTPUT_DIR        = "output"
CHARTS_DIR        = os.path.join(OUTPUT_DIR, "charts")
VCP_CHARTS_DIR    = os.path.join(OUTPUT_DIR, "vcp_charts")
MIN_ROWS          = 260

app = Flask(__name__)

# ── Shared scan state ─────────────────────────────────────────────────────────
_state = {
    "running": False,
    "current_idx": 0,
    "current_ticker": "",
    "total": len(NIFTY500),
    "passed": 0,
    "results": [],
    "log": [],
    "completed": False,
    "started_at": None,
    "completed_at": None,
    "benchmark_return": None,
    "error": None,
}
_lock = threading.Lock()


def _squeeze(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.squeeze()
    return obj


def _dl(symbol):
    try:
        df = yf.download(symbol, period="2y", auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return None


def _log(msg):
    with _lock:
        _state["log"].append(msg)
        if len(_state["log"]) > 60:
            _state["log"] = _state["log"][-60:]


def _bg_scan():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHARTS_DIR, exist_ok=True)

    with _lock:
        _state.update({
            "running": True,
            "current_idx": 0,
            "current_ticker": "Downloading benchmark…",
            "passed": 0,
            "results": [],
            "log": [],
            "completed": False,
            "started_at": datetime.now().isoformat(),
            "completed_at": None,
            "benchmark_return": None,
            "error": None,
        })

    # ── Benchmark ─────────────────────────────────────────────────────────────
    df_b = _dl(BENCHMARK_TICKER)
    if df_b is None or len(df_b) < 252:
        with _lock:
            _state["running"] = False
            _state["completed"] = True
            _state["error"] = "Could not download benchmark (^CRSLDX)"
        return

    close_b = _squeeze(df_b["Close"]).astype(float).dropna()
    bench_ret = float((close_b.iloc[-1] / close_b.iloc[-252]) - 1)

    with _lock:
        _state["benchmark_return"] = round(bench_ret * 100, 2)

    _log(f"Benchmark ready — Nifty 500 12m return: {bench_ret:+.2%}")

    # ── Ticker loop ───────────────────────────────────────────────────────────
    for idx, ticker in enumerate(NIFTY500, 1):
        with _lock:
            _state["current_idx"] = idx
            _state["current_ticker"] = ticker

        try:
            df = _dl(ticker)
            if df is None or len(df) < MIN_ROWS:
                _log(f"SKIP  {ticker}  (insufficient data)")
                continue

            ind = calculate_indicators(df)
            if ind is None:
                _log(f"SKIP  {ticker}  (indicator calc failed)")
                continue

            passed, conditions = check_sepa_conditions(ind, bench_ret)
            if not passed:
                bad = next((k for k, v in conditions.items() if not v), "unknown")
                _log(f"FAIL  {ticker}  → {bad}")
                continue

            price    = ind["Last_Close"]
            high_52w = ind["52w_High"]
            low_52w  = ind["52w_Low"]

            row = {
                "Ticker":        ticker.replace(".NS", ""),
                "TickerFull":    ticker,
                "Last_Close":    round(price, 2),
                "SMA_50":        round(ind["SMA_50"], 2),
                "SMA_150":       round(ind["SMA_150"], 2),
                "SMA_200":       round(ind["SMA_200"], 2),
                "SMA_200_Slope": round(ind["SMA_200"] - ind["SMA_200_21d_ago"], 4),
                "52w_High":      round(high_52w, 2),
                "52w_Low":       round(low_52w, 2),
                "Pct_From_High": round((price / high_52w - 1) * 100, 2),
                "Pct_From_Low":  round((price / low_52w - 1) * 100, 2),
                "HighLow_Ratio": round(high_52w / low_52w, 3),
                "RS_vs_Nifty":   round((ind["Return_12m"] - bench_ret) * 100, 2),
                "Vol_Ratio":     round(ind["Vol_Ratio"], 2) if ind["Vol_Ratio"] else None,
            }

            with _lock:
                _state["results"].append(row)
                _state["passed"] += 1

            _log(f"PASS  {ticker}  ✓  RS={row['RS_vs_Nifty']:+.1f}%")

            try:
                safe = ticker.replace(".NS", "").replace("&", "_").replace("-", "_")
                generate_chart(ticker, df, ind, os.path.join(CHARTS_DIR, f"{safe}.png"))
            except Exception:
                pass

        except Exception as e:
            _log(f"ERROR {ticker}  ({e})")

    with _lock:
        _state["running"] = False
        _state["completed"] = True
        _state["completed_at"] = datetime.now().isoformat()

    _log(f"Scan complete — {_state['passed']} / {len(NIFTY500)} stocks passed all 8 conditions")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/scan/start", methods=["POST"])
def start_scan():
    with _lock:
        if _state["running"]:
            return jsonify({"error": "Scan already running"}), 400
    threading.Thread(target=_bg_scan, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/scan/progress")
def progress():
    with _lock:
        data = {k: v for k, v in _state.items() if k != "log"}
        data["log"] = list(_state["log"])
    return jsonify(data)


@app.route("/api/chart/<path:ticker>")
def chart(ticker):
    safe = ticker.replace(".NS", "").replace("&", "_").replace("-", "_")
    path = os.path.join(CHARTS_DIR, f"{safe}.png")
    if not os.path.exists(path):
        return jsonify({"error": "Chart not found"}), 404
    return send_file(path, mimetype="image/png")


@app.route("/api/export/csv")
def export_csv():
    with _lock:
        results = list(_state["results"])
    if not results:
        return jsonify({"error": "No results to export"}), 404

    buf = io.StringIO()
    export_keys = [k for k in results[0].keys() if k != "TickerFull"]
    writer = csv.DictWriter(buf, fieldnames=export_keys, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(results)

    date_str = datetime.now().strftime("%Y-%m-%d")
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=sepa_signals_{date_str}.csv"},
    )


# ═════════════════════════════════════════════════════════════════════════════
# VCP SCANNER
# ═════════════════════════════════════════════════════════════════════════════

_vcp_state = {
    "running":          False,
    "current_idx":      0,
    "current_ticker":   "",
    "total":            len(NIFTY500),
    "passed":           0,
    "results":          [],
    "log":              [],
    "completed":        False,
    "started_at":       None,
    "completed_at":     None,
    "benchmark_return": None,
    "error":            None,
}
_vcp_lock = threading.Lock()


def _vcp_log(msg):
    with _vcp_lock:
        _vcp_state["log"].append(msg)
        if len(_vcp_state["log"]) > 60:
            _vcp_state["log"] = _vcp_state["log"][-60:]


def _bg_vcp_scan():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(VCP_CHARTS_DIR, exist_ok=True)

    with _vcp_lock:
        _vcp_state.update({
            "running":          True,
            "current_idx":      0,
            "current_ticker":   "Downloading benchmark…",
            "passed":           0,
            "results":          [],
            "log":              [],
            "completed":        False,
            "started_at":       datetime.now().isoformat(),
            "completed_at":     None,
            "benchmark_return": None,
            "error":            None,
        })

    # ── Benchmark ─────────────────────────────────────────────────────────────
    df_b = _dl(BENCHMARK_TICKER)
    if df_b is None or len(df_b) < 252:
        with _vcp_lock:
            _vcp_state["running"]   = False
            _vcp_state["completed"] = True
            _vcp_state["error"]     = "Could not download benchmark (^CRSLDX)"
        return

    close_b   = _squeeze(df_b["Close"]).astype(float).dropna()
    bench_ret = float((close_b.iloc[-1] / close_b.iloc[-252]) - 1)

    with _vcp_lock:
        _vcp_state["benchmark_return"] = round(bench_ret * 100, 2)

    _vcp_log(f"Benchmark ready — Nifty 500 12m return: {bench_ret:+.2%}")

    # ── Ticker loop ───────────────────────────────────────────────────────────
    for idx, ticker in enumerate(NIFTY500, 1):
        with _vcp_lock:
            _vcp_state["current_idx"]    = idx
            _vcp_state["current_ticker"] = ticker

        try:
            df = _dl(ticker)
            if df is None or len(df) < MIN_ROWS:
                _vcp_log(f"SKIP  {ticker}  (insufficient data)")
                continue

            result = scan_ticker_vcp(ticker, df, bench_ret)

            if result is None:
                _vcp_log(f"FAIL  {ticker}")
                continue

            # Generate chart before stripping _seq
            try:
                safe = (ticker.replace(".NS", "")
                              .replace("&", "_")
                              .replace("-", "_"))
                generate_vcp_chart(
                    ticker, df, result,
                    os.path.join(VCP_CHARTS_DIR, f"{safe}.png"),
                )
            except Exception:
                pass

            # Strip internal-only key before storing
            row = {k: v for k, v in result.items() if k != "_seq"}

            with _vcp_lock:
                _vcp_state["results"].append(row)
                _vcp_state["passed"] += 1

            _vcp_log(
                f"PASS  {ticker}  ✓  "
                f"Score={result['VCP_Score']}  "
                f"C={result['N_Contractions']}  "
                f"R/R={result['RR_Ratio']}×"
            )

        except Exception as e:
            _vcp_log(f"ERROR {ticker}  ({e})")

    with _vcp_lock:
        _vcp_state["running"]      = False
        _vcp_state["completed"]    = True
        _vcp_state["completed_at"] = datetime.now().isoformat()

    _vcp_log(
        f"VCP scan complete — "
        f"{_vcp_state['passed']} / {len(NIFTY500)} setups found"
    )


# ── VCP routes ────────────────────────────────────────────────────────────────

@app.route("/api/vcp/start", methods=["POST"])
def vcp_start():
    with _vcp_lock:
        if _vcp_state["running"]:
            return jsonify({"error": "VCP scan already running"}), 400
    threading.Thread(target=_bg_vcp_scan, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/vcp/progress")
def vcp_progress():
    with _vcp_lock:
        data = {k: v for k, v in _vcp_state.items() if k != "log"}
        data["log"] = list(_vcp_state["log"])
    return jsonify(data)


@app.route("/api/vcp/chart/<path:ticker>")
def vcp_chart(ticker):
    safe = (ticker.replace(".NS", "")
                  .replace("&", "_")
                  .replace("-", "_"))
    path = os.path.join(VCP_CHARTS_DIR, f"{safe}.png")
    if not os.path.exists(path):
        return jsonify({"error": "Chart not found"}), 404
    return send_file(path, mimetype="image/png")


@app.route("/api/vcp/export/csv")
def vcp_export_csv():
    with _vcp_lock:
        results = list(_vcp_state["results"])
    if not results:
        return jsonify({"error": "No VCP results to export"}), 404

    buf = io.StringIO()
    skip = {"TickerFull", "_seq"}
    export_keys = [k for k in results[0].keys() if k not in skip]
    writer = csv.DictWriter(buf, fieldnames=export_keys, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(results)

    date_str = datetime.now().strftime("%Y-%m-%d")
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={
            "Content-Disposition":
                f"attachment; filename=vcp_signals_{date_str}.csv"
        },
    )


# ═════════════════════════════════════════════════════════════════════════════
# BACKTEST
# ═════════════════════════════════════════════════════════════════════════════

_bt_state = {
    "running":      False,
    "completed":    False,
    "error":        None,
    "progress":     {"pct": 0, "msg": "Not started", "done": False, "error": None},
    "results":      None,   # full results dict set when done
}
_bt_lock = threading.Lock()

_opt_state = {
    "running":   False,
    "completed": False,
    "error":     None,
    "done":      0,
    "total":     0,
    "results":   None,
}
_opt_lock = threading.Lock()


def _bg_backtest(tickers, params):
    bt = VCPBacktester(params)

    def _sync_progress():
        with _bt_lock:
            _bt_state["progress"] = dict(bt.progress)

    # load data
    ok = bt.load_data(tickers)
    _sync_progress()
    if not ok:
        with _bt_lock:
            _bt_state["running"] = False
            _bt_state["completed"] = True
            _bt_state["error"] = bt.progress.get("error", "Data load failed")
        return

    # run simulation
    bt.run(tickers)
    _sync_progress()

    # compute analytics
    metrics = calculate_metrics(bt.all_trades, bt.daily_values, params["initial_capital"])
    mc      = run_monte_carlo(bt.all_trades, params["initial_capital"])

    # build trade log as list of dicts for JSON
    trade_log = bt.all_trades

    # Add the actual portfolio path to the fan chart for display
    actual_path = []
    equity = params["initial_capital"]
    for t in bt.all_trades:
        equity *= (1 + t["Net_PL_Pct"] / 100)
        actual_path.append(round(equity, 2))
    mc["actual_path"] = actual_path

    results = {
        "metrics":      metrics,
        "trade_log":    trade_log,
        "equity_curve": bt.daily_values,
        "monte_carlo":  mc,
    }

    with _bt_lock:
        _bt_state["running"]   = False
        _bt_state["completed"] = True
        _bt_state["results"]   = results


def _bg_optimise(tickers, param_grid, base_params, metric):
    total_combos = 1
    for v in param_grid.values():
        total_combos *= len(v)

    with _opt_lock:
        _opt_state["total"] = total_combos
        _opt_state["done"]  = 0

    def _cb(done, total):
        with _opt_lock:
            _opt_state["done"]  = done
            _opt_state["total"] = total

    try:
        df, best_params, best_score = optimise_parameters(
            tickers, param_grid, base_params=base_params,
            metric=metric, max_workers=4, progress_callback=_cb,
        )
        with _opt_lock:
            _opt_state["running"]   = False
            _opt_state["completed"] = True
            _opt_state["results"]   = {
                "rows":        df.to_dict(orient="records") if not df.empty else [],
                "best_params": best_params,
                "best_score":  best_score,
                "metric":      metric,
            }
    except Exception as e:
        with _opt_lock:
            _opt_state["running"]   = False
            _opt_state["completed"] = True
            _opt_state["error"]     = str(e)


# ── Backtest routes ───────────────────────────────────────────────────────────

@app.route("/api/backtest/start", methods=["POST"])
def backtest_start():
    with _bt_lock:
        if _bt_state["running"]:
            return jsonify({"error": "Backtest already running"}), 400
        _bt_state.update({
            "running": True, "completed": False, "error": None, "results": None,
            "progress": {"pct": 0, "msg": "Starting…", "done": False, "error": None},
        })

    body   = request.get_json(silent=True) or {}
    params = body.get("params", {})
    tickers_req = body.get("tickers", None)

    # Use a small default set if none supplied (for quick tests)
    if tickers_req:
        tickers = [t if t.endswith(".NS") else t + ".NS" for t in tickers_req]
    else:
        from tickers import NIFTY500
        tickers = NIFTY500

    threading.Thread(target=_bg_backtest, args=(tickers, params), daemon=True).start()
    return jsonify({"status": "started", "n_tickers": len(tickers)})


@app.route("/api/backtest/progress")
def backtest_progress():
    with _bt_lock:
        return jsonify({
            "running":   _bt_state["running"],
            "completed": _bt_state["completed"],
            "error":     _bt_state["error"],
            "progress":  _bt_state["progress"],
        })


@app.route("/api/backtest/results")
def backtest_results():
    with _bt_lock:
        if not _bt_state["completed"] or _bt_state["results"] is None:
            return jsonify({"error": "Results not ready"}), 404
        return jsonify(_bt_state["results"])


@app.route("/api/backtest/optimise/start", methods=["POST"])
def optimise_start():
    with _opt_lock:
        if _opt_state["running"]:
            return jsonify({"error": "Optimiser already running"}), 400
        _opt_state.update({
            "running": True, "completed": False, "error": None, "results": None,
            "done": 0, "total": 0,
        })

    body       = request.get_json(silent=True) or {}
    param_grid = body.get("param_grid", {
        "stop_loss_pct":   [0.06, 0.08, 0.10],
        "vcp_min_score":   [40, 55],
        "min_contractions":[2, 3],
        "time_stop_days":  [14, 21],
    })
    base_params   = body.get("base_params", {})
    metric        = body.get("metric", "sharpe_ratio")
    tickers_req   = body.get("tickers", None)

    if tickers_req:
        tickers = [t if t.endswith(".NS") else t + ".NS" for t in tickers_req]
    else:
        from tickers import NIFTY500
        tickers = NIFTY500[:50]   # default to Nifty 50 for speed

    threading.Thread(
        target=_bg_optimise,
        args=(tickers, param_grid, base_params, metric),
        daemon=True,
    ).start()
    return jsonify({"status": "started"})


@app.route("/api/backtest/optimise/progress")
def optimise_progress():
    with _opt_lock:
        return jsonify({
            "running":   _opt_state["running"],
            "completed": _opt_state["completed"],
            "error":     _opt_state["error"],
            "done":      _opt_state["done"],
            "total":     _opt_state["total"],
        })


@app.route("/api/backtest/optimise/results")
def optimise_results():
    with _opt_lock:
        if not _opt_state["completed"] or _opt_state["results"] is None:
            return jsonify({"error": "Optimiser results not ready"}), 404
        return jsonify(_opt_state["results"])


if __name__ == "__main__":
    app.run(debug=True, port=5001, use_reloader=False)
