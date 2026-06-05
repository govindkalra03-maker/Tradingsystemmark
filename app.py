"""
SEPA Scanner Web Platform
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
from flask import Flask, Response, jsonify, render_template, send_file

warnings.filterwarnings("ignore")

from indicators import calculate_indicators
from plot import generate_chart
from sepa_filter import check_sepa_conditions
from tickers import NIFTY500

BENCHMARK_TICKER = "^CRSLDX"
OUTPUT_DIR = "output"
CHARTS_DIR = os.path.join(OUTPUT_DIR, "charts")
MIN_ROWS = 260

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


if __name__ == "__main__":
    app.run(debug=True, port=5001, use_reloader=False)
