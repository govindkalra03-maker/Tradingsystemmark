"""
Minervini SEPA Trend Template Scanner — NSE Nifty 500
Run:  python scanner.py
"""

import os
import sys
import warnings
import pandas as pd
import yfinance as yf
from datetime import datetime

warnings.filterwarnings("ignore")

from tickers import NIFTY500
from indicators import calculate_indicators
from sepa_filter import check_sepa_conditions
from plot import generate_chart

BENCHMARK_TICKER = "^CRSLDX"   # Nifty 500 index
OUTPUT_DIR = "output"
CHARTS_DIR = os.path.join(OUTPUT_DIR, "charts")
MIN_ROWS = 260


# ─── helpers ─────────────────────────────────────────────────────────────────

def _squeeze(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.squeeze()
    return obj


def download_ticker(symbol: str) -> pd.DataFrame | None:
    """Download 2 years of daily OHLCV. Returns None on any failure."""
    try:
        df = yf.download(symbol, period="2y", auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None
        # Flatten MultiIndex columns that yfinance ≥0.2.x sometimes returns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return None


def get_benchmark_return() -> float | None:
    """Download Nifty 500 and return its 12-month return as a decimal."""
    print(f"Downloading benchmark ({BENCHMARK_TICKER}) …", flush=True)
    df = download_ticker(BENCHMARK_TICKER)
    if df is None or len(df) < 252:
        print("  WARNING: insufficient benchmark data — RS condition disabled")
        return None
    close = _squeeze(df["Close"]).astype(float).dropna()
    if len(close) < 252:
        return None
    ret = float((close.iloc[-1] / close.iloc[-252]) - 1)
    print(f"  Nifty 500 12-month return: {ret:+.2%}\n")
    return ret


# ─── main scan ───────────────────────────────────────────────────────────────

def scan():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHARTS_DIR, exist_ok=True)

    benchmark_return = get_benchmark_return()
    if benchmark_return is None:
        print("ERROR: Cannot continue without benchmark return. Exiting.")
        sys.exit(1)

    results = []
    total = len(NIFTY500)

    print(f"Scanning {total} Nifty 500 tickers …\n")
    print(f"{'#':<6} {'Ticker':<20} {'Status'}")
    print("─" * 55)

    for idx, ticker in enumerate(NIFTY500, 1):
        label = f"[{idx}/{total}]"
        print(f"{label:<6} {ticker:<20}", end="", flush=True)

        try:
            df = download_ticker(ticker)

            if df is None or len(df) < MIN_ROWS:
                print(f"SKIP  (only {len(df) if df is not None else 0} rows)")
                continue

            ind = calculate_indicators(df)
            if ind is None:
                print("SKIP  (indicator calc failed)")
                continue

            passed, conditions = check_sepa_conditions(ind, benchmark_return)

            if not passed:
                failed_conds = [k for k, v in conditions.items() if not v]
                reason = failed_conds[0] if failed_conds else "unknown"
                print(f"FAIL  ({reason})")
                continue

            # ── All 8 conditions passed ──────────────────────────────────────
            price    = ind["Last_Close"]
            high_52w = ind["52w_High"]
            low_52w  = ind["52w_Low"]

            results.append({
                "Ticker":         ticker,
                "Last_Close":     round(price, 2),
                "SMA_50":         round(ind["SMA_50"], 2),
                "SMA_150":        round(ind["SMA_150"], 2),
                "SMA_200":        round(ind["SMA_200"], 2),
                "200_SMA_Slope":  round(ind["SMA_200"] - ind["SMA_200_21d_ago"], 4),
                "52w_High":       round(high_52w, 2),
                "52w_Low":        round(low_52w, 2),
                "Pct_From_High":  round((price / high_52w - 1) * 100, 2),
                "Pct_From_Low":   round((price / low_52w - 1) * 100, 2),
                "HighLow_Ratio":  round(high_52w / low_52w, 3),
                "RS_vs_Nifty":    round((ind["Return_12m"] - benchmark_return) * 100, 2),
                "Vol_Ratio":      round(ind["Vol_Ratio"], 2) if ind["Vol_Ratio"] else None,
            })

            print("PASS ✓")

            # Generate chart (non-fatal if it fails)
            try:
                chart_path = os.path.join(
                    CHARTS_DIR,
                    ticker.replace(".NS", "").replace("&", "_").replace("-", "_") + ".png"
                )
                generate_chart(ticker, df, ind, chart_path)
            except Exception as chart_err:
                print(f"         ↳ Chart error: {chart_err}")

        except Exception as e:
            print(f"ERROR ({e})")
            continue

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 55)
    print(f"  SEPA scan complete — {len(results)} / {total} stocks passed")
    print("═" * 55 + "\n")

    if not results:
        print("No stocks passed all 8 conditions today.")
        return []

    df_out = pd.DataFrame(results).sort_values("RS_vs_Nifty", ascending=False)

    date_str = datetime.now().strftime("%Y-%m-%d")
    csv_path = os.path.join(OUTPUT_DIR, f"sepa_signals_{date_str}.csv")
    df_out.to_csv(csv_path, index=False)

    # Pretty-print to terminal
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", "{:.2f}".format)
    print(df_out.to_string(index=False))
    print(f"\nCSV  → {csv_path}")
    print(f"Charts → {CHARTS_DIR}/")

    return results


if __name__ == "__main__":
    scan()
