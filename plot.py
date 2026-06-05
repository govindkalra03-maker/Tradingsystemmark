"""
Chart generation for SEPA signals — dark-theme, 1-year price + volume panel.
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe for cron / headless)

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np

# ── colour palette ────────────────────────────────────────────────────────────
BG_OUTER   = "#0d0d1a"
BG_PANEL   = "#12122a"
C_PRICE    = "#00d4ff"
C_SMA50    = "#ffd700"
C_SMA150   = "#ff8c00"
C_SMA200   = "#7cfc00"
C_52H      = "#ff4545"
C_52L      = "#45ff88"
C_VOL_UP   = "#2ecc71"
C_VOL_DN   = "#e74c3c"
C_VOLAVG   = "#ffd700"
C_GRID     = "#1e1e3a"
C_TEXT     = "#ccccdd"


def _fmt_vol(x, _pos):
    if x >= 1e7:
        return f"{x/1e7:.1f}Cr"
    if x >= 1e5:
        return f"{x/1e5:.1f}L"
    return f"{x:.0f}"


def generate_chart(ticker: str, df: pd.DataFrame, ind: dict, save_path: str) -> None:
    """
    Save a 1-year SEPA chart (price + SMAs + volume) as a PNG.

    Args:
        ticker:    e.g. "RELIANCE.NS"
        df:        full OHLCV DataFrame (2 years)
        ind:       indicator dict from calculate_indicators()
        save_path: absolute path for the output .png
    """
    # Use the last 365 calendar days worth of rows (≈252 trading days)
    df_plot = df.tail(365).copy()

    # Flatten MultiIndex if needed
    if isinstance(df_plot.columns, pd.MultiIndex):
        df_plot.columns = df_plot.columns.get_level_values(0)

    close  = df_plot["Close"].squeeze().astype(float)
    open_  = df_plot["Open"].squeeze().astype(float)
    volume = df_plot["Volume"].squeeze().astype(float)
    dates  = df_plot.index

    # Recalculate SMAs on full history, then trim to plot window
    full_close  = df["Close"].squeeze().astype(float)
    sma_50_s    = full_close.rolling(50).mean().tail(365)
    sma_150_s   = full_close.rolling(150).mean().tail(365)
    sma_200_s   = full_close.rolling(200).mean().tail(365)
    vol_avg_20s = df["Volume"].squeeze().astype(float).rolling(20).mean().tail(365)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 10),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    fig.patch.set_facecolor(BG_OUTER)
    for ax in (ax1, ax2):
        ax.set_facecolor(BG_PANEL)
        ax.tick_params(colors=C_TEXT, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")
        ax.grid(True, color=C_GRID, linewidth=0.6, linestyle="--", alpha=0.8)

    # ── Price panel ───────────────────────────────────────────────────────────
    ax1.plot(dates, close,     color=C_PRICE,  linewidth=1.6, label="Price",   zorder=5)
    ax1.plot(dates, sma_50_s,  color=C_SMA50,  linewidth=1.1, label="SMA 50",  linestyle="--")
    ax1.plot(dates, sma_150_s, color=C_SMA150, linewidth=1.1, label="SMA 150", linestyle="--")
    ax1.plot(dates, sma_200_s, color=C_SMA200, linewidth=1.1, label="SMA 200", linestyle="--")

    ax1.axhline(ind["52w_High"], color=C_52H, linewidth=0.9, linestyle=":",
                alpha=0.85, label=f"52w High ₹{ind['52w_High']:,.1f}")
    ax1.axhline(ind["52w_Low"],  color=C_52L, linewidth=0.9, linestyle=":",
                alpha=0.85, label=f"52w Low  ₹{ind['52w_Low']:,.1f}")

    # Shade the zone between price and 52w high
    ax1.fill_between(dates, close, ind["52w_High"],
                     where=(close <= ind["52w_High"]),
                     alpha=0.04, color=C_52H)

    # Current price annotation
    price = ind["Last_Close"]
    ax1.annotate(
        f"  ₹{price:,.2f}",
        xy=(dates[-1], price),
        xytext=(dates[-1], price),
        color=C_PRICE, fontsize=9, fontweight="bold",
    )

    # Key metrics in title
    name = ticker.replace(".NS", "")
    pct_from_hi = (price / ind["52w_High"] - 1) * 100
    pct_from_lo = (price / ind["52w_Low"]  - 1) * 100
    ret_12m     = ind.get("Return_12m", 0) or 0
    vol_ratio   = ind.get("Vol_Ratio", 1) or 1
    slope       = ind["SMA_200"] - (ind.get("SMA_200_21d_ago") or ind["SMA_200"])

    ax1.set_title(
        f"{name}  ·  SEPA Signal  |  "
        f"₹{price:,.2f}  |  "
        f"{pct_from_hi:.1f}% from 52w High  |  "
        f"+{pct_from_lo:.1f}% from 52w Low  |  "
        f"12m Ret: {ret_12m:+.1%}  |  "
        f"Vol×: {vol_ratio:.1f}  |  "
        f"SMA200 slope: {slope:+.2f}",
        color="#ffffff", fontsize=10, fontweight="bold", pad=8,
    )
    ax1.set_ylabel("Price (₹)", color=C_TEXT)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    ax1.legend(
        loc="upper left", fontsize=8,
        facecolor=BG_OUTER, edgecolor="#333355", labelcolor=C_TEXT,
        framealpha=0.8,
    )

    # ── Volume panel ─────────────────────────────────────────────────────────
    vol_colors = [C_VOL_UP if c >= o else C_VOL_DN
                  for c, o in zip(close, open_)]
    ax2.bar(dates, volume, color=vol_colors, alpha=0.75, width=0.8, zorder=3)
    ax2.plot(dates, vol_avg_20s, color=C_VOLAVG, linewidth=1.0,
             label="Vol MA 20", zorder=4)
    ax2.set_ylabel("Volume", color=C_TEXT)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_vol))
    ax2.legend(loc="upper left", fontsize=8,
               facecolor=BG_OUTER, edgecolor="#333355",
               labelcolor=C_TEXT, framealpha=0.8)

    # ── X-axis ────────────────────────────────────────────────────────────────
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=25, ha="right", color=C_TEXT)

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(save_path, dpi=130, bbox_inches="tight", facecolor=BG_OUTER)
    plt.close(fig)
