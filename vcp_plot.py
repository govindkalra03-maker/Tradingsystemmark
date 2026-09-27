"""
VCP chart generation — dark theme, 1-year price with SMAs, pivot, stop,
and the three profit targets clearly marked.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np

# ── Colour palette (matches plot.py) ─────────────────────────────────────────
BG_OUTER  = "#0d0d1a"
BG_PANEL  = "#12122a"
C_PRICE   = "#00d4ff"
C_SMA50   = "#ffd700"
C_SMA150  = "#ff8c00"
C_SMA200  = "#7cfc00"
C_PIVOT   = "#ffffff"
C_STOP    = "#ff4545"
C_T1      = "#45ff88"
C_T2      = "#88ffaa"
C_T3      = "#bbffcc"
C_VOL_UP  = "#2ecc71"
C_VOL_DN  = "#e74c3c"
C_VOLAVG  = "#ffd700"
C_GRID    = "#1e1e3a"
C_TEXT    = "#ccccdd"


def _fmt_vol(x, _pos):
    if x >= 1e7:
        return f"{x/1e7:.1f}Cr"
    if x >= 1e5:
        return f"{x/1e5:.1f}L"
    return f"{x:.0f}"


def generate_vcp_chart(
    ticker: str,
    df: pd.DataFrame,
    result: dict,
    save_path: str,
) -> None:
    """
    Save a 1-year VCP chart (price + SMAs + VCP levels + volume) as PNG.

    Args:
        ticker:    e.g. "RELIANCE.NS"
        df:        full 2-year OHLCV DataFrame
        result:    VCP result dict from scan_ticker_vcp()
        save_path: destination path for the .png file
    """
    df_plot = df.tail(365).copy()
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
    vol_avg_s   = df["Volume"].squeeze().astype(float).rolling(20).mean().tail(365)

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

    # ── Price + SMAs ──────────────────────────────────────────────────────────
    ax1.plot(dates, close,     color=C_PRICE,  linewidth=1.6, label="Price",   zorder=5)
    ax1.plot(dates, sma_50_s,  color=C_SMA50,  linewidth=1.1, label="SMA 50",  linestyle="--")
    ax1.plot(dates, sma_150_s, color=C_SMA150, linewidth=1.1, label="SMA 150", linestyle="--")
    ax1.plot(dates, sma_200_s, color=C_SMA200, linewidth=1.1, label="SMA 200", linestyle="--")

    # ── VCP levels ────────────────────────────────────────────────────────────
    pivot = result["Pivot"]
    stop  = result["Stop_Loss"]
    t1    = result["Target_1"]
    t2    = result["Target_2"]
    t3    = result["Target_3"]

    ax1.axhline(pivot, color=C_PIVOT, linewidth=1.4, linestyle="-",
                alpha=0.95, label=f"Pivot  ₹{pivot:,.2f}", zorder=6)
    ax1.axhline(stop,  color=C_STOP,  linewidth=1.1, linestyle="--",
                alpha=0.90, label=f"Stop   ₹{stop:,.2f}  (-{result['Stop_Pct']}%)")
    ax1.axhline(t1, color=C_T1, linewidth=0.9, linestyle=":",
                alpha=0.85, label=f"T1     ₹{t1:,.2f}  (+20%)")
    ax1.axhline(t2, color=C_T2, linewidth=0.9, linestyle=":",
                alpha=0.60, label=f"T2     ₹{t2:,.2f}  (+35%)")
    ax1.axhline(t3, color=C_T3, linewidth=0.9, linestyle=":",
                alpha=0.40, label=f"T3     ₹{t3:,.2f}  (+50%)")

    # Risk zone shading (stop → pivot)
    ax1.fill_between(dates, [stop] * len(dates), [pivot] * len(dates),
                     alpha=0.05, color=C_STOP)

    # Reward zone shading (pivot → T1)
    ax1.fill_between(dates, [pivot] * len(dates), [t1] * len(dates),
                     alpha=0.04, color=C_T1)

    # Current price annotation
    price = result["Last_Close"]
    ax1.annotate(
        f"  ₹{price:,.2f}",
        xy=(dates[-1], price), xytext=(dates[-1], price),
        color=C_PRICE, fontsize=9, fontweight="bold",
    )

    name = ticker.replace(".NS", "")
    ax1.set_title(
        f"{name}  ·  VCP Signal  |  "
        f"Score: {result['VCP_Score']}/100  |  "
        f"{result['N_Contractions']} contractions  |  "
        f"Tightening: {result['Tightening_Pct']}%  |  "
        f"Pivot: ₹{pivot:,.2f}  |  "
        f"Stop: ₹{stop:,.2f} (-{result['Stop_Pct']}%)  |  "
        f"R/R: {result['RR_Ratio']}×",
        color="#ffffff", fontsize=10, fontweight="bold", pad=8,
    )
    ax1.set_ylabel("Price (₹)", color=C_TEXT)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    ax1.legend(
        loc="upper left", fontsize=8,
        facecolor=BG_OUTER, edgecolor="#333355", labelcolor=C_TEXT,
        framealpha=0.85,
    )

    # ── Volume panel ──────────────────────────────────────────────────────────
    vol_colors = [C_VOL_UP if c >= o else C_VOL_DN for c, o in zip(close, open_)]
    ax2.bar(dates, volume, color=vol_colors, alpha=0.75, width=0.8, zorder=3)
    ax2.plot(dates, vol_avg_s, color=C_VOLAVG, linewidth=1.0,
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

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight", facecolor=BG_OUTER)
    plt.close(fig)
