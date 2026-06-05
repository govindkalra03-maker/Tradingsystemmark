# Minervini SEPA Trend Template Scanner — NSE India

Scans the full Nifty 500 universe daily and surfaces stocks that satisfy all 8 of Mark Minervini's **Specific Entry Point Analysis (SEPA)** Trend Template conditions.

---

## Project layout

```
sepa_scanner/
├── scanner.py        ← main orchestrator — run this
├── indicators.py     ← SMA, 52w high/low, RS, volume calculations
├── sepa_filter.py    ← all 8 SEPA condition checks
├── plot.py           ← dark-theme chart generation
├── tickers.py        ← full Nifty 500 .NS ticker list
├── run_daily.sh      ← bash wrapper for cron
├── requirements.txt  ← Python dependencies
└── output/
    ├── sepa_signals_YYYY-MM-DD.csv   ← daily results
    └── charts/                        ← one PNG per passing stock
```

---

## Quick start

```bash
# 1. Clone / navigate to the directory
cd sepa_scanner

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the scanner
python scanner.py
```

Results are printed to the terminal and saved in `output/sepa_signals_YYYY-MM-DD.csv`.  
A chart PNG is generated for every passing stock in `output/charts/`.

---

## The 8 SEPA conditions

| # | Condition | Rationale |
|---|-----------|-----------|
| 1 | Price > 150-day SMA | Above intermediate trend |
| 2 | Price > 200-day SMA | Above long-term trend |
| 3 | 200-day SMA > value 21 days ago | Long-term trend is rising |
| 4 | SMA 50 > SMA 150 > SMA 200 | All averages perfectly stacked |
| 5 | Price ≥ 75% of 52-week high | Near the top, not a laggard |
| 6 | 52w High ≥ 130% of 52w Low | Stock has meaningful range / volatility |
| 7 | Price ≥ 125% of 52-week low | Well off the bottom |
| 8 | 12-month return > Nifty 500 return | Genuine relative strength |

A stock must satisfy **all 8** to appear in results.

---

## Output columns

| Column | Description |
|--------|-------------|
| `Ticker` | Yahoo Finance symbol (e.g. `RELIANCE.NS`) |
| `Last_Close` | Last closing price (₹) |
| `SMA_50/150/200` | Moving average values |
| `200_SMA_Slope` | SMA 200 now minus SMA 200 21 days ago (positive = rising) |
| `52w_High / 52w_Low` | 252-day rolling high and low |
| `Pct_From_High` | % below 52-week high (negative; closer to 0 = stronger) |
| `Pct_From_Low` | % above 52-week low (higher = further off the bottom) |
| `HighLow_Ratio` | 52w High ÷ 52w Low |
| `RS_vs_Nifty` | Stock 12m return minus Nifty 500 12m return (pp) |
| `Vol_Ratio` | Today's volume ÷ 20-day average volume |

Results are **sorted by `RS_vs_Nifty` descending** — highest relative strength leaders first.

---

## Scheduled daily run (cron)

```bash
# Make the script executable
chmod +x run_daily.sh

# Open crontab
crontab -e

# Run Monday–Friday at 09:20 AM IST (03:50 UTC)
50 3 * * 1-5 /Users/govindkalra/Documents/developer/sepa_scanner/run_daily.sh
```

Logs are written to `logs/scan_YYYY-MM-DD.log` (last 30 days kept automatically).

---

## Updating the ticker list

The `NIFTY500` list in `tickers.py` may drift as NSE rebalances the index.  
Download the latest constituents from:

> [https://www.niftyindices.com/indices/equity/broad-based-indices/NIFTY-500](https://www.niftyindices.com/indices/equity/broad-based-indices/NIFTY-500)

Append `.NS` to each symbol and update the list accordingly.

---

## Benchmark

The scanner uses **`^CRSLDX`** (Nifty 500 index) as the RS benchmark.  
Change `BENCHMARK_TICKER` in `scanner.py` to swap in another index (e.g. `^NSEI` for Nifty 50).

---

## Disclaimer

This tool is for **educational and research purposes only**.  
Nothing here is financial advice. Always do your own due diligence.
