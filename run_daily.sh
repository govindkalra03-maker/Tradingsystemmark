#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_daily.sh — Run the Minervini SEPA scanner every trading morning
#
# Recommended cron (Mon–Fri at 9:20 AM IST, after NSE open):
#   20 3 * * 1-5 /path/to/sepa_scanner/run_daily.sh >> /path/to/sepa_scanner/logs/cron.log 2>&1
#
# To set up: crontab -e  (then paste the line above with correct paths)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
DATE_STR="$(date '+%Y-%m-%d')"
LOG_FILE="$LOG_DIR/scan_$DATE_STR.log"

mkdir -p "$LOG_DIR"

echo "══════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "  SEPA Scanner started: $(date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a "$LOG_FILE"
echo "══════════════════════════════════════════" | tee -a "$LOG_FILE"

cd "$SCRIPT_DIR"

# Activate virtual environment if present
if [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/venv/bin/activate"
    echo "  Using venv: $SCRIPT_DIR/venv" | tee -a "$LOG_FILE"
fi

# Run the scanner and capture exit code
python scanner.py 2>&1 | tee -a "$LOG_FILE"
EXIT_CODE="${PIPESTATUS[0]}"

echo "" | tee -a "$LOG_FILE"
echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S %Z') — exit $EXIT_CODE" | tee -a "$LOG_FILE"

# Keep only the last 30 log files
cd "$LOG_DIR" && ls -t scan_*.log 2>/dev/null | tail -n +31 | xargs rm -f --

exit "$EXIT_CODE"
