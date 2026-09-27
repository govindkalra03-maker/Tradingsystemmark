"""
Position sizing and risk management for VCP setups.
"""

DEFAULT_CAPITAL  = 500_000   # ₹5 Lakh
DEFAULT_RISK_PCT = 0.01      # 1% of capital per trade
T1_MULT = 1.20               # +20%
T2_MULT = 1.35               # +35%
T3_MULT = 1.50               # +50%


def calculate_position(
    entry: float,
    stop: float,
    capital: float = DEFAULT_CAPITAL,
    risk_pct: float = DEFAULT_RISK_PCT,
) -> dict | None:
    """
    Calculate shares, targets, and R/R for a given entry and stop.

    Args:
        entry:    pivot / entry price
        stop:     stop-loss price
        capital:  total capital available
        risk_pct: fraction of capital to risk (0.01 = 1%)

    Returns dict with position details, or None if inputs are invalid.
    """
    if entry <= 0 or stop >= entry or stop <= 0:
        return None

    risk_per_share = entry - stop
    capital_at_risk = capital * risk_pct
    shares = max(1, int(capital_at_risk / risk_per_share))

    # Never exceed full capital
    if shares * entry > capital:
        shares = max(1, int(capital / entry))

    position_value  = round(shares * entry, 2)
    actual_risk     = round(shares * risk_per_share, 2)
    stop_pct        = round((entry - stop) / entry * 100, 2)

    t1 = round(entry * T1_MULT, 2)
    t2 = round(entry * T2_MULT, 2)
    t3 = round(entry * T3_MULT, 2)

    rr = round((t1 - entry) / risk_per_share, 2)

    return {
        "shares":          shares,
        "position_value":  position_value,
        "capital_at_risk": actual_risk,
        "stop_pct":        stop_pct,
        "rr_ratio":        rr,
        "target_1":        t1,
        "target_2":        t2,
        "target_3":        t3,
    }
