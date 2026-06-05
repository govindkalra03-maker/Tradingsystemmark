"""
Minervini SEPA (Specific Entry Point Analysis) Trend Template — 8 conditions.

All 8 must pass for a stock to appear in the scan results.
"""


def check_sepa_conditions(ind: dict, benchmark_return: float) -> tuple[bool, dict]:
    """
    Evaluate all 8 SEPA Trend Template conditions.

    Args:
        ind: indicator dict from calculate_indicators()
        benchmark_return: Nifty 500 12-month return as a decimal (e.g. 0.18 = 18%)

    Returns:
        (all_passed: bool, conditions: dict[str, bool])
    """
    price    = ind["Last_Close"]
    sma_50   = ind["SMA_50"]
    sma_150  = ind["SMA_150"]
    sma_200  = ind["SMA_200"]
    sma_200_21d = ind["SMA_200_21d_ago"]
    high_52w = ind["52w_High"]
    low_52w  = ind["52w_Low"]
    ret_12m  = ind["Return_12m"]

    # Bail out early if required values are missing or degenerate
    if sma_200_21d is None or ret_12m is None:
        return False, {}
    if low_52w <= 0 or high_52w <= 0:
        return False, {}

    conditions = {
        # 1. Price above 150-day SMA
        "C1_Price_gt_SMA150": price > sma_150,

        # 2. Price above 200-day SMA
        "C2_Price_gt_SMA200": price > sma_200,

        # 3. 200-day SMA is trending up (higher than 21 trading days ago)
        "C3_SMA200_Slope_Rising": sma_200 > sma_200_21d,

        # 4. Moving averages perfectly stacked: 50 > 150 > 200
        "C4_SMA_Stack_50_150_200": (sma_50 > sma_150) and (sma_150 > sma_200),

        # 5. Price within 25% of its 52-week high
        "C5_Within_25pct_52w_High": (price / high_52w) >= 0.75,

        # 6. 52-week high is at least 30% above the 52-week low (meaningful range)
        "C6_52w_Range_gte_30pct": (high_52w / low_52w) >= 1.30,

        # 7. Price is at least 25% above the 52-week low (off the bottom)
        "C7_Price_25pct_Above_52w_Low": (price / low_52w) >= 1.25,

        # 8. Stock's 12-month return beats Nifty 500 (relative strength)
        "C8_RS_Beats_Nifty500": ret_12m > benchmark_return,
    }

    return all(conditions.values()), conditions
