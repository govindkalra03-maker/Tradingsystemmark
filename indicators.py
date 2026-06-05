import pandas as pd
import numpy as np


def _squeeze(series_or_df):
    """Ensure we always work with a 1-D Series."""
    if isinstance(series_or_df, pd.DataFrame):
        return series_or_df.squeeze()
    return series_or_df


def calculate_indicators(df: pd.DataFrame) -> dict | None:
    """
    Calculate all SEPA technical indicators.

    Returns a dict of indicator values, or None if data is insufficient.
    Requires at least 260 rows of OHLCV data (caller should pre-check).
    """
    try:
        close = _squeeze(df["Close"]).astype(float)
        volume = _squeeze(df["Volume"]).astype(float)

        if close.isna().all() or len(close.dropna()) < 260:
            return None

        # Moving averages
        sma_50_s = close.rolling(window=50).mean()
        sma_150_s = close.rolling(window=150).mean()
        sma_200_s = close.rolling(window=200).mean()

        last_close = float(close.iloc[-1])
        last_sma_50 = float(sma_50_s.iloc[-1])
        last_sma_150 = float(sma_150_s.iloc[-1])
        last_sma_200 = float(sma_200_s.iloc[-1])

        # 200 SMA value 21 trading days ago (index -22 = 21 bars before the last bar)
        valid_sma200 = sma_200_s.dropna()
        if len(valid_sma200) >= 22:
            sma_200_21d_ago = float(sma_200_s.iloc[-22])
        else:
            sma_200_21d_ago = None

        # 52-week high / low — 252 trading day rolling window
        high_52w = float(close.rolling(window=252, min_periods=1).max().iloc[-1])
        low_52w = float(close.rolling(window=252, min_periods=1).min().iloc[-1])

        # 20-day average volume
        avg_vol_20d = float(volume.rolling(window=20).mean().iloc[-1])
        last_vol = float(volume.iloc[-1])
        vol_ratio = (last_vol / avg_vol_20d) if avg_vol_20d > 0 else None

        # 12-month return (~252 trading days back)
        if len(close) >= 252:
            price_252_ago = float(close.iloc[-252])
            return_12m = (last_close / price_252_ago) - 1 if price_252_ago > 0 else None
        else:
            return_12m = None

        # Guard against NaN in key values
        for val in [last_close, last_sma_50, last_sma_150, last_sma_200, high_52w, low_52w]:
            if np.isnan(val):
                return None

        return {
            "Last_Close":      last_close,
            "SMA_50":          last_sma_50,
            "SMA_150":         last_sma_150,
            "SMA_200":         last_sma_200,
            "SMA_200_21d_ago": sma_200_21d_ago,
            "52w_High":        high_52w,
            "52w_Low":         low_52w,
            "Return_12m":      return_12m,
            "Avg_Vol_20d":     avg_vol_20d,
            "Vol_Ratio":       vol_ratio,
        }

    except Exception:
        return None
