"""
data_pipeline.py
----------------
ETL pipeline for financial time-series data.
Stage 1 - Ingest:  Download OHLCV data via yfinance
Stage 2 - Transform: Feature engineering + MinMaxScaler normalization
Stage 3 - Load: Build supervised learning sequences for RNN/LSTM input

Usage:
    from data_pipeline import build_pipeline
    X_train, X_test, y_train, y_test, scaler, df = build_pipeline("AAPL")
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

try:
    import yfinance as yf
except ImportError:
    raise ImportError("Please install yfinance: pip install yfinance")

# Configure logging so pipeline steps are visible during execution
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# STAGE 1: INGEST
# ──────────────────────────────────────────────

def fetch_stock_data(
    ticker: str,
    start: str = "2020-01-01",
    end: str = "2024-01-01"
) -> pd.DataFrame:
    """
    Download daily OHLCV data for a given ticker from Yahoo Finance.

    Parameters
    ----------
    ticker : str   e.g. "AAPL", "TSLA", "META"
    start  : str   ISO date string
    end    : str   ISO date string

    Returns
    -------
    pd.DataFrame with columns: Open, High, Low, Close, Volume
    """
    logger.info(f"[INGEST] Fetching {ticker} from {start} to {end}")
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check symbol and date range.")

    # Flatten multi-level columns if present (yfinance v0.2+)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Keep only OHLCV columns
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)

    logger.info(f"[INGEST] Downloaded {len(df)} trading days for {ticker}")
    return df


# ──────────────────────────────────────────────
# STAGE 2: TRANSFORM
# ──────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicator features to raw OHLCV data.

    Features added:
    - MA_5  : 5-day moving average of Close
    - MA_20 : 20-day moving average of Close
    - Volatility : 20-day rolling standard deviation of Close
    - Target : next-day Close price (shifted by -1)

    Rows with NaN (from rolling windows) are dropped.
    """
    logger.info("[TRANSFORM] Engineering features: MA_5, MA_20, Volatility, Target")

    df = df.copy()
    df["MA_5"]       = df["Close"].rolling(5).mean()
    df["MA_20"]      = df["Close"].rolling(20).mean()
    df["Volatility"] = df["Close"].rolling(20).std()
    df["Target"]     = df["Close"].shift(-1)   # next-day close = prediction target

    df.dropna(inplace=True)
    logger.info(f"[TRANSFORM] After feature engineering: {len(df)} rows, {df.shape[1]} columns")
    return df


def normalize(df: pd.DataFrame, feature_cols: list, target_col: str = "Target"):
    """
    Apply MinMaxScaler to features and target independently.

    Returns
    -------
    df_scaled  : pd.DataFrame  normalized dataframe
    scaler     : MinMaxScaler  fitted on Close column (used for inverse transform)
    """
    logger.info(f"[TRANSFORM] Normalizing columns: {feature_cols + [target_col]}")

    df_scaled = df.copy()
    scaler = MinMaxScaler()

    cols_to_scale = feature_cols + [target_col]
    df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    return df_scaled, scaler


# ──────────────────────────────────────────────
# STAGE 3: LOAD
# ──────────────────────────────────────────────

def build_sequences(
    df_scaled: pd.DataFrame,
    feature_cols: list,
    target_col: str = "Target",
    train_ratio: float = 0.8
) -> tuple:
    """
    Construct input/target arrays for supervised learning.

    For RNN/LSTM, each input x[t] is a 1xM row vector (row convention),
    and y[t] is the next-day Close (scalar).

    Parameters
    ----------
    df_scaled   : normalized DataFrame
    feature_cols: list of column names to use as input features
    target_col  : column name for prediction target
    train_ratio : fraction of data used for training (default 80%)

    Returns
    -------
    inputs_train, inputs_test  : pd.DataFrame slices
    targets_train, targets_test: pd.Series slices
    split_idx                  : int, index of train/test boundary
    """
    inputs  = df_scaled[feature_cols]
    targets = df_scaled[target_col]

    split_idx = int(len(df_scaled) * train_ratio)

    inputs_train  = inputs.iloc[:split_idx]
    inputs_test   = inputs.iloc[split_idx:]
    targets_train = targets.iloc[:split_idx]
    targets_test  = targets.iloc[split_idx:]

    logger.info(
        f"[LOAD] Train: {len(inputs_train)} samples | "
        f"Test: {len(inputs_test)} samples | "
        f"Features: {feature_cols}"
    )
    return inputs_train, inputs_test, targets_train, targets_test, split_idx


# ──────────────────────────────────────────────
# PUBLIC API: single-call pipeline
# ──────────────────────────────────────────────

FEATURE_COLS = ["Close", "Volume", "MA_5", "MA_20", "Volatility"]

def build_pipeline(
    ticker: str,
    start: str = "2020-01-01",
    end: str = "2024-01-01",
    train_ratio: float = 0.8
):
    """
    End-to-end ETL pipeline: Ingest → Transform → Load.

    Parameters
    ----------
    ticker      : Yahoo Finance ticker symbol (e.g. "AAPL")
    start / end : date range for download
    train_ratio : fraction for train split

    Returns
    -------
    inputs_train, inputs_test   : pd.DataFrame
    targets_train, targets_test : pd.Series
    scaler                      : fitted MinMaxScaler
    df_scaled                   : full normalized DataFrame
    """
    # Stage 1
    df_raw = fetch_stock_data(ticker, start, end)

    # Stage 2
    df_feat   = engineer_features(df_raw)
    df_scaled, scaler = normalize(df_feat, FEATURE_COLS)

    # Stage 3
    inputs_train, inputs_test, targets_train, targets_test, _ = build_sequences(
        df_scaled, FEATURE_COLS, train_ratio=train_ratio
    )

    return inputs_train, inputs_test, targets_train, targets_test, scaler, df_scaled


# ──────────────────────────────────────────────
# Quick smoke-test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    i_train, i_test, t_train, t_test, scaler, df = build_pipeline("AAPL")
    print("\nPipeline output summary:")
    print(f"  Train inputs  : {i_train.shape}")
    print(f"  Test  inputs  : {i_test.shape}")
    print(f"  Train targets : {t_train.shape}")
    print(f"  Test  targets : {t_test.shape}")
    print(f"\nFirst 3 rows of scaled training data:")
    print(i_train.head(3))
