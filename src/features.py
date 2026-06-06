import numpy as np
import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # =========================
    # TARGET STYLE FEATURES
    # =========================
    df['oil_ret'] = np.log(df['global_oil_price']).diff()

    # lags (returns only)
    for lag in [1, 2, 3]:
        df[f'oil_ret_lag{lag}'] = df['oil_ret'].shift(lag)
        df[f'palm_ret_lag{lag}'] = np.log(df['palm_oil_price']).diff().shift(lag)
        df[f'crude_ret_lag{lag}'] = np.log(df['crude_oil_price']).diff().shift(lag)

    # =========================
    # SPREAD FEATURES
    # =========================
    df['palm_vs_oil'] = np.log(df['palm_oil_price']) - np.log(df['global_oil_price'])
    df['crude_vs_oil'] = np.log(df['crude_oil_price']) - np.log(df['global_oil_price'])

    # =========================
    # VOLATILITY
    # =========================
    df['oil_volatility'] = df['oil_ret'].rolling(3).std()
    df['palm_volatility'] = np.log(df['palm_oil_price']).diff().rolling(3).std()

    # =========================
    # MOMENTUM
    # =========================
    df['oil_momentum'] = df['oil_ret'].rolling(3).mean()

    # =========================
    # MACRO
    # =========================
    df['fx_change'] = df['DTW'].pct_change()

    return df