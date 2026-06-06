import numpy as np
import pandas as pd

from src.features import create_features


def forecast(model, features, df, steps=3):

    sim_df = df.copy()

    last_price = df['global_oil_price'].iloc[-1]

    forecasts = []

    for _ in range(steps):

        sim_df = create_features(sim_df).dropna()

        X_last = sim_df[features].iloc[-1:]

        pred_ret = model.predict(X_last)[0]

        next_price = last_price * np.exp(pred_ret)

        forecasts.append(next_price)

        # append new row
        new_row = sim_df.iloc[-1:].copy()
        new_row['global_oil_price'] = next_price

        sim_df = pd.concat([sim_df, new_row])

        last_price = next_price

    print("\n===== FORECAST =====")
    print(forecasts)

    return forecasts