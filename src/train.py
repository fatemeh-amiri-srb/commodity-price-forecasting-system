import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.features import create_features
from src.model import get_model


def prepare_data(df):

    df = create_features(df)

    # target (stationary)
    df['target'] = np.log(df['global_oil_price']).diff().shift(-1)

    df = df.dropna().copy()

    return df


def get_features(df):

    features = [col for col in df.columns if col != 'target']

    # remove high correlation
    corr = df[features].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]

    features = [f for f in features if f not in to_drop]

    return features


def train_model(df):

    df = prepare_data(df)
    features = get_features(df)

    X = df[features]
    y = df['target']

    model = get_model()

    # =========================
    # CROSS VALIDATION
    # =========================
    tscv = TimeSeriesSplit(n_splits=5)

    maes = []
    rmses = []

    for train_idx, test_idx in tscv.split(df):

        train, test = df.iloc[train_idx], df.iloc[test_idx]

        X_train = train[features]
        y_train = train['target']

        X_test = test[features]
        y_test = test['target']

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        maes.append(mean_absolute_error(y_test, preds))
        rmses.append(np.sqrt(mean_squared_error(y_test, preds)))

    print("\n===== CROSS VALIDATION =====")
    print("MAE:", np.mean(maes))
    print("RMSE:", np.mean(rmses))

    # final training
    model.fit(X, y)

    return model, features, df