"""
forecast_model.py
------------------
Two simple, transparent forecasting approaches applied to the bank
customer-service dataset:

1. Moving Average (7-day and 30-day) - smooths short-term noise.
2. Linear Regression (scikit-learn) - captures the underlying trend
   and projects it forward 30 days, using day-index + day-of-week
   as features.

This module is imported by dashboard/app.py. It can also be run
directly to print a quick forecast summary to the terminal.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def load_data(path="data/bank_service_data.csv"):
    df = pd.read_csv(path, parse_dates=["date"])
    df["day_index"] = np.arange(len(df))
    df["dow"] = df["date"].dt.weekday
    return df


def add_moving_averages(df, column="calls_received"):
    df = df.copy()
    df[f"{column}_ma7"] = df[column].rolling(7, min_periods=1).mean()
    df[f"{column}_ma30"] = df[column].rolling(30, min_periods=1).mean()
    return df


def train_regression(df, column="calls_received"):
    """Trains a linear regression on day_index + one-hot day-of-week."""
    X = pd.get_dummies(df[["day_index", "dow"]], columns=["dow"], drop_first=True)
    y = df[column]
    model = LinearRegression()
    model.fit(X, y)
    return model, X.columns


def forecast_future(df, model, feature_cols, days_ahead=30, column="calls_received"):
    last_index = df["day_index"].max()
    last_date = df["date"].max()

    future_rows = []
    for step in range(1, days_ahead + 1):
        future_date = last_date + pd.Timedelta(days=step)
        future_rows.append({"day_index": last_index + step, "dow": future_date.weekday(), "date": future_date})

    future_df = pd.DataFrame(future_rows)
    X_future = pd.get_dummies(future_df[["day_index", "dow"]], columns=["dow"], drop_first=True)
    X_future = X_future.reindex(columns=feature_cols, fill_value=0)

    future_df[f"{column}_forecast"] = model.predict(X_future)
    future_df[f"{column}_forecast"] = future_df[f"{column}_forecast"].clip(lower=0)
    return future_df[["date", f"{column}_forecast"]]


if __name__ == "__main__":
    df = load_data()
    df = add_moving_averages(df)
    model, cols = train_regression(df)
    future = forecast_future(df, model, cols, days_ahead=30)
    print("Next 30-day call volume forecast (first 10 rows):")
    print(future.head(10).to_string(index=False))
