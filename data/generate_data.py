"""
generate_data.py
----------------
Simulates 18 months of daily operational data for a bank's
Customer Service Centre (call centre + branch footfall + social
media engagement). Produces a realistic, seasonal, noisy dataset
that the forecasting model in /model/forecast_model.py will use.

Run:
    python data/generate_data.py

Output:
    data/bank_service_data.csv
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

START_DATE = datetime(2025, 1, 1)
NUM_DAYS = 18 * 30  # ~18 months of daily data

dates = [START_DATE + timedelta(days=i) for i in range(NUM_DAYS)]

rows = []
for i, date in enumerate(dates):
    day_of_week = date.weekday()          # 0=Mon ... 6=Sun
    month = date.month

    # --- Base call volume (customer service calls per day) ---
    base_calls = 420
    weekly_pattern = [1.15, 1.10, 1.05, 1.05, 1.20, 0.55, 0.35][day_of_week]  # banks quieter weekends
    monthly_seasonality = 1 + 0.18 * np.sin((month / 12) * 2 * np.pi)         # salary/season cycles
    yearly_growth = 1 + (i / NUM_DAYS) * 0.10                                 # gradual +10% growth
    noise = np.random.normal(0, 18)

    calls = max(0, base_calls * weekly_pattern * monthly_seasonality * yearly_growth + noise)

    # --- Average wait time (minutes), rises when call volume is high ---
    wait_time = 2.5 + (calls / 300) + np.random.normal(0, 0.4)
    wait_time = max(0.5, wait_time)

    # --- Branch foot traffic (correlated with calls, but weekday-shifted) ---
    foot_traffic = calls * 0.6 * (1.3 if day_of_week in [0, 4] else 1.0) + np.random.normal(0, 15)
    foot_traffic = max(0, foot_traffic)

    # --- Customer satisfaction score (CSAT, 1-100), drops when wait time is high ---
    csat = 92 - (wait_time * 2.2) + np.random.normal(0, 2.5)
    csat = min(100, max(40, csat))

    # --- Social media engagement (mentions + comments per day) ---
    social_engagement = 60 + 25 * np.sin((i / 30) * 2 * np.pi) + np.random.normal(0, 8)
    social_engagement = max(0, social_engagement)

    rows.append({
        "date": date.strftime("%Y-%m-%d"),
        "day_of_week": date.strftime("%A"),
        "calls_received": round(calls),
        "avg_wait_time_min": round(wait_time, 2),
        "branch_foot_traffic": round(foot_traffic),
        "csat_score": round(csat, 1),
        "social_engagement": round(social_engagement),
    })

df = pd.DataFrame(rows)
df.to_csv("data/bank_service_data.csv", index=False)
print(f"Generated {len(df)} days of data -> data/bank_service_data.csv")
