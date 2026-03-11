"""
FX Rate Dataset Generator for CBDC Clearing Engine
Generates synthetic but realistic FX rate time series for research purposes.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# Base FX rates relative to USD (as of simulation baseline)
BASE_RATES = {
    "USD": 1.0,
    "EUR": 0.92,
    "GBP": 0.79,
    "INR": 83.5,
    "SGD": 1.34,
    "CNY": 7.24,
    "AED": 3.67,
}

CURRENCIES = list(BASE_RATES.keys())


def generate_fx_timeseries(days=365, freq="1h"):
    """
    Generate synthetic FX rate time series using Geometric Brownian Motion.
    GBM is the standard model for FX rate simulation in quantitative finance.

    dS = μS dt + σS dW
    where μ = drift, σ = volatility, dW = Wiener process increment
    """
    periods = days * 24  # hourly data
    dates = pd.date_range(end=datetime.now(), periods=periods, freq=freq)

    # Volatility parameters (annualized) per currency pair vs USD
    volatility = {
        "EUR": 0.07,
        "GBP": 0.09,
        "INR": 0.05,
        "SGD": 0.04,
        "CNY": 0.03,  # managed float - lower vol
        "AED": 0.005,  # pegged - very low vol
    }

    drift = {
        "EUR": -0.01,
        "GBP": -0.02,
        "INR": 0.03,  # mild depreciation trend
        "SGD": -0.01,
        "CNY": 0.01,
        "AED": 0.0,
    }

    records = []
    dt = 1 / (365 * 24)  # hourly time step

    for ccy, base_rate in BASE_RATES.items():
        if ccy == "USD":
            rates = np.ones(periods)
        else:
            vol = volatility.get(ccy, 0.06)
            mu = drift.get(ccy, 0.0)
            # GBM simulation
            rand_shocks = np.random.normal(0, 1, periods)
            log_returns = (mu - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * rand_shocks
            price_path = base_rate * np.exp(np.cumsum(log_returns))
            rates = price_path

        for i, (date, rate) in enumerate(zip(dates, rates)):
            records.append(
                {
                    "timestamp": date,
                    "currency": ccy,
                    "rate_vs_usd": rate,
                    "currency_pair": f"{ccy}/USD",
                }
            )

    df = pd.DataFrame(records)

    # Also create cross-currency pairs
    cross_pairs = []
    pivoted = df.pivot(index="timestamp", columns="currency", values="rate_vs_usd")

    for base in CURRENCIES:
        for quote in CURRENCIES:
            if base != quote:
                cross_rate = pivoted[quote] / pivoted[base]
                pair_df = pd.DataFrame(
                    {
                        "timestamp": pivoted.index,
                        "base_currency": base,
                        "quote_currency": quote,
                        "rate": cross_rate.values,
                        "pair": f"{base}/{quote}",
                    }
                )
                cross_pairs.append(pair_df)

    cross_df = pd.concat(cross_pairs, ignore_index=True)
    cross_df.to_csv("fx_rates.csv", index=False)
    print(f"Generated {len(cross_df)} FX rate records across {len(CURRENCIES)**2 - len(CURRENCIES)} currency pairs")
    print(f"Date range: {cross_df['timestamp'].min()} to {cross_df['timestamp'].max()}")
    return cross_df


if __name__ == "__main__":
    df = generate_fx_timeseries(days=365)
    print(df.head(10))
