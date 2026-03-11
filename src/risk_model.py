"""
FX Risk Forecasting Model — Module 3
======================================
Estimates exchange rate volatility and outputs risk scores per currency pair.

Pipeline:
    FX dataset → preprocessing → GARCH/rolling-vol model → risk score [0,1]

We implement two models:
  1. Rolling Realized Volatility (fast, interpretable — default for prototype)
  2. AR(1) mean-reversion model with volatility clustering (ARIMA-lite)

Risk score formula:
    risk_score = clip( realized_vol_annualized / vol_ceiling, 0, 1 )
    where vol_ceiling = 0.30 (30% annualized vol = maximum risk score of 1.0)

This mirrors the VaR-style risk normalization used in FX trading desks.
"""

import numpy as np
import pandas as pd
import pickle
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOL_CEILING = 0.30          # 30% annualized vol → risk score = 1.0
TRADING_HOURS_PER_YEAR = 365 * 24  # hourly data
EWMA_HALFLIFE_HOURS = 24    # EWMA half-life for volatility estimate

# Currency pair categories for risk bucketing
RISK_TIERS = {
    "MAJOR":   ["USD/EUR", "EUR/USD", "USD/GBP", "GBP/USD"],
    "CROSS":   ["EUR/GBP", "GBP/EUR", "USD/SGD", "SGD/USD",
                "EUR/SGD", "SGD/EUR", "USD/AED", "AED/USD"],
    "EM":      ["USD/INR", "INR/USD", "EUR/INR", "INR/EUR",
                "GBP/INR", "INR/GBP", "AED/INR", "INR/AED",
                "SGD/INR", "INR/SGD"],
    "MANAGED": ["USD/CNY", "CNY/USD", "SGD/CNY", "CNY/SGD",
                "AED/CNY", "CNY/AED"],
}

TIER_RISK_FLOOR = {
    "MAJOR": 0.02,
    "CROSS": 0.04,
    "EM":    0.08,
    "MANAGED": 0.06,
}


def _get_tier(pair: str) -> str:
    for tier, pairs in RISK_TIERS.items():
        if pair in pairs:
            return tier
    return "CROSS"


# ---------------------------------------------------------------------------
# Data Preprocessor
# ---------------------------------------------------------------------------

class FXDataPreprocessor:
    """Loads and prepares FX time-series data for modeling."""

    def __init__(self, data_path: str = "../data/fx_rates.csv"):
        self.data_path = data_path
        self.data: Optional[pd.DataFrame] = None

    def load(self) -> "FXDataPreprocessor":
        self.data = pd.read_csv(self.data_path, parse_dates=["timestamp"])
        self.data.sort_values(["pair", "timestamp"], inplace=True)
        return self

    def get_pair_series(self, pair: str) -> pd.Series:
        """Return the price series for a given currency pair."""
        if self.data is None:
            raise RuntimeError("Call .load() first")
        mask = self.data["pair"] == pair
        series = self.data.loc[mask].set_index("timestamp")["rate"]
        return series

    def compute_log_returns(self, series: pd.Series) -> pd.Series:
        """Compute log returns: r_t = ln(P_t / P_{t-1})."""
        return np.log(series / series.shift(1)).dropna()

    def available_pairs(self) -> list:
        if self.data is None:
            return []
        return self.data["pair"].unique().tolist()


# ---------------------------------------------------------------------------
# Volatility Models
# ---------------------------------------------------------------------------

@dataclass
class RiskScore:
    pair: str
    realized_vol_hourly: float
    realized_vol_annualized: float
    ewma_vol_annualized: float
    risk_score: float           # [0, 1]
    risk_tier: str
    vol_regime: str             # "low" / "medium" / "high"


class RollingVolatilityModel:
    """
    Realized volatility model using rolling window and EWMA.

    Rationale:
      - Realized vol (σ_r) = std of log returns over window, annualized
      - EWMA vol (σ_e) gives more weight to recent observations
      - Final risk score is EWMA-based (more responsive to recent shocks)
    """

    def __init__(self, window_hours: int = 168):  # 1-week window
        self.window = window_hours

    def fit_and_score(self, returns: pd.Series, pair: str) -> RiskScore:
        """Compute volatility and risk score for a return series."""
        if len(returns) < self.window:
            # Fallback for short series
            vol_hourly = returns.std()
        else:
            vol_hourly = returns.rolling(self.window).std().iloc[-1]

        # Annualize: σ_annual = σ_hourly * √(trading_hours)
        vol_annual = vol_hourly * np.sqrt(TRADING_HOURS_PER_YEAR)

        # EWMA volatility (more responsive)
        ewma_var = returns.ewm(halflife=EWMA_HALFLIFE_HOURS).var().iloc[-1]
        ewma_vol_annual = np.sqrt(ewma_var * TRADING_HOURS_PER_YEAR)

        # Risk score: clipped to [0, 1]
        tier = _get_tier(pair)
        floor = TIER_RISK_FLOOR.get(tier, 0.04)
        raw_score = max(ewma_vol_annual / VOL_CEILING, floor)
        risk_score = min(raw_score, 1.0)

        # Classify vol regime (useful for stress testing alerts)
        if ewma_vol_annual < 0.05:
            regime = "low"
        elif ewma_vol_annual < 0.15:
            regime = "medium"
        else:
            regime = "high"

        return RiskScore(
            pair=pair,
            realized_vol_hourly=vol_hourly,
            realized_vol_annualized=vol_annual,
            ewma_vol_annualized=ewma_vol_annual,
            risk_score=risk_score,
            risk_tier=tier,
            vol_regime=regime,
        )


# ---------------------------------------------------------------------------
# AR(1) Mean-Reversion Forecaster (ARIMA-lite)
# ---------------------------------------------------------------------------

class ARForecastModel:
    """
    Simple AR(1) forecaster for FX rates:
        r_t = μ + φ·r_{t-1} + ε_t
    where φ is the autoregressive coefficient estimated by OLS.

    Outputs a 24-hour volatility forecast to augment the rolling model.
    """

    def __init__(self):
        self.params: Dict[str, Tuple[float, float, float]] = {}  # pair: (mu, phi, sigma)

    def fit(self, returns: pd.Series, pair: str):
        """Estimate AR(1) parameters by OLS."""
        y = returns.values[1:]
        X = returns.values[:-1]
        # OLS: phi = cov(X,y) / var(X)
        phi = np.cov(X, y)[0, 1] / np.var(X) if np.var(X) > 0 else 0.0
        mu  = np.mean(y) - phi * np.mean(X)
        # Residual std
        y_hat = mu + phi * X
        sigma = np.std(y - y_hat)
        self.params[pair] = (mu, phi, sigma)

    def forecast_vol(self, pair: str, horizon_hours: int = 24) -> float:
        """
        Compute h-step-ahead conditional variance for AR(1):
            σ²(h) = σ²_ε · (1 - φ^{2h}) / (1 - φ²)   [for |φ| < 1]
        Returns annualized volatility forecast.
        """
        if pair not in self.params:
            return 0.1  # default fallback
        mu, phi, sigma = self.params[pair]
        if abs(phi) >= 1:
            phi = 0.95  # clamp to stationary
        h = horizon_hours
        if abs(phi) < 1e-6:
            cond_var = sigma**2 * h
        else:
            cond_var = sigma**2 * (1 - phi**(2*h)) / (1 - phi**2)
        annualized = np.sqrt(cond_var * TRADING_HOURS_PER_YEAR / h)
        return annualized


# ---------------------------------------------------------------------------
# Risk Model Orchestrator
# ---------------------------------------------------------------------------

class FXRiskModel:
    """
    Main risk model: combines rolling vol + AR(1) forecast.
    Produces a risk score dictionary for the ClearingEngine.
    """

    def __init__(self, data_path: str = "../data/fx_rates.csv"):
        self.preprocessor = FXDataPreprocessor(data_path)
        self.rolling_model = RollingVolatilityModel()
        self.ar_model      = ARForecastModel()
        self.risk_scores: Dict[str, float] = {}
        self.risk_details: Dict[str, RiskScore] = {}

    def fit(self, pairs: Optional[list] = None) -> "FXRiskModel":
        """
        Fit volatility models for all (or selected) currency pairs.
        Stores risk scores for use by ClearingEngine.
        """
        self.preprocessor.load()
        available = self.preprocessor.available_pairs()
        target_pairs = pairs if pairs else available

        for pair in target_pairs:
            try:
                series  = self.preprocessor.get_pair_series(pair)
                returns = self.preprocessor.compute_log_returns(series)
                if len(returns) < 10:
                    continue

                # Rolling vol score
                rs = self.rolling_model.fit_and_score(returns, pair)

                # AR model forecast
                self.ar_model.fit(returns, pair)
                ar_vol = self.ar_model.forecast_vol(pair, horizon_hours=24)

                # Blend: 60% EWMA realized, 40% AR forecast
                blended_vol = 0.60 * rs.ewma_vol_annualized + 0.40 * ar_vol
                blended_score = min(max(blended_vol / VOL_CEILING, TIER_RISK_FLOOR.get(rs.risk_tier, 0.04)), 1.0)

                # Update risk score with blended value
                rs.risk_score = blended_score
                self.risk_scores[pair] = blended_score
                self.risk_details[pair] = rs

            except Exception as e:
                print(f"Warning: could not model {pair}: {e}")

        print(f"Fitted risk model for {len(self.risk_scores)} currency pairs")
        return self

    def get_risk_score(self, src: str, tgt: str) -> float:
        pair = f"{src}/{tgt}"
        return self.risk_scores.get(pair, 0.10)

    def get_risk_scores_dict(self) -> Dict[str, float]:
        return dict(self.risk_scores)

    def get_risk_heatmap_data(self) -> pd.DataFrame:
        """Return risk data formatted for heatmap visualization."""
        rows = []
        for pair, rs in self.risk_details.items():
            src, tgt = pair.split("/")
            rows.append({
                "pair": pair,
                "source": src,
                "target": tgt,
                "risk_score": rs.risk_score,
                "vol_annualized": rs.ewma_vol_annualized,
                "vol_regime": rs.vol_regime,
                "risk_tier": rs.risk_tier,
            })
        return pd.DataFrame(rows)

    def save(self, path: str = "../models/fx_model.pkl"):
        """Persist fitted model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"risk_scores": self.risk_scores, "risk_details": self.risk_details}, f)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str = "../models/fx_model.pkl") -> "FXRiskModel":
        """Load a previously fitted model."""
        obj = cls.__new__(cls)
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj.risk_scores   = data["risk_scores"]
        obj.risk_details  = data["risk_details"]
        obj.preprocessor  = FXDataPreprocessor()
        obj.rolling_model = RollingVolatilityModel()
        obj.ar_model      = ARForecastModel()
        return obj


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os, sys
    os.chdir(os.path.dirname(__file__))

    model = FXRiskModel("../data/fx_rates.csv").fit()
    model.save("../models/fx_model.pkl")

    print("\n=== Top 10 Riskiest Currency Pairs ===")
    df = model.get_risk_heatmap_data().sort_values("risk_score", ascending=False)
    print(df[["pair", "risk_score", "vol_annualized", "vol_regime", "risk_tier"]].head(10).to_string(index=False))

    print(f"\nINR/USD Risk Score: {model.get_risk_score('INR','USD'):.4f}")
    print(f"USD/EUR Risk Score: {model.get_risk_score('USD','EUR'):.4f}")
