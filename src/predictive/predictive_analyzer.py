"""
Predictive analysis utilities for YouTube community data.

This module provides simple, explainable forecasting using linear regression
on time-series features (e.g., engagement over time, community size over time).
It's designed to be lightweight and dependency-friendly (scikit-learn).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Literal
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


@dataclass
class ForecastResult:
    metric_name: str
    horizon: int
    last_observed: float
    forecast: List[float]
    growth_rate: float
    r2: Optional[float]
    timestamps: List[str]


class PredictiveAnalyzer:
    """
    Provides basic predictive capabilities:
    - Forecast engagement metrics over time
    - Forecast community size over time
    - Forecast influencer (composite) scores over time

    Approach:
    - Aggregate time series by a chosen frequency (D/W/M)
    - Fit a simple Linear Regression on time index vs. metric
    - Produce next-horizon predictions
    """

    def __init__(self, freq: Literal["D", "W", "M"] = "D"):
        """
        Args:
            freq: Resample frequency ('D' daily, 'W' weekly, 'M' monthly)
        """
        self.freq = freq

    def _prepare_series(
        self, df: pd.DataFrame, time_col: str, value_col: str
    ) -> pd.Series:
        """Aggregate and resample a time-series column."""
        ts = (
            df[[time_col, value_col]]
            .dropna()
            .assign(**{time_col: lambda x: pd.to_datetime(x[time_col])})
            .set_index(time_col)[value_col]
            .resample(self.freq)
            .sum()
        )
        return ts

    def _fit_linear_forecast(
        self, ts: pd.Series, horizon: int
    ) -> ForecastResult:
        """Fit linear regression on index vs. values and forecast future points."""
        if len(ts) < 3:
            raise ValueError("Need at least 3 data points for forecasting.")

        # Prepare training data
        ts = ts.sort_index()
        X = np.arange(len(ts)).reshape(-1, 1)
        y = ts.values

        model = LinearRegression()
        model.fit(X, y)

        # Predict future
        future_idx = np.arange(len(ts), len(ts) + horizon).reshape(-1, 1)
        y_pred = model.predict(future_idx)

        # Growth rate (slope)
        growth_rate = float(model.coef_[0])

        # R^2 score
        r2 = float(model.score(X, y))

        # Build timestamps for forecast
        last_ts = ts.index[-1]
        future_dates = pd.date_range(last_ts, periods=horizon + 1, freq=self.freq)[1:]
        timestamps = [d.isoformat() for d in future_dates]

        return ForecastResult(
            metric_name=str(ts.name),
            horizon=horizon,
            last_observed=float(ts.iloc[-1]),
            forecast=[float(v) for v in y_pred],
            growth_rate=growth_rate,
            r2=r2,
            timestamps=timestamps,
        )

    def forecast_metric(
        self,
        df: pd.DataFrame,
        time_col: str,
        value_col: str,
        horizon: int = 7,
    ) -> ForecastResult:
        """
        Generic metric forecast.

        Args:
            df: DataFrame with time and value columns
            time_col: Timestamp column
            value_col: Metric column to forecast
            horizon: Number of future periods to predict
        """
        ts = self._prepare_series(df, time_col, value_col)
        ts.name = value_col
        return self._fit_linear_forecast(ts, horizon)

    def forecast_engagement(
        self,
        comments: List[Dict],
        horizon: int = 7,
        time_field: str = "published_at",
        value_field: str = "like_count",
    ) -> ForecastResult:
        """
        Forecast engagement (e.g., likes or replies) over time.

        Args:
            comments: List of comment dicts with timestamps and engagement fields
            horizon: Number of future periods to predict
            time_field: Timestamp field in comments (e.g., published_at)
            value_field: Engagement field to sum (e.g., like_count or reply_count)
        """
        df = pd.DataFrame(comments)
        if time_field not in df.columns or value_field not in df.columns:
            raise ValueError(f"Comments must include '{time_field}' and '{value_field}'")
        return self.forecast_metric(df, time_field, value_field, horizon)

    def forecast_community_size(
        self,
        snapshots: List[Dict],
        horizon: int = 4,
        time_field: str = "snapshot_date",
        value_field: str = "community_size",
    ) -> ForecastResult:
        """
        Forecast community size over time from snapshots.

        Args:
            snapshots: List of dicts with time_field and value_field
            horizon: Number of future periods to predict
            time_field: Timestamp field (e.g., snapshot_date)
            value_field: Metric field (e.g., community_size)
        """
        df = pd.DataFrame(snapshots)
        if time_field not in df.columns or value_field not in df.columns:
            raise ValueError(f"Snapshots must include '{time_field}' and '{value_field}'")
        return self.forecast_metric(df, time_field, value_field, horizon)

    def forecast_influencer_score(
        self,
        influencer_history: List[Dict],
        horizon: int = 4,
        time_field: str = "timestamp",
        value_field: str = "composite_score",
    ) -> ForecastResult:
        """
        Forecast influencer composite scores over time.

        Args:
            influencer_history: List of dicts with time_field and value_field
            horizon: Number of future periods to predict
            time_field: Timestamp field (e.g., timestamp)
            value_field: Metric field (e.g., composite_score)
        """
        df = pd.DataFrame(influencer_history)
        if time_field not in df.columns or value_field not in df.columns:
            raise ValueError(f"Influencer history must include '{time_field}' and '{value_field}'")
        return self.forecast_metric(df, time_field, value_field, horizon)

