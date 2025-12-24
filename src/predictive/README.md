# Predictive Analysis Module

Lightweight forecasting utilities for:
- Engagement metrics (likes, replies) over time
- Community size over time
- Influencer composite scores over time

## Design
- Uses simple, explainable linear regression (scikit-learn)
- Aggregates time-series data at a chosen frequency (`D`, `W`, `M`)
- Minimal dependencies; keeps forecasts easy to reason about

## Usage (Python)

```python
from src.predictive import PredictiveAnalyzer

# Example engagement history
history = [
    {"published_at": "2024-01-01", "like_count": 10},
    {"published_at": "2024-01-02", "like_count": 12},
    {"published_at": "2024-01-03", "like_count": 15},
]

predictive = PredictiveAnalyzer(freq="D")
forecast = predictive.forecast_engagement(
    history,
    horizon=7,
    time_field="published_at",
    value_field="like_count"
)

print(forecast)
```

## Methods
- `forecast_metric(df, time_col, value_col, horizon)`: generic metric forecast
- `forecast_engagement(comments, horizon, time_field, value_field)`
- `forecast_community_size(snapshots, horizon, time_field, value_field)`
- `forecast_influencer_score(influencer_history, horizon, time_field, value_field)`

## Output
Returns a `ForecastResult` dataclass:
```python
ForecastResult(
    metric_name: str,
    horizon: int,
    last_observed: float,
    forecast: List[float],
    growth_rate: float,
    r2: Optional[float],
    timestamps: List[str],
)
```

## Notes
- Requires at least 3 data points for forecasting.
- For more complex patterns (seasonality, non-linear trends), you can extend this module with additional models (e.g., Prophet, ARIMA). This implementation stays minimal by design.

