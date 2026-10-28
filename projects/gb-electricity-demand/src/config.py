"""Shared paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DEMAND_RAW = RAW / "demand.parquet"
NESO_RAW = RAW / "neso_forecast.parquet"
WEATHER_RAW = RAW / "weather.parquet"
CLEAN = ROOT / "data" / "processed" / "clean.parquet"
WEATHER_CLEAN = ROOT / "data" / "processed" / "weather_halfhourly.parquet"
NESO_CLEAN = ROOT / "data" / "processed" / "neso_daily.parquet"
DASHBOARD = ROOT / "data" / "processed" / "dashboard.json"
PREDS = ROOT / "data" / "processed" / "test_predictions.parquet"
MODEL = ROOT / "models" / "model.joblib"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
METRICS = REPORTS / "metrics.json"
QUALITY = REPORTS / "data_quality.json"

API = "https://api.neso.energy/api/3/action/datastore_search"
DEMAND_RESOURCES = {2022: "bb44a1b5-75b1-4db2-8491-257f23385006",
                    2023: "bf5ab335-9b40-4ea4-b93a-ab4af7bce003",
                    2024: "f6d02c0f-957b-48cb-82ee-09003f2ba759",
                    2025: "b2bde559-3455-4021-b179-dfe60c0337b0",
                    2026: "8a4a771c-3929-4e56-93ad-cdf13219dea5"}
NESO_FORECAST_RESOURCE = "9847e7bb-986e-49be-8138-717b25933fbb"
HIST_FORECAST_API = "https://historical-forecast-api.open-meteo.com/v1/forecast"
ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"
START = "2022-01-01"
TODAY = "2026-10-28"

# (name, lat, lon, population weight) -- approximate metro populations, normalised
CITIES = {"London": (51.51, -0.13, 9.0), "Birmingham": (52.49, -1.89, 2.9),
          "Manchester": (53.48, -2.24, 2.8), "Leeds": (53.80, -1.55, 1.9),
          "Glasgow": (55.86, -4.25, 1.8), "Bristol": (51.45, -2.59, 1.0)}
_tot = sum(v[2] for v in CITIES.values())
WEIGHTS = {k: v[2] / _tot for k, v in CITIES.items()}
WEATHER_VARS = ["temperature_2m", "shortwave_radiation", "wind_speed_10m", "cloud_cover"]

TRAIN_END, VAL_START, VAL_END, TEST_START = "2024-12-31", "2025-01-01", "2025-06-30", "2025-07-01"
SEED = 42
