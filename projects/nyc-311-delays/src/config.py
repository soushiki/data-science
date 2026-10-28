"""Shared paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "311_2026.parquet"
CLEAN = ROOT / "data" / "processed" / "clean.parquet"
DASHBOARD = ROOT / "data" / "processed" / "dashboard.json"
MODEL = ROOT / "models" / "model.joblib"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
METRICS = REPORTS / "metrics.json"
QUALITY = REPORTS / "data_quality.json"
MODEL2 = ROOT / "models" / "model_study2.joblib"
PREDS2 = ROOT / "data" / "processed" / "test_predictions_study2.parquet"
THRESHOLDS2 = ROOT / "data" / "processed" / "study2_thresholds.parquet"
PREDS = ROOT / "data" / "processed" / "test_predictions.parquet"

START, END = "2026-03-01", "2026-09-01"
THRESHOLD_H = 72
SEED = 42
