"""Shared paths, constants and business assumptions."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW_CSV = RAW / "hillstrom.csv"
CLEAN = ROOT / "data" / "processed" / "clean.parquet"
DASHBOARD = ROOT / "data" / "processed" / "dashboard.json"
CURVES = ROOT / "data" / "processed" / "uplift_curves.json"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
METRICS = REPORTS / "metrics.json"
QUALITY = REPORTS / "data_quality.json"

URL = ("http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_"
       "DataMiningChallenge_2008.03.20.csv")

SEED = 42
ALPHA, POWER = 0.05, 0.80
N_BOOT = 2000            # bootstrap resamples for lift CIs
N_BOOT_POLICY = 2000     # bootstrap resamples for policy CIs
N_SPLITS = 20            # random 70/30 splits for Qini stability (visit)
N_SPLITS_SPEND = 10      # ... for the secondary spend outcome
N_POLICY_SPLITS = 10     # splits on which policies are evaluated
TEST_SIZE = 0.30
K_GRID = [0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]   # % of customers emailed
CV_FOLDS = 3

# ---- Business assumptions (illustrative, NOT from the dataset) ----
EMAIL_COST = 0.10        # $ per email sent
GROSS_MARGIN = 0.30      # share of spend kept as gross profit

ARMS = {0: "No E-Mail", 1: "Mens E-Mail", 2: "Womens E-Mail"}
FEATURES = ["recency", "history", "mens", "womens", "newbie", "zip_suburban", "zip_urban",
            "chan_phone", "chan_web"]
# pre-period covariates allowed in regression adjustment
ADJ_COVARIATES = ["history", "recency", "mens", "womens", "newbie", "channel"]
# pre-specified heterogeneity segments (column -> display name)
SEGMENTS = {"mens": "Bought men's line", "womens": "Bought women's line", "newbie": "New customer",
            "channel": "Channel", "history_segment": "Past-year spend band", "zip_code": "Zip type"}
