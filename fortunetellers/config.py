from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_EXCEL_NAME = "online_retail_II.xlsx"

NON_PRODUCT_ENTRIES = [
    "POSTAGE",
    "DOTCOM POSTAGE",
    "Manual",
    "CARRIAGE",
    "Discount",
    "SAMPLES",
    " Bank Charges",
    "AMAZON FEE",
    "ebay",
    "nan",
    "CRUK Commission",
    "This is a test product.",
    "Adjust bad debt",
    "Dotcomgiftshop Gift Voucher £10.00",
    "Dotcomgiftshop Gift Voucher £20.00",
    "Dotcomgiftshop Gift Voucher £30.00",
    "Dotcomgiftshop Gift Voucher £40.00",
    "Dotcomgiftshop Gift Voucher £50.00",
    "Dotcomgiftshop Gift Voucher £60.00",
    "Dotcomgiftshop Gift Voucher £70.00",
    "Dotcomgiftshop Gift Voucher £80.00",
    "Adjustment by john on 26/01/2010 16",
    "Adjustment by Peter on Jun 25 2010 ",
]

DEFAULT_LGBM_PARAMS = {
    "n_estimators": 700,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_child_samples": 40,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.2,
    "reg_lambda": 0.5,
}

LGBM_PARAM_GRID = [
    {
        "n_estimators": 500,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 40,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
    },
    {
        "n_estimators": 700,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_child_samples": 40,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
    },
    {
        "n_estimators": 900,
        "learning_rate": 0.02,
        "num_leaves": 63,
        "min_child_samples": 50,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.1,
        "reg_lambda": 0.2,
    },
    {
        "n_estimators": 1100,
        "learning_rate": 0.02,
        "num_leaves": 95,
        "min_child_samples": 60,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.2,
        "reg_lambda": 0.4,
    },
    {
        "n_estimators": 700,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 30,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
    },
    {
        "n_estimators": 900,
        "learning_rate": 0.03,
        "num_leaves": 47,
        "min_child_samples": 50,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 0.3,
    },
    {
        "n_estimators": 1200,
        "learning_rate": 0.015,
        "num_leaves": 63,
        "min_child_samples": 60,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.2,
        "reg_lambda": 0.5,
    },
    {
        "n_estimators": 800,
        "learning_rate": 0.025,
        "num_leaves": 79,
        "min_child_samples": 45,
        "subsample": 0.9,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.05,
        "reg_lambda": 0.2,
    },
]

BASELINE_RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 20,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
}

BEST_C2_RF_PARAMS = {
    "n_estimators": 250,
    "max_depth": 20,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
}

CLUSTERING_COLS = [
    "cv",
    "seasonal_conc",
    "q4_pct",
    "log_mean_price",
    "cancel_rate",
]

PRODUCT_FEATURE_COLS = [
    "cv",
    "pct_zero_weeks",
    "q4_pct",
    "q1_pct",
    "seasonal_conc",
    "log_mean_weekly_sales",
    "log_mean_price",
    "n_unique_customers",
    "cancel_rate",
    "trend_log_diff",
]

LAG_WINDOWS = [1, 2, 4, 8, 13, 26, 52]
ROLL_WINDOWS = [4, 8, 13, 26]
SPORADIC_THRESHOLD = 0.85
CROSTON_THRESHOLD = 0.50
CROSTON_ALPHA = 0.10
MIN_ABS_ACTUAL = 1.0


@dataclass
class ProjectPaths:
    raw_excel: Path = field(default_factory=lambda: default_raw_excel_path())
    artifact_root: Path = field(default_factory=lambda: PROJECT_ROOT / "fortunetellers_artifacts")

    processed_dir: Path = field(init=False)
    clustering_dir: Path = field(init=False)
    modeling_dir: Path = field(init=False)
    pipeline_dir: Path = field(init=False)
    agent_dir: Path = field(init=False)

    cleaned_retail_csv: Path = field(init=False)
    cleaned_products_csv: Path = field(init=False)
    product_features_full_csv: Path = field(init=False)
    product_features_scaled_csv: Path = field(init=False)
    product_features_clustered_csv: Path = field(init=False)
    cluster_metadata_json: Path = field(init=False)
    selection_summary_csv: Path = field(init=False)
    candidate_metrics_csv: Path = field(init=False)
    tuned_lgbm_trials_csv: Path = field(init=False)
    best_model_params_json: Path = field(init=False)

    def __post_init__(self) -> None:
        self.raw_excel = Path(self.raw_excel).expanduser().resolve()
        self.artifact_root = Path(self.artifact_root).expanduser().resolve()

        self.processed_dir = self.artifact_root / "processed"
        self.clustering_dir = self.artifact_root / "clustering"
        self.modeling_dir = self.artifact_root / "modeling"
        self.pipeline_dir = self.artifact_root / "pipeline"
        self.agent_dir = self.artifact_root / "agent"

        self.cleaned_retail_csv = self.processed_dir / "total_retail_cleaned.csv"
        self.cleaned_products_csv = self.processed_dir / "all_products_cleaned.csv"
        self.product_features_full_csv = self.clustering_dir / "product_features_full.csv"
        self.product_features_scaled_csv = self.clustering_dir / "product_features_scaled.csv"
        self.product_features_clustered_csv = self.clustering_dir / "product_features_clustered.csv"
        self.cluster_metadata_json = self.clustering_dir / "cluster_metadata.json"
        self.selection_summary_csv = self.modeling_dir / "cluster_model_selection_summary.csv"
        self.candidate_metrics_csv = self.modeling_dir / "cluster_candidate_metrics.csv"
        self.tuned_lgbm_trials_csv = self.modeling_dir / "cluster_tuned_lgbm_trials.csv"
        self.best_model_params_json = self.modeling_dir / "best_model_params.json"

    def ensure_dirs(self) -> None:
        for path in (
            self.artifact_root,
            self.processed_dir,
            self.clustering_dir,
            self.modeling_dir,
            self.pipeline_dir,
            self.agent_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def default_raw_excel_path() -> Path:
    env_path = os.getenv("FORTUNETELLERS_RAW_EXCEL")
    candidates = []
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(PROJECT_ROOT / "data" / "raw" / DEFAULT_RAW_EXCEL_NAME)
    candidates.append(Path.home() / "Downloads" / DEFAULT_RAW_EXCEL_NAME)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return (PROJECT_ROOT / "data" / "raw" / DEFAULT_RAW_EXCEL_NAME).resolve()
