from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .config import BEST_C2_RF_PARAMS, DEFAULT_LGBM_PARAMS, PRODUCT_FEATURE_COLS, ProjectPaths
from .data import DatasetBundle, load_or_prepare_transactions
from .features import (
    add_calendar_features,
    add_lag_features,
    attach_product_features,
    build_spine,
    build_or_load_feature_artifacts,
    make_weekly_actuals,
)
from .modeling import (
    LGBMRegressor,
    croston_predict_by_sku,
    fit_signedlog_model_with_params,
    predict_signedlog_model,
    signed_expm1,
    signed_log1p,
)


@dataclass
class ForecastOutput:
    product_id: str
    product_description: str
    country: str
    cluster: int
    cluster_label: str
    selected_model: str
    test_mape_selected: float | None
    warning_flag: str
    recommendation: str
    statistics: dict[str, Any]
    recent_12_weeks: list[dict[str, Any]]
    forecast_12_weeks: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "product_id": self.product_id,
            "product_description": self.product_description,
            "country": self.country,
            "cluster": self.cluster,
            "cluster_label": self.cluster_label,
            "selected_model": self.selected_model,
            "test_mape_selected": self.test_mape_selected,
            "warning_flag": self.warning_flag,
            "recommendation": self.recommendation,
            "statistics": self.statistics,
            "recent_12_weeks": self.recent_12_weeks,
            "forecast_12_weeks": self.forecast_12_weeks,
        }


def _normalize_country(country: str) -> str:
    return "ALL" if str(country).strip().upper() == "ALL" else str(country).strip()


def _business_guidance(cluster_label: str) -> tuple[str, str]:
    label = cluster_label.lower()
    if "steady regulars" in label:
        return "normal", "Automation-ready; use forecast for replenishment planning."
    if "volatile" in label:
        return "medium", "Use for short-cycle planning; trust the near term more than later weeks."
    if "cancellation" in label:
        return "review", "Directional only; review manually before acting because returns can distort net demand."
    if "intermittent" in label:
        return "low", "Low-confidence segment; use as a safety-stock signal rather than direct automation."
    return "low", "Very low-confidence segment; do not automate and prefer manual review or fixed buffers."


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if pd.isna(value):
        return None
    return value


class ForecastingPipeline:
    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths
        self.dataset: DatasetBundle = load_or_prepare_transactions(paths)
        self.feature_artifacts = build_or_load_feature_artifacts(self.dataset, paths, rebuild=False)
        self.feat_df_all = self.feature_artifacts.feat_df_all.copy()
        self.selection_df = pd.read_csv(paths.selection_summary_csv)
        self.best_model_payload = json.loads(paths.best_model_params_json.read_text(encoding="utf-8"))
        self.total_retail = self.dataset.total_retail.copy()
        self.full_weeks = self.dataset.all_weeks
        self.last_history_week = self.full_weeks.max()
        self._cluster_training_cache: dict[int, tuple[pd.DataFrame, list[str]]] = {}
        self._cluster_model_cache: dict[int, tuple[str, Any, list[str], dict[str, Any]]] = {}

    def _get_cluster_config(self, cluster_id: int) -> dict[str, Any]:
        for config in self.best_model_payload["cluster_configs"]:
            if int(config["cluster"]) == int(cluster_id):
                return config
        raise KeyError(f"No cluster config found for cluster {cluster_id}")

    def _build_cluster_training_panel(self, cluster_id: int) -> tuple[pd.DataFrame, list[str]]:
        if cluster_id in self._cluster_training_cache:
            return self._cluster_training_cache[cluster_id]

        cluster_products = self.feat_df_all[self.feat_df_all["cluster"] == cluster_id].index.astype(str).tolist()
        cluster_tx = self.total_retail[self.total_retail["StockCode"].isin(cluster_products)].copy()

        weekly_actuals = make_weekly_actuals(cluster_tx, drop_cancellations=False)
        panel = build_spine(cluster_products, self.full_weeks)
        panel = panel.merge(weekly_actuals, on=["StockCode", "week"], how="left")
        panel["sales"] = panel["sales"].fillna(0.0)
        panel = add_calendar_features(panel)
        panel = attach_product_features(panel, self.feat_df_all, [c for c in PRODUCT_FEATURE_COLS if c in self.feat_df_all.columns])
        panel = add_lag_features(panel)
        train_panel = panel.dropna(subset=["lag_52w"]).copy()

        exclude = {
            "StockCode",
            "week",
            "week_start",
            "week_key",
            "sales",
            "cluster",
            "cluster_label",
            "is_sporadic",
            "is_croston",
            "is_active",
        }
        feature_cols = [c for c in train_panel.columns if c not in exclude]
        self._cluster_training_cache[cluster_id] = (train_panel.reset_index(drop=True), feature_cols)
        return self._cluster_training_cache[cluster_id]

    def _train_cluster_model(self, cluster_id: int) -> tuple[str, Any, list[str], dict[str, Any]]:
        if cluster_id in self._cluster_model_cache:
            return self._cluster_model_cache[cluster_id]

        cluster_config = self._get_cluster_config(cluster_id)
        selected_model = str(cluster_config["selected_model"])
        params = dict(cluster_config.get("params", {}))
        train_panel, feature_cols = self._build_cluster_training_panel(cluster_id)

        if selected_model == "RF_Default":
            model = RandomForestRegressor(**params)
            model.fit(train_panel[feature_cols].fillna(0.0), signed_log1p(train_panel["sales"].values))
        elif selected_model == "RF_C2_BEST":
            effective_params = params or BEST_C2_RF_PARAMS
            model = RandomForestRegressor(**effective_params)
            model.fit(train_panel[feature_cols].fillna(0.0), signed_log1p(train_panel["sales"].values))
        elif selected_model in {"LGBM_Default", "LGBM_Tuned"}:
            effective_params = params or DEFAULT_LGBM_PARAMS
            model = fit_signedlog_model_with_params(train_panel, feature_cols, effective_params)
        elif selected_model == "CrostonSBA":
            model = None
        else:
            raise NotImplementedError(f"Forecast pipeline does not yet support selected model: {selected_model}")

        self._cluster_model_cache[cluster_id] = (selected_model, model, feature_cols, params)
        return self._cluster_model_cache[cluster_id]

    def _build_product_history_panel(self, product_id: str, country: str, horizon: int) -> pd.DataFrame:
        product_all = self.total_retail[self.total_retail["StockCode"] == product_id].copy()
        if country != "ALL":
            product_history = product_all[product_all["Country"] == country].copy()
        else:
            product_history = product_all.copy()
        if product_history.empty:
            raise ValueError(f"No sales data found for product {product_id} in country filter '{country}'")

        future_weeks = pd.period_range(start=self.last_history_week + 1, periods=horizon, freq="W")
        all_weeks = self.full_weeks.append(future_weeks)
        weekly_actuals = make_weekly_actuals(product_history, drop_cancellations=False)

        panel = build_spine([product_id], all_weeks)
        panel = panel.merge(weekly_actuals, on=["StockCode", "week"], how="left")
        history_mask = panel["week"].isin(self.full_weeks)
        panel.loc[history_mask, "sales"] = panel.loc[history_mask, "sales"].fillna(0.0)
        panel = add_calendar_features(panel)
        panel = attach_product_features(panel, self.feat_df_all, [c for c in PRODUCT_FEATURE_COLS if c in self.feat_df_all.columns])
        return panel

    def _predict_row(self, selected_model: str, model: Any, row: pd.DataFrame, feature_cols: list[str], history_panel: pd.DataFrame) -> float:
        if selected_model == "CrostonSBA":
            history_rows = history_panel[history_panel["sales"].notna()].copy()
            pred = croston_predict_by_sku(history_rows, row[["StockCode", "week"]].assign(sales=np.nan), alpha=0.1)[0]
            return float(max(0.0, pred))

        X = row[feature_cols].fillna(0.0)
        if selected_model in {"RF_Default", "RF_C2_BEST"}:
            pred = signed_expm1(model.predict(X))[0]
        elif selected_model in {"LGBM_Default", "LGBM_Tuned"}:
            pred = predict_signedlog_model(model, row, feature_cols)[0]
        else:
            raise NotImplementedError(f"Unsupported pipeline prediction method: {selected_model}")
        return float(max(0.0, pred))

    def forecast_product(self, product_id: str, country: str = "United Kingdom", horizon: int = 12) -> ForecastOutput:
        product_id = str(product_id)
        country = _normalize_country(country)

        if product_id not in self.total_retail["StockCode"].astype(str).values:
            raise ValueError(f"Product {product_id} not found in the cleaned retail dataset")
        if product_id not in self.feat_df_all.index.astype(str):
            raise ValueError(f"Product {product_id} is missing from the clustering feature artifacts")

        assigned_cluster = int(self.feat_df_all.loc[product_id, "cluster"])
        cluster_label = str(self.feat_df_all.loc[product_id, "cluster_label"])
        selected_model, model, feature_cols, _params = self._train_cluster_model(assigned_cluster)
        cluster_row = self.selection_df[self.selection_df["cluster"] == assigned_cluster].iloc[0]

        product_all = self.total_retail[self.total_retail["StockCode"] == product_id].copy()
        if country != "ALL":
            product_filtered = product_all[product_all["Country"] == country].copy()
        else:
            product_filtered = product_all.copy()
        if product_filtered.empty:
            available_countries = sorted(product_all["Country"].astype(str).unique().tolist())
            raise ValueError(
                f"No sales data for product {product_id} in {country}. Available countries: {', '.join(available_countries[:10])}"
            )

        panel = self._build_product_history_panel(product_id, country, horizon)
        future_weeks = pd.period_range(start=self.last_history_week + 1, periods=horizon, freq="W")

        for future_week in future_weeks:
            panel = add_lag_features(panel)
            row_mask = panel["week"] == future_week
            row = panel.loc[row_mask].copy()
            pred = self._predict_row(selected_model, model, row, feature_cols, panel)
            panel.loc[row_mask, "sales"] = pred

        weekly_history = (
            product_filtered.assign(Week=product_filtered["InvoiceDate"].dt.to_period("W"))
            .groupby("Week")["Sales"]
            .sum()
            .reset_index()
            .rename(columns={"Week": "week", "Sales": "sales"})
        )
        weekly_history["week_start"] = weekly_history["week"].apply(lambda p: p.start_time.normalize())

        product_desc = str(product_all["Description"].iloc[0]) if len(product_all) else "Unknown"
        total_sales = float(weekly_history["sales"].sum())
        avg_weekly_sales = float(weekly_history["sales"].mean()) if len(weekly_history) else 0.0
        weeks_with_sales = int((weekly_history["sales"] > 0).sum())
        pct_zero_weeks = float(1.0 - (weeks_with_sales / len(weekly_history))) if len(weekly_history) else 1.0
        recent_history = weekly_history.tail(12).copy()
        recent_avg = float(recent_history["sales"].mean()) if len(recent_history) else 0.0

        future_rows = panel[panel["week"].isin(future_weeks)][["week", "sales"]].copy()
        future_rows["week_start"] = future_rows["week"].apply(lambda p: p.start_time.normalize())
        forecast_records = [
            {
                "week": str(row["week"]),
                "week_start": str(pd.Timestamp(row["week_start"]).date()),
                "forecast_sales": round(float(row["sales"]), 2),
            }
            for _, row in future_rows.iterrows()
        ]

        recent_records = [
            {
                "week": str(row["week"]),
                "week_start": str(pd.Timestamp(row["week_start"]).date()),
                "sales": round(float(row["sales"]), 2),
            }
            for _, row in recent_history.iterrows()
        ]

        warning_flag, recommendation = _business_guidance(cluster_label)
        statistics = {
            "total_sales": round(total_sales, 2),
            "avg_weekly_sales": round(avg_weekly_sales, 2),
            "recent_12week_avg": round(recent_avg, 2),
            "weeks_with_sales": weeks_with_sales,
            "total_weeks": int(len(weekly_history)),
            "pct_zero_weeks": round(pct_zero_weeks * 100, 2),
            "total_transactions": int(len(product_filtered)),
            "unique_customers": int(product_filtered["CustomerID"].nunique()),
        }

        return ForecastOutput(
            product_id=product_id,
            product_description=product_desc,
            country=country,
            cluster=assigned_cluster,
            cluster_label=cluster_label,
            selected_model=selected_model,
            test_mape_selected=_json_safe_value(cluster_row["test_mape_selected"]),
            warning_flag=warning_flag,
            recommendation=recommendation,
            statistics=statistics,
            recent_12_weeks=recent_records,
            forecast_12_weeks=forecast_records,
        )

    def save_forecast(self, forecast: ForecastOutput, output_path: str | Path | None = None) -> Path:
        if output_path is None:
            safe_country = forecast.country.replace(" ", "_")
            output_path = self.paths.pipeline_dir / f"forecast_{forecast.product_id}_{safe_country}.json"
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(forecast.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return out_path
