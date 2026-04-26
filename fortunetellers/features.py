from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product as iterproduct
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler

from .config import (
    CLUSTERING_COLS,
    CROSTON_THRESHOLD,
    LAG_WINDOWS,
    PRODUCT_FEATURE_COLS,
    ROLL_WINDOWS,
    SPORADIC_THRESHOLD,
    ProjectPaths,
)
from .data import DatasetBundle


@dataclass
class FeatureArtifacts:
    feat_df_all: pd.DataFrame
    feat_scaled: pd.DataFrame
    best_k: int
    silhouette_scores: dict[int, float]
    cluster_labels: dict[int, str]


def herfindahl(monthly_totals: pd.Series) -> float:
    total = monthly_totals.sum()
    if total == 0:
        return np.nan
    shares = monthly_totals / total
    return float((shares**2).sum())


def build_product_feature_table(retail_train: pd.DataFrame) -> pd.DataFrame:
    retail_mod = retail_train.copy()
    retail_mod["is_cancelled"] = retail_mod["Invoice"].astype(str).str.startswith("C")

    cancellations = retail_mod[retail_mod["is_cancelled"]].copy()
    sales = retail_mod[~retail_mod["is_cancelled"]].copy()

    sales["week"] = sales["InvoiceDate"].dt.to_period("W")
    sales["month"] = sales["InvoiceDate"].dt.month

    clustering_weeks = pd.period_range(
        start=sales["InvoiceDate"].min().to_period("W"),
        end=sales["InvoiceDate"].max().to_period("W"),
        freq="W",
    )
    total_weeks = len(clustering_weeks)

    weekly = (
        sales.groupby(["StockCode", "week"])["Sales"]
        .sum()
        .reset_index()
        .rename(columns={"Sales": "weekly_sales"})
    )
    weekly_pivot = (
        weekly.pivot(index="StockCode", columns="week", values="weekly_sales")
        .reindex(columns=clustering_weeks)
        .fillna(0.0)
    )

    features: list[dict[str, Any]] = []
    for stock_code, group in sales.groupby("StockCode"):
        weekly_sales = weekly_pivot.loc[stock_code]
        mean_weekly_sales = float(weekly_sales.mean())
        median_weekly_sales = float(weekly_sales.median())
        total_sales = float(weekly_sales.sum())
        mean_txn_size = float(group["Sales"].mean())

        std_weekly_sales = float(weekly_sales.std())
        cv = std_weekly_sales / mean_weekly_sales if mean_weekly_sales > 0 else np.nan
        spike_ratio = float(weekly_sales.max()) / mean_weekly_sales if mean_weekly_sales > 0 else np.nan

        zero_weeks = int((weekly_sales == 0).sum())
        pct_zero_weeks = zero_weeks / total_weeks

        txn_dates = group["InvoiceDate"].sort_values()
        if len(txn_dates) > 1:
            avg_idi = float(txn_dates.diff().dt.days.dropna().mean())
        else:
            avg_idi = np.nan

        current_streak = 0
        longest_zero_streak = 0
        for value in weekly_sales:
            if value == 0:
                current_streak += 1
                longest_zero_streak = max(longest_zero_streak, current_streak)
            else:
                current_streak = 0

        monthly_sales = group.groupby("month")["Sales"].sum().reindex(range(1, 13), fill_value=0.0)
        q4_pct = monthly_sales[[10, 11, 12]].sum() / total_sales if total_sales > 0 else np.nan
        q1_pct = monthly_sales[[1, 2, 3]].sum() / total_sales if total_sales > 0 else np.nan
        peak_month = int(monthly_sales.idxmax())
        seasonal_conc = herfindahl(monthly_sales)

        weekly_values = weekly_sales.values
        if len(weekly_values) > 1 and weekly_values.sum() > 0:
            trend_slope = float(linregress(np.arange(len(weekly_values)), weekly_values).slope)
        else:
            trend_slope = np.nan

        date_min = group["InvoiceDate"].min()
        date_max = group["InvoiceDate"].max()
        cutoff_early = date_min + pd.DateOffset(months=3)
        cutoff_late = date_max - pd.DateOffset(months=3)
        early_sales = float(group[group["InvoiceDate"] < cutoff_early]["Sales"].sum())
        late_sales = float(group[group["InvoiceDate"] > cutoff_late]["Sales"].sum())
        trend_log_diff = float(np.log1p(late_sales) - np.log1p(early_sales))

        mean_price = float(group["Price"].mean())
        std_price = float(group["Price"].std())
        n_unique_customers = int(group["CustomerID"].nunique())
        total_customers = int(group["CustomerID"].count())
        repeat_customers = total_customers - n_unique_customers
        pct_repeat = repeat_customers / total_customers if total_customers > 0 else np.nan
        n_countries = int(group["Country"].nunique())
        top_country_pct = float(group["Country"].value_counts(normalize=True).iloc[0]) if len(group) else np.nan

        cancelled_sales = float(cancellations[cancellations["StockCode"] == stock_code]["Sales"].abs().sum())
        cancel_rate = cancelled_sales / (total_sales + cancelled_sales) if (total_sales + cancelled_sales) > 0 else 0.0

        features.append(
            {
                "StockCode": stock_code,
                "mean_weekly_sales": mean_weekly_sales,
                "median_weekly_sales": median_weekly_sales,
                "total_sales": total_sales,
                "mean_txn_size": mean_txn_size,
                "std_weekly_sales": std_weekly_sales,
                "cv": cv,
                "spike_ratio": spike_ratio,
                "pct_zero_weeks": pct_zero_weeks,
                "avg_idi_days": avg_idi,
                "longest_zero_streak": longest_zero_streak,
                "q4_pct": q4_pct,
                "q1_pct": q1_pct,
                "peak_month": peak_month,
                "seasonal_conc": seasonal_conc,
                "trend_slope": trend_slope,
                "trend_log_diff": trend_log_diff,
                "mean_price": mean_price,
                "std_price": std_price,
                "n_unique_customers": n_unique_customers,
                "pct_repeat_customers": pct_repeat,
                "n_countries": n_countries,
                "top_country_pct": top_country_pct,
                "cancel_rate": cancel_rate,
            }
        )

    feat_df = pd.DataFrame(features).set_index("StockCode")
    feat_df["log_mean_weekly_sales"] = np.log1p(feat_df["mean_weekly_sales"])
    feat_df["log_mean_price"] = np.log1p(feat_df["mean_price"])

    for col in ("avg_idi_days", "longest_zero_streak", "trend_log_diff"):
        cap = feat_df[col].quantile(0.95)
        feat_df[col] = feat_df[col].clip(upper=cap)

    feat_df["is_sporadic"] = feat_df["pct_zero_weeks"] > SPORADIC_THRESHOLD
    feat_df["is_croston"] = (feat_df["pct_zero_weeks"] > CROSTON_THRESHOLD) & ~feat_df["is_sporadic"]
    feat_df["is_active"] = feat_df["pct_zero_weeks"] <= CROSTON_THRESHOLD
    return feat_df


def _auto_label(summary: pd.DataFrame, cluster_id: int) -> str:
    row = summary.loc[cluster_id]
    if row.get("cancel_rate", 0) > summary["cancel_rate"].quantile(0.75):
        return "High cancellation risk"
    if row.get("q4_pct", 0) > 0.35 and row.get("seasonal_conc", 0) > 0.18:
        return "Christmas / seasonal"
    if row.get("cv", 0) < summary["cv"].median() and row.get("cancel_rate", 0) < 0.05:
        return "Steady regulars"
    if row.get("cv", 0) >= summary["cv"].median():
        return "Volatile mid-range"
    return "Steady mid-range"


def build_feature_artifacts(
    retail_train: pd.DataFrame,
    paths: ProjectPaths,
    clustering_weeks: pd.PeriodIndex | None = None,
) -> FeatureArtifacts:
    feat_df = build_product_feature_table(retail_train)

    feat_df_sporadic = feat_df[feat_df["is_sporadic"]].copy()
    feat_df_croston = feat_df[feat_df["is_croston"]].copy()
    feat_df_active = feat_df[feat_df["is_active"]].copy()

    feat_df_sporadic["cluster"] = -2
    feat_df_croston["cluster"] = -1
    feat_df_sporadic["cluster_label"] = "Truly sporadic"
    feat_df_croston["cluster_label"] = "Intermittent (Croston)"

    feat_cluster = feat_df_active[CLUSTERING_COLS].fillna(feat_df_active[CLUSTERING_COLS].median())
    scaler = RobustScaler()
    feat_scaled = pd.DataFrame(
        scaler.fit_transform(feat_cluster),
        index=feat_cluster.index,
        columns=feat_cluster.columns,
    )

    inertias: dict[int, float] = {}
    silhouette_scores: dict[int, float] = {}
    models: dict[int, tuple[KMeans, np.ndarray]] = {}
    for k in range(3, 6):
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(feat_scaled.values)
        inertias[k] = float(model.inertia_)
        silhouette_scores[k] = float(silhouette_score(feat_scaled.values, labels))
        models[k] = (model, labels)

    best_k = max(silhouette_scores, key=silhouette_scores.get)
    _, best_labels = models[best_k]

    feat_df_active = feat_df_active.copy()
    feat_df_active.loc[feat_scaled.index, "cluster"] = best_labels

    summary = feat_df_active.groupby("cluster")[CLUSTERING_COLS].median()
    summary["n_products"] = feat_df_active.groupby("cluster").size()

    cluster_labels = {int(cid): _auto_label(summary, cid) for cid in summary.index}
    feat_df_active["cluster_label"] = feat_df_active["cluster"].map(cluster_labels)

    feat_df_all = pd.concat([feat_df_active, feat_df_croston, feat_df_sporadic]).sort_index()
    feat_df_all["cluster"] = feat_df_all["cluster"].astype(int)

    metadata = {
        "best_k": int(best_k),
        "silhouette_scores": {str(k): float(v) for k, v in silhouette_scores.items()},
        "inertias": {str(k): float(v) for k, v in inertias.items()},
        "cluster_counts": {str(int(k)): int(v) for k, v in feat_df_all["cluster"].value_counts().sort_index().items()},
        "cluster_labels": {str(int(k)): v for k, v in {**cluster_labels, -1: "Intermittent (Croston)", -2: "Truly sporadic"}.items()},
    }
    if clustering_weeks is not None and len(clustering_weeks) > 0:
        metadata["clustering_split"] = {
            "n_weeks": int(len(clustering_weeks)),
            "start_week": str(clustering_weeks.min()),
            "end_week": str(clustering_weeks.max()),
            "rule": "all_weeks[:-24] equivalent (all weeks except validation and test horizons)",
        }

    paths.ensure_dirs()
    feat_df.to_csv(paths.product_features_full_csv)
    feat_scaled.to_csv(paths.product_features_scaled_csv)
    feat_df_all.to_csv(paths.product_features_clustered_csv)
    paths.cluster_metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return FeatureArtifacts(
        feat_df_all=feat_df_all,
        feat_scaled=feat_scaled,
        best_k=best_k,
        silhouette_scores=silhouette_scores,
        cluster_labels=cluster_labels,
    )


def load_feature_artifacts(paths: ProjectPaths) -> FeatureArtifacts:
    feat_df_all = pd.read_csv(paths.product_features_clustered_csv, index_col="StockCode")
    feat_scaled = pd.read_csv(paths.product_features_scaled_csv, index_col="StockCode")

    bool_cols = ["is_sporadic", "is_croston", "is_active"]
    for col in bool_cols:
        if col in feat_df_all.columns:
            feat_df_all[col] = feat_df_all[col].astype(str).str.lower().map({"true": True, "false": False}).fillna(feat_df_all[col]).astype(bool)
    feat_df_all["cluster"] = feat_df_all["cluster"].astype(int)

    metadata = json.loads(paths.cluster_metadata_json.read_text(encoding="utf-8"))
    return FeatureArtifacts(
        feat_df_all=feat_df_all,
        feat_scaled=feat_scaled,
        best_k=int(metadata["best_k"]),
        silhouette_scores={int(k): float(v) for k, v in metadata["silhouette_scores"].items()},
        cluster_labels={int(k): v for k, v in metadata["cluster_labels"].items() if int(k) >= 0},
    )


def build_or_load_feature_artifacts(
    dataset: DatasetBundle,
    paths: ProjectPaths,
    rebuild: bool = False,
) -> FeatureArtifacts:
    if (
        not rebuild
        and paths.product_features_clustered_csv.exists()
        and paths.product_features_scaled_csv.exists()
        and paths.cluster_metadata_json.exists()
    ):
        return load_feature_artifacts(paths)
    return build_feature_artifacts(
        dataset.retail_clustering_train,
        paths,
        clustering_weeks=dataset.clustering_training_weeks,
    )


def make_weekly_actuals(transactions: pd.DataFrame, drop_cancellations: bool = True) -> pd.DataFrame:
    df = transactions.copy()
    if drop_cancellations and "Invoice" in df.columns:
        df = df[~df["Invoice"].astype(str).str.startswith("C")].copy()
    df["week"] = df["InvoiceDate"].dt.to_period("W")
    return (
        df.groupby(["StockCode", "week"])["Sales"]
        .sum()
        .reset_index()
        .rename(columns={"Sales": "sales"})
    )


def _weeks_to_christmas(ts: pd.Timestamp) -> float:
    xmas = pd.Timestamp(f"{ts.year}-12-25")
    diff = (xmas - ts).days / 7
    if diff < 0:
        xmas = pd.Timestamp(f"{ts.year + 1}-12-25")
        diff = (xmas - ts).days / 7
    return round(min(26, max(0, diff)), 2)


def _easter_dates(years: np.ndarray) -> set[pd.Timestamp]:
    try:
        import holidays
    except ImportError:
        return set()

    easter_days: set[pd.Timestamp] = set()
    for year in years:
        uk = holidays.UK(years=int(year))
        for date_value, name in uk.items():
            if "Easter" not in name:
                continue
            ts = pd.Timestamp(date_value)
            for offset in range(-3, 4):
                easter_days.add(pd.Timestamp(ts + pd.Timedelta(days=offset)).normalize())
    return easter_days


def add_calendar_features(df: pd.DataFrame, date_col: str = "week_start") -> pd.DataFrame:
    out = df.copy()
    dates = out[date_col]
    out["week_of_year"] = dates.dt.isocalendar().week.astype(int)
    out["month"] = dates.dt.month
    out["quarter"] = dates.dt.quarter
    out["year"] = dates.dt.year
    out["sin_week"] = np.sin(2 * np.pi * out["week_of_year"] / 52)
    out["cos_week"] = np.cos(2 * np.pi * out["week_of_year"] / 52)
    out["sin_month"] = np.sin(2 * np.pi * out["month"] / 12)
    out["cos_month"] = np.cos(2 * np.pi * out["month"] / 12)
    out["is_christmas_week"] = out["week_of_year"].isin([51, 52]).astype(int)
    out["is_peak_season"] = (out["week_of_year"] >= 44).astype(int)
    out["weeks_to_christmas"] = dates.apply(_weeks_to_christmas)
    out["is_valentines"] = ((out["month"] == 2) & (dates.dt.day <= 14)).astype(int)

    easter_days = _easter_dates(dates.dt.year.unique())
    if easter_days:
        out["is_easter"] = dates.dt.normalize().isin(easter_days).astype(int)
    else:
        out["is_easter"] = 0
    return out


def build_spine(stock_codes: list[str], week_range: pd.PeriodIndex) -> pd.DataFrame:
    spine = pd.DataFrame(list(iterproduct(stock_codes, list(week_range))), columns=["StockCode", "week"])
    spine["week_start"] = spine["week"].apply(lambda p: p.start_time.normalize())
    spine["week_key"] = spine["week_start"].dt.strftime("%Y-%m-%d")
    return spine


def attach_product_features(panel: pd.DataFrame, feat_df_all: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    prod_feats = feat_df_all[feature_cols].copy()
    prod_feats.index.name = "StockCode"
    return panel.merge(prod_feats.reset_index(), on="StockCode", how="left")


def add_lag_features(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.sort_values(["StockCode", "week"]).copy()
    grp = df.groupby("StockCode")["sales"]

    for lag in LAG_WINDOWS:
        df[f"lag_{lag}w"] = grp.shift(lag)

    for window in ROLL_WINDOWS:
        df[f"roll_mean_{window}w"] = grp.transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

    df["lag_52w_roll4"] = grp.transform(lambda x: x.shift(52).rolling(4, min_periods=1).mean())
    df["short_trend"] = (
        grp.transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
        - grp.transform(lambda x: x.shift(5).rolling(4, min_periods=1).mean())
    )
    return df


def build_cluster_panels(
    feat_df_all: pd.DataFrame,
    dataset: DatasetBundle,
    product_feature_cols: list[str] | None = None,
) -> dict[int, dict[str, Any]]:
    if product_feature_cols is None:
        product_feature_cols = PRODUCT_FEATURE_COLS

    cluster_ids = sorted(feat_df_all["cluster"].unique())
    full_transactions = pd.concat([dataset.retail_train, dataset.retail_test], ignore_index=True)
    panels: dict[int, dict[str, Any]] = {}

    for cluster_id in cluster_ids:
        cluster_products = feat_df_all[feat_df_all["cluster"] == cluster_id].index.astype(str).tolist()
        if not cluster_products:
            continue

        label = feat_df_all.loc[feat_df_all["cluster"] == cluster_id, "cluster_label"].iloc[0]
        cluster_tx = full_transactions[full_transactions["StockCode"].isin(cluster_products)].copy()
        # Forecast the gross demand signal and keep cancellations only as auxiliary features.
        weekly_actuals = make_weekly_actuals(cluster_tx, drop_cancellations=True)

        panel = build_spine(cluster_products, dataset.all_weeks)
        panel = panel.merge(weekly_actuals, on=["StockCode", "week"], how="left")
        panel["sales"] = panel["sales"].fillna(0.0)
        panel = add_calendar_features(panel)
        panel = attach_product_features(panel, feat_df_all, [c for c in product_feature_cols if c in feat_df_all.columns])
        panel = add_lag_features(panel)

        train_panel = panel[panel["week"].isin(dataset.training_weeks)].copy()
        test_panel = panel[panel["week"].isin(dataset.test_weeks)].copy()
        train_panel = train_panel.dropna(subset=["lag_52w"]).copy()

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
        panels[int(cluster_id)] = {
            "train": train_panel.reset_index(drop=True),
            "test": test_panel.reset_index(drop=True),
            "features": feature_cols,
            "label": label,
        }

    return panels
