from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from itertools import product as iterproduct
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except (ImportError, OSError):  # pragma: no cover
    class LGBMRegressor:  # type: ignore[no-redef]
        def __init__(self, objective: str = "regression", random_state: int = 42, n_jobs: int = -1, verbose: int = -1, **kwargs: Any) -> None:
            max_iter = int(kwargs.pop("n_estimators", kwargs.pop("max_iter", 300)))
            learning_rate = float(kwargs.pop("learning_rate", 0.05))
            min_samples_leaf = int(kwargs.pop("min_child_samples", 20))
            max_depth = kwargs.pop("max_depth", None)
            l2_regularization = float(kwargs.pop("reg_lambda", kwargs.pop("l2_regularization", 0.0)))
            self.model = HistGradientBoostingRegressor(
                learning_rate=learning_rate,
                max_iter=max_iter,
                min_samples_leaf=min_samples_leaf,
                max_depth=max_depth,
                l2_regularization=l2_regularization,
                random_state=random_state,
            )

        def fit(self, X: pd.DataFrame, y: np.ndarray) -> "LGBMRegressor":
            self.model.fit(X, y)
            return self

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return self.model.predict(X)

    class LGBMClassifier:  # type: ignore[no-redef]
        def __init__(self, objective: str = "binary", random_state: int = 42, n_jobs: int = -1, verbose: int = -1, **kwargs: Any) -> None:
            max_iter = int(kwargs.pop("n_estimators", kwargs.pop("max_iter", 300)))
            learning_rate = float(kwargs.pop("learning_rate", 0.05))
            min_samples_leaf = int(kwargs.pop("min_child_samples", 20))
            max_depth = kwargs.pop("max_depth", None)
            l2_regularization = float(kwargs.pop("reg_lambda", kwargs.pop("l2_regularization", 0.0)))
            self.model = HistGradientBoostingClassifier(
                learning_rate=learning_rate,
                max_iter=max_iter,
                min_samples_leaf=min_samples_leaf,
                max_depth=max_depth,
                l2_regularization=l2_regularization,
                random_state=random_state,
            )

        def fit(self, X: pd.DataFrame, y: np.ndarray) -> "LGBMClassifier":
            self.model.fit(X, y)
            return self

        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            return self.model.predict_proba(X)

from .config import (
    AGGREGATE_LAG_WINDOWS,
    AGGREGATE_ROLL_WINDOWS,
    BASELINE_RF_PARAMS,
    BEST_C2_RF_PARAMS,
    CROSTON_ALPHA,
    DEFAULT_AGGREGATE_MLP_PARAMS,
    DEFAULT_LGBM_PARAMS,
    LGBM_PARAM_GRID,
    LAG_WINDOWS,
    MIN_ABS_ACTUAL,
    ProjectPaths,
    ROLL_WINDOWS,
)
from .data import DatasetBundle
from .features import add_calendar_features, make_weekly_actuals


@dataclass
class ModelingArtifacts:
    selection_df: pd.DataFrame
    candidate_df: pd.DataFrame
    tuned_trials_df: pd.DataFrame
    best_model_payload: dict[str, Any]


def mape_100(y_true: np.ndarray, y_pred: np.ndarray, min_abs_actual: float = MIN_ABS_ACTUAL) -> tuple[float, int]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) >= min_abs_actual
    if mask.sum() == 0:
        return np.nan, 0
    ape = np.abs((y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])) * 100
    return float(np.mean(ape)), int(mask.sum())


def signed_log1p(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))


def signed_expm1(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.expm1(np.abs(x))


def split_train_valid_time(df: pd.DataFrame, valid_ratio: float = 0.15, min_valid_weeks: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    sorted_df = df.sort_values(["week", "StockCode"]).copy()
    weeks = np.array(sorted(sorted_df["week"].unique()))

    if len(weeks) < (min_valid_weeks + 2):
        cut = int(len(sorted_df) * (1 - valid_ratio))
        cut = max(1, min(cut, len(sorted_df) - 1))
        return sorted_df.iloc[:cut].copy(), sorted_df.iloc[cut:].copy()

    n_valid_weeks = max(min_valid_weeks, int(len(weeks) * valid_ratio))
    n_valid_weeks = min(n_valid_weeks, max(min_valid_weeks, len(weeks) - 2))
    valid_weeks = set(weeks[-n_valid_weeks:])

    train_part = sorted_df[~sorted_df["week"].isin(valid_weeks)].copy()
    valid_part = sorted_df[sorted_df["week"].isin(valid_weeks)].copy()
    return train_part, valid_part


def croston_sba_forecast(y: np.ndarray, alpha: float = CROSTON_ALPHA) -> float:
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return 0.0
    y = np.clip(y, 0.0, None)
    nz_idx = np.flatnonzero(y > 0)
    if len(nz_idx) == 0:
        return 0.0

    z = y[nz_idx[0]]
    p = nz_idx[0] + 1
    last = nz_idx[0]
    for idx in nz_idx[1:]:
        demand = y[idx]
        interval = idx - last
        z = z + alpha * (demand - z)
        p = p + alpha * (interval - p)
        last = idx
    croston = z / p if p > 0 else 0.0
    return float(max(0.0, (1 - alpha / 2.0) * croston))


def croston_predict_by_sku(train_df: pd.DataFrame, test_df: pd.DataFrame, alpha: float = CROSTON_ALPHA) -> np.ndarray:
    train_sorted = train_df.sort_values(["StockCode", "week"]).copy()
    test_sorted = test_df.sort_values(["StockCode", "week"]).copy()

    sku_forecast: dict[str, float] = {}
    for sku, grp in train_sorted.groupby("StockCode"):
        sku_forecast[str(sku)] = croston_sba_forecast(grp["sales"].values, alpha=alpha)

    cluster_mean = float(np.clip(train_sorted["sales"], 0.0, None).mean()) if len(train_sorted) else 0.0
    pred = test_sorted["StockCode"].astype(str).map(sku_forecast).fillna(cluster_mean).values
    return pd.Series(pred, index=test_sorted.index).reindex(test_df.index).values


def build_weekly_raw_features(transactions: pd.DataFrame) -> pd.DataFrame:
    df = transactions.copy()
    df["week"] = df["InvoiceDate"].dt.to_period("W")
    df["is_cancel"] = df["Invoice"].astype(str).str.startswith("C") if "Invoice" in df.columns else False
    df["pos_sales"] = df["Sales"].clip(lower=0.0)
    df["neg_sales_abs"] = -df["Sales"].clip(upper=0.0)

    agg = (
        df.groupby(["StockCode", "week"]).agg(
            txn_count=("Sales", "size"),
            customer_count=("CustomerID", "nunique"),
            avg_price=("Price", "mean"),
            gross_sales=("pos_sales", "sum"),
            return_sales_abs=("neg_sales_abs", "sum"),
            cancel_count=("is_cancel", "sum"),
        )
        .reset_index()
    )

    invoice_count = (
        df.groupby(["StockCode", "week"])["Invoice"]
        .nunique()
        .rename("invoice_count")
        .reset_index()
    )
    top_country = (
        df.groupby(["StockCode", "week"])["Country"]
        .apply(lambda s: float(s.value_counts(normalize=True).iloc[0]) if len(s) else 0.0)
        .rename("top_country_share")
        .reset_index()
    )
    agg = agg.merge(invoice_count, on=["StockCode", "week"], how="left")
    agg = agg.merge(top_country, on=["StockCode", "week"], how="left")

    fill_cols = ["avg_price", "top_country_share", "invoice_count"]
    for col in fill_cols:
        agg[col] = agg[col].fillna(0.0)

    denom = agg["gross_sales"] + agg["return_sales_abs"] + 1e-6
    agg["return_ratio"] = agg["return_sales_abs"] / denom
    return agg


def add_lagged_raw_features(raw_panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = raw_panel.sort_values(["StockCode", "week"]).copy()
    metric_cols = [
        "txn_count",
        "invoice_count",
        "customer_count",
        "avg_price",
        "gross_sales",
        "return_sales_abs",
        "cancel_count",
        "top_country_share",
        "return_ratio",
    ]

    lag_cols: list[str] = []
    for col in metric_cols:
        grp = df.groupby("StockCode")[col]
        lag1 = f"{col}_lag1"
        roll4 = f"{col}_roll4"
        df[lag1] = grp.shift(1).fillna(0.0)
        df[roll4] = grp.transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean()).fillna(0.0)
        lag_cols.extend([lag1, roll4])
    return df, lag_cols


def fit_signedlog_model_with_params(train_df: pd.DataFrame, feat_cols: list[str], params: dict[str, Any]) -> Any:
    model = LGBMRegressor(
        objective="regression",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        **params,
    )
    model.fit(train_df[feat_cols].fillna(0.0), signed_log1p(train_df["sales"].values))
    return model


def predict_signedlog_model(model: Any, df: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
    pred = model.predict(df[feat_cols].fillna(0.0))
    return signed_expm1(pred)


def fit_return_models(train_df: pd.DataFrame, feat_cols: list[str]) -> dict[str, Any]:
    df = train_df.copy()
    df["return_amount"] = np.abs(np.minimum(df["sales"], 0.0))
    df["has_return"] = (df["return_amount"] > 0).astype(int)

    X = df[feat_cols].fillna(0.0)
    y_evt = df["has_return"].values

    if y_evt.sum() < 20 or np.unique(y_evt).size < 2:
        return {"status": "fallback_no_return_events", "cls": None, "reg": None}

    cls = LGBMClassifier(
        objective="binary",
        random_state=42,
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=25,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=0.2,
        n_jobs=-1,
        verbose=-1,
    )
    cls.fit(X, y_evt)

    ret_rows = df[df["has_return"] == 1].copy()
    if len(ret_rows) < 30:
        return {"status": "fallback_low_return_amount_rows", "cls": cls, "reg": None}

    reg = LGBMRegressor(
        objective="regression",
        random_state=42,
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=0.2,
        n_jobs=-1,
        verbose=-1,
    )
    reg.fit(ret_rows[feat_cols].fillna(0.0), np.log1p(ret_rows["return_amount"].values))
    return {"status": "ok", "cls": cls, "reg": reg}


def predict_expected_return(return_bundle: dict[str, Any], df: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
    if return_bundle["cls"] is None:
        return np.zeros(len(df), dtype=float)
    X = df[feat_cols].fillna(0.0)
    p_return = return_bundle["cls"].predict_proba(X)[:, 1]
    if return_bundle["reg"] is None:
        return np.zeros(len(df), dtype=float)
    ret_amount = np.expm1(return_bundle["reg"].predict(X))
    return p_return * np.clip(ret_amount, 0.0, None)


def residual_correction_predict_rolling(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    feat_cols: list[str],
    base_params: dict[str, Any],
    val_span: int = 4,
) -> tuple[np.ndarray, float]:
    train_sorted = train_df.sort_values(["week", "StockCode"]).copy()
    all_train_weeks = np.array(sorted(train_sorted["week"].unique()))
    min_train_weeks = max(26, val_span * 2)
    fold_ends = list(range(min_train_weeks, len(all_train_weeks) - val_span + 1, val_span))
    if not fold_ends:
        fold_ends = [len(all_train_weeks) - val_span]

    alpha_grid = np.linspace(0.0, 1.0, 11)
    rows: list[dict[str, Any]] = []

    for end_idx in fold_ends:
        fit_weeks = set(all_train_weeks[:end_idx])
        val_weeks = set(all_train_weeks[end_idx : end_idx + val_span])

        tr_fit = train_sorted[train_sorted["week"].isin(fit_weeks)].copy()
        tr_val = train_sorted[train_sorted["week"].isin(val_weeks)].copy()
        if len(tr_fit) == 0 or len(tr_val) == 0:
            continue

        base_model = fit_signedlog_model_with_params(tr_fit, feat_cols, base_params)
        pred_val_base = predict_signedlog_model(base_model, tr_val, feat_cols)
        ret_bundle = fit_return_models(tr_fit, feat_cols)
        pred_val_ret = predict_expected_return(ret_bundle, tr_val, feat_cols)

        for alpha in alpha_grid:
            pred_val = pred_val_base - alpha * pred_val_ret
            mape_value, n_used = mape_100(tr_val["sales"].values, pred_val)
            rows.append({"alpha": float(alpha), "mape": mape_value, "n": n_used})

    cv_df = pd.DataFrame(rows)
    cv_df = cv_df[(cv_df["n"] > 0) & np.isfinite(cv_df["mape"])].copy()
    if cv_df.empty:
        best_alpha = 0.0
    else:
        agg = cv_df.groupby("alpha").apply(lambda g: np.average(g["mape"], weights=g["n"]))
        best_alpha = float(agg.reset_index(name="weighted_cv_mape").sort_values("weighted_cv_mape").iloc[0]["alpha"])

    base_model_full = fit_signedlog_model_with_params(train_df, feat_cols, base_params)
    pred_base = predict_signedlog_model(base_model_full, pred_df, feat_cols)
    ret_bundle_full = fit_return_models(train_df, feat_cols)
    pred_ret = predict_expected_return(ret_bundle_full, pred_df, feat_cols)
    return pred_base - best_alpha * pred_ret, best_alpha


def tune_lgbm_for_cluster(train_df: pd.DataFrame, valid_df: pd.DataFrame, feat_cols: list[str]) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for idx, params in enumerate(LGBM_PARAM_GRID, start=1):
        try:
            model = fit_signedlog_model_with_params(train_df, feat_cols, params)
            pred_valid = predict_signedlog_model(model, valid_df, feat_cols)
            valid_mape, n_valid = mape_100(valid_df["sales"].values, pred_valid)
            rows.append({"trial": idx, "valid_mape": valid_mape, "n_valid": n_valid, "error": None, **params})
        except Exception as exc:  # pragma: no cover
            rows.append({"trial": idx, "valid_mape": np.nan, "n_valid": 0, "error": str(exc), **params})

    tune_df = pd.DataFrame(rows).sort_values("valid_mape", na_position="last").reset_index(drop=True)
    valid_only = tune_df[(tune_df["n_valid"] > 0) & np.isfinite(tune_df["valid_mape"])].copy()
    if valid_only.empty:
        return dict(DEFAULT_LGBM_PARAMS), tune_df

    best_row = valid_only.iloc[0]
    best_params = {
        "n_estimators": int(best_row["n_estimators"]),
        "learning_rate": float(best_row["learning_rate"]),
        "num_leaves": int(best_row["num_leaves"]),
        "min_child_samples": int(best_row["min_child_samples"]),
        "subsample": float(best_row["subsample"]),
        "colsample_bytree": float(best_row["colsample_bytree"]),
        "reg_alpha": float(best_row["reg_alpha"]),
        "reg_lambda": float(best_row["reg_lambda"]),
    }
    return best_params, tune_df


def _pred_rf_signedlog(train_df: pd.DataFrame, pred_df: pd.DataFrame, feat_cols: list[str], params: dict[str, Any]) -> np.ndarray:
    model = RandomForestRegressor(**params)
    model.fit(train_df[feat_cols].fillna(0.0), signed_log1p(train_df["sales"].values))
    return np.clip(signed_expm1(model.predict(pred_df[feat_cols].fillna(0.0))), 0.0, None)


def _normalize_aggregate_mlp_params(params: dict[str, Any] | None = None) -> dict[str, Any]:
    effective = dict(DEFAULT_AGGREGATE_MLP_PARAMS)
    if params:
        effective.update(params)
    hidden_layers = effective.get("hidden_layer_sizes")
    if hidden_layers is not None:
        effective["hidden_layer_sizes"] = tuple(int(v) for v in hidden_layers)
    return effective


def build_cluster_aggregate_panel(rows: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    agg = (
        rows.groupby("week", as_index=False)
        .agg(sales=("sales", lambda s: float(s.sum(min_count=1)) if len(s) else np.nan))
        .sort_values("week")
        .reset_index(drop=True)
    )
    agg["week_start"] = agg["week"].apply(lambda p: p.start_time.normalize())
    agg = add_calendar_features(agg, date_col="week_start")

    series = agg["sales"]
    for lag in AGGREGATE_LAG_WINDOWS:
        agg[f"cluster_lag_{lag}w"] = series.shift(lag)

    for window in AGGREGATE_ROLL_WINDOWS:
        agg[f"cluster_roll_mean_{window}w"] = series.shift(1).rolling(window, min_periods=1).mean()

    agg["cluster_short_trend"] = (
        series.shift(1).rolling(4, min_periods=1).mean()
        - series.shift(5).rolling(4, min_periods=1).mean()
    )

    feature_cols = [col for col in agg.columns if col not in {"week", "week_start", "sales"}]
    return agg, feature_cols


def fit_aggregate_mlp_model(train_df: pd.DataFrame, feat_cols: list[str], params: dict[str, Any] | None = None) -> Pipeline:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(**_normalize_aggregate_mlp_params(params))),
        ]
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(train_df[feat_cols].fillna(0.0), signed_log1p(train_df["sales"].values))
    return model


def predict_aggregate_mlp(model: Pipeline, df: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
    if df.empty:
        return np.array([], dtype=float)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        pred = model.predict(df[feat_cols].fillna(0.0))
    return np.clip(signed_expm1(pred), 0.0, None)


def compute_recent_sku_weights(train_df: pd.DataFrame, window: int = 13) -> dict[str, float]:
    weeks = np.array(sorted(train_df["week"].unique()))
    if weeks.size == 0:
        return {}

    recent_weeks = set(weeks[-min(window, len(weeks)) :])
    recent = train_df[train_df["week"].isin(recent_weeks)].copy()
    recent["weight_sales"] = recent["sales"].clip(lower=0.0)
    sku_weights = recent.groupby("StockCode")["weight_sales"].sum()

    if float(sku_weights.sum()) <= 0.0:
        fallback = train_df.copy()
        fallback["weight_sales"] = fallback["sales"].clip(lower=0.0)
        sku_weights = fallback.groupby("StockCode")["weight_sales"].sum()

    if float(sku_weights.sum()) <= 0.0:
        sku_weights = train_df.groupby("StockCode").size().astype(float)

    total = float(sku_weights.sum())
    if total <= 0.0:
        return {}
    return {str(stock_code): float(weight / total) for stock_code, weight in sku_weights.items()}


def disaggregate_cluster_forecast(
    pred_df: pd.DataFrame,
    aggregate_pred_df: pd.DataFrame,
    sku_weights: dict[str, float],
) -> np.ndarray:
    out = pred_df[["StockCode", "week"]].copy().reset_index()
    out = out.merge(aggregate_pred_df, on="week", how="left")
    unique_skus = out["StockCode"].astype(str).drop_duplicates()
    default_weight = 1.0 / max(1, len(unique_skus))
    out["sku_weight"] = out["StockCode"].astype(str).map(sku_weights).fillna(default_weight)

    dedup = out[["StockCode", "sku_weight"]].drop_duplicates()
    weight_sum = float(dedup["sku_weight"].sum())
    if weight_sum > 0.0:
        out["sku_weight"] = out["sku_weight"] / weight_sum
    else:
        out["sku_weight"] = default_weight

    out["pred"] = out["aggregate_pred"].fillna(0.0) * out["sku_weight"]
    return out.sort_values("index")["pred"].to_numpy()


def aggregate_mlp_disagg_predict(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    params: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    full_rows = pd.concat(
        [
            train_df[["StockCode", "week", "sales"]],
            pred_df[["StockCode", "week", "sales"]],
        ],
        ignore_index=True,
    )
    aggregate_panel, aggregate_feat_cols = build_cluster_aggregate_panel(full_rows)
    required_lag = f"cluster_lag_{max(AGGREGATE_LAG_WINDOWS)}w"

    train_weeks = set(train_df["week"].unique())
    pred_weeks = set(pred_df["week"].unique())
    aggregate_train = aggregate_panel[aggregate_panel["week"].isin(train_weeks)].dropna(subset=[required_lag]).copy()
    aggregate_pred = aggregate_panel[aggregate_panel["week"].isin(pred_weeks)].dropna(subset=[required_lag]).copy()

    if aggregate_train.empty or aggregate_pred.empty:
        raise ValueError("Aggregate MLP panel does not have enough lagged history to score this split.")

    model = fit_aggregate_mlp_model(aggregate_train, aggregate_feat_cols, params=params)
    aggregate_pred_values = predict_aggregate_mlp(model, aggregate_pred, aggregate_feat_cols)
    aggregate_pred_df = aggregate_pred[["week"]].copy()
    aggregate_pred_df["aggregate_pred"] = aggregate_pred_values

    sku_weights = compute_recent_sku_weights(train_df)
    disagg_pred = disaggregate_cluster_forecast(pred_df, aggregate_pred_df, sku_weights)
    meta = {
        "params": _normalize_aggregate_mlp_params(params),
        "n_aggregate_train_weeks": int(len(aggregate_train)),
        "n_aggregate_pred_weeks": int(len(aggregate_pred)),
        "n_skus": int(train_df["StockCode"].nunique()),
    }
    return disagg_pred, meta


def recursive_aggregate_mlp_forecast(
    history_df: pd.DataFrame,
    horizon: int,
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    aggregate_history, aggregate_feat_cols = build_cluster_aggregate_panel(history_df)
    required_lag = f"cluster_lag_{max(AGGREGATE_LAG_WINDOWS)}w"
    aggregate_train = aggregate_history.dropna(subset=[required_lag]).copy()
    if aggregate_train.empty:
        raise ValueError("Aggregate MLP forecast needs more history before it can fit.")

    model = fit_aggregate_mlp_model(aggregate_train, aggregate_feat_cols, params=params)
    future_weeks = pd.period_range(start=aggregate_history["week"].max() + 1, periods=horizon, freq="W")
    working = pd.concat(
        [
            aggregate_history[["week", "sales"]],
            pd.DataFrame({"week": future_weeks, "sales": np.nan}),
        ],
        ignore_index=True,
    )

    for future_week in future_weeks:
        panel, feat_cols = build_cluster_aggregate_panel(working)
        future_row = panel[panel["week"] == future_week].copy()
        pred = predict_aggregate_mlp(model, future_row, feat_cols)[0]
        working.loc[working["week"] == future_week, "sales"] = pred

    final_panel, _ = build_cluster_aggregate_panel(working)
    future_panel = final_panel[final_panel["week"].isin(future_weeks)][["week", "sales"]].copy()
    future_panel = future_panel.rename(columns={"sales": "aggregate_pred"}).reset_index(drop=True)
    return future_panel


def build_raw_lag_cache_for_cluster(
    cluster_id: int,
    feat_df_all: pd.DataFrame,
    dataset: DatasetBundle,
) -> tuple[pd.DataFrame, list[str]]:
    cluster_products = feat_df_all[feat_df_all["cluster"] == cluster_id].index.astype(str).tolist()
    full_transactions = pd.concat([dataset.retail_train, dataset.retail_test], ignore_index=True)
    cluster_tx = full_transactions[full_transactions["StockCode"].isin(cluster_products)].copy()

    raw_weekly = build_weekly_raw_features(cluster_tx)
    spine = pd.DataFrame(list(iterproduct(cluster_products, list(dataset.all_weeks))), columns=["StockCode", "week"])
    raw_panel = spine.merge(raw_weekly, on=["StockCode", "week"], how="left")

    fill_cols = [
        "txn_count",
        "invoice_count",
        "customer_count",
        "avg_price",
        "gross_sales",
        "return_sales_abs",
        "cancel_count",
        "top_country_share",
        "return_ratio",
    ]
    for col in fill_cols:
        raw_panel[col] = raw_panel[col].fillna(0.0)

    raw_panel, raw_lag_cols = add_lagged_raw_features(raw_panel)
    return raw_panel[["StockCode", "week"] + raw_lag_cols].copy(), raw_lag_cols


def two_stage_rawlag_predict(
    cluster_id: int,
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    feat_cols: list[str],
    raw_cache: tuple[pd.DataFrame, list[str]],
) -> tuple[np.ndarray, str]:
    raw_join, raw_lag_cols = raw_cache
    train = train_df.merge(raw_join, on=["StockCode", "week"], how="left")
    pred = pred_df.merge(raw_join, on=["StockCode", "week"], how="left")
    train[raw_lag_cols] = train[raw_lag_cols].fillna(0.0)
    pred[raw_lag_cols] = pred[raw_lag_cols].fillna(0.0)

    all_feat_cols = list(feat_cols) + raw_lag_cols
    X_train = train[all_feat_cols].fillna(0.0)
    X_pred = pred[all_feat_cols].fillna(0.0)
    y_train = train["sales"].values
    y_event = (y_train > 0).astype(int)

    if y_event.sum() < 20 or np.unique(y_event).size < 2:
        return croston_predict_by_sku(train_df, pred_df, alpha=CROSTON_ALPHA), "TwoStageRawLag->Croston(fallback_events)"

    cls = LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=0.2,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    cls.fit(X_train, y_event)

    amount_train = train[train["sales"] > 0].copy()
    if len(amount_train) < 30:
        return croston_predict_by_sku(train_df, pred_df, alpha=CROSTON_ALPHA), "TwoStageRawLag->Croston(fallback_amount)"

    reg = LGBMRegressor(
        objective="regression",
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=0.2,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    reg.fit(amount_train[all_feat_cols].fillna(0.0), np.log1p(amount_train["sales"].values))

    p_sale = cls.predict_proba(X_pred)[:, 1]
    amount_pred = np.expm1(reg.predict(X_pred))
    amount_pred = np.clip(amount_pred, 0.0, None)

    cap = np.nanpercentile(amount_train["sales"].values, 99)
    if np.isfinite(cap) and cap > 0:
        amount_pred = np.clip(amount_pred, 0.0, cap * 3.0)
    return p_sale * amount_pred, "TwoStageRawLag"


def deepar_predict(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    h: int,
    input_size: int = 24,
    max_steps: int = 500,
) -> np.ndarray:
    import logging
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

    from neuralforecast import NeuralForecast
    from neuralforecast.models import DeepAR
    from neuralforecast.losses.pytorch import DistributionLoss

    nf_train = (
        train_df[["StockCode", "week_start", "sales"]]
        .rename(columns={"StockCode": "unique_id", "week_start": "ds", "sales": "y"})
        .sort_values(["unique_id", "ds"])
        .copy()
    )
    nf_train["y"] = nf_train["y"].clip(lower=0.0)
    nf_train["ds"] = pd.to_datetime(nf_train["ds"])

    model = DeepAR(
        h=h,
        input_size=input_size,
        loss=DistributionLoss(distribution="NegativeBinomial", level=[90]),
        max_steps=max_steps,
        random_seed=42,
        accelerator="cpu",
    )
    nf = NeuralForecast(models=[model], freq="7D")
    nf.fit(nf_train)

    forecasts = nf.predict().reset_index(drop=True)
    point_col = next((c for c in forecasts.columns if c.startswith("DeepAR") and "-" not in c), None)
    if point_col is None:
        return np.zeros(len(pred_df))

    forecasts = forecasts.rename(columns={"unique_id": "StockCode", "ds": "week_start", point_col: "_pred"})
    forecasts["week_start"] = pd.to_datetime(forecasts["week_start"])

    pred_ws = pd.to_datetime(pred_df["week_start"])
    result = (
        pred_df.assign(week_start=pred_ws)
        .reset_index()
        .merge(forecasts[["StockCode", "week_start", "_pred"]], on=["StockCode", "week_start"], how="left")
        .set_index("index")
        .reindex(pred_df.index)["_pred"]
        .fillna(0.0)
        .values
    )
    return np.clip(result, 0.0, None)


def _candidate_methods_for_cluster(cluster_id: int) -> list[str]:
    # Keep model selection strictly at the cluster level.
    # We intentionally exclude per-SKU baselines and multi-stage hybrids here
    # so every candidate corresponds to one pooled model fit per cluster.
    if cluster_id in {-2, -1}:
        return ["DeepAR", "RF_Default", "LGBM_Default", "AggregateMLP_Disagg"]
    if cluster_id == 2:
        return ["RF_C2_BEST", "RF_Default", "AggregateMLP_Disagg", "LGBM_Tuned", "LGBM_Default"]
    return ["RF_Default", "AggregateMLP_Disagg", "LGBM_Tuned", "LGBM_Default"]


def train_cluster_models(
    feat_df_all: pd.DataFrame,
    panels: dict[int, dict[str, Any]],
    dataset: DatasetBundle,
    paths: ProjectPaths,
) -> ModelingArtifacts:
    paths.ensure_dirs()

    selection_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    tuned_trial_rows: list[dict[str, Any]] = []
    tuned_params_by_cluster: dict[int, dict[str, Any]] = {}

    for cluster_id in sorted(panels):
        label = panels[cluster_id]["label"]
        full_train = panels[cluster_id]["train"].sort_values(["week", "StockCode"]).copy()
        test_df = panels[cluster_id]["test"].sort_values(["week", "StockCode"]).copy()
        feat_cols = list(panels[cluster_id]["features"])

        train_df, valid_df = split_train_valid_time(full_train, valid_ratio=0.15, min_valid_weeks=4)

        tuned_params = dict(DEFAULT_LGBM_PARAMS)
        if cluster_id >= 0:
            tuned_params, tune_df = tune_lgbm_for_cluster(train_df, valid_df, feat_cols)
            tune_df = tune_df.copy()
            tune_df.insert(0, "cluster", cluster_id)
            tuned_trial_rows.extend(tune_df.to_dict("records"))
            tuned_params_by_cluster[int(cluster_id)] = tuned_params

        best_method = None
        best_valid_mape = np.inf
        best_valid_meta: dict[str, Any] = {}

        for method in _candidate_methods_for_cluster(cluster_id):
            try:
                if method == "CrostonSBA":
                    y_valid_pred = croston_predict_by_sku(train_df, valid_df)
                    meta = {}
                elif method == "TwoStageRawLag":
                    raw_cache = build_raw_lag_cache_for_cluster(cluster_id, feat_df_all, dataset)
                    y_valid_pred, inner_name = two_stage_rawlag_predict(cluster_id, train_df, valid_df, feat_cols, raw_cache)
                    meta = {"inner_name": inner_name}
                elif method == "LGBM_Default":
                    model = fit_signedlog_model_with_params(train_df, feat_cols, DEFAULT_LGBM_PARAMS)
                    y_valid_pred = predict_signedlog_model(model, valid_df, feat_cols)
                    meta = {}
                elif method == "LGBM_Tuned":
                    model = fit_signedlog_model_with_params(train_df, feat_cols, tuned_params)
                    y_valid_pred = predict_signedlog_model(model, valid_df, feat_cols)
                    meta = {"params": tuned_params}
                elif method == "RF_Default":
                    y_valid_pred = _pred_rf_signedlog(train_df, valid_df, feat_cols, BASELINE_RF_PARAMS)
                    meta = {}
                elif method == "RF_C2_BEST":
                    y_valid_pred = _pred_rf_signedlog(train_df, valid_df, feat_cols, BEST_C2_RF_PARAMS)
                    meta = {}
                elif method == "DeepAR":
                    y_valid_pred = deepar_predict(train_df, valid_df, h=valid_df["week"].nunique())
                    meta = {}
                elif method == "AggregateMLP_Disagg":
                    y_valid_pred, meta = aggregate_mlp_disagg_predict(train_df, valid_df, DEFAULT_AGGREGATE_MLP_PARAMS)
                elif method == "ResidualCorrectionRollingCV":
                    y_valid_pred, alpha = residual_correction_predict_rolling(train_df, valid_df, feat_cols, tuned_params)
                    meta = {"alpha": alpha}
                else:  # pragma: no cover
                    raise ValueError(f"Unknown method: {method}")

                valid_mape, n_valid = mape_100(valid_df["sales"].values, y_valid_pred)
            except Exception as exc:  # pragma: no cover
                valid_mape, n_valid = np.nan, 0
                meta = {"error": str(exc)}

            candidate_rows.append(
                {
                    "cluster": cluster_id,
                    "label": label,
                    "candidate": method,
                    "valid_mape": float(valid_mape) if np.isfinite(valid_mape) else np.nan,
                    "n_valid": int(n_valid),
                    "meta": json.dumps(meta, ensure_ascii=False),
                }
            )

            if np.isfinite(valid_mape) and n_valid > 0 and valid_mape < best_valid_mape:
                best_valid_mape = float(valid_mape)
                best_method = method
                best_valid_meta = meta

        if best_method is None:
            best_method = "LGBM_Default"
            best_valid_mape = np.nan
            best_valid_meta = {"fallback": "no_valid_candidate"}

        train_plus_valid = pd.concat([train_df, valid_df], ignore_index=True)
        if best_method == "CrostonSBA":
            y_test_pred = croston_predict_by_sku(train_plus_valid, test_df)
            test_meta = {}
        elif best_method == "TwoStageRawLag":
            raw_cache = build_raw_lag_cache_for_cluster(cluster_id, feat_df_all, dataset)
            y_test_pred, inner_name = two_stage_rawlag_predict(cluster_id, train_plus_valid, test_df, feat_cols, raw_cache)
            test_meta = {"inner_name": inner_name}
        elif best_method == "LGBM_Default":
            model = fit_signedlog_model_with_params(train_plus_valid, feat_cols, DEFAULT_LGBM_PARAMS)
            y_test_pred = predict_signedlog_model(model, test_df, feat_cols)
            test_meta = {}
        elif best_method == "LGBM_Tuned":
            model = fit_signedlog_model_with_params(train_plus_valid, feat_cols, tuned_params)
            y_test_pred = predict_signedlog_model(model, test_df, feat_cols)
            test_meta = {"params": tuned_params}
        elif best_method == "RF_Default":
            y_test_pred = _pred_rf_signedlog(train_plus_valid, test_df, feat_cols, BASELINE_RF_PARAMS)
            test_meta = {}
        elif best_method == "RF_C2_BEST":
            y_test_pred = _pred_rf_signedlog(train_plus_valid, test_df, feat_cols, BEST_C2_RF_PARAMS)
            test_meta = {}
        elif best_method == "DeepAR":
            y_test_pred = deepar_predict(train_plus_valid, test_df, h=test_df["week"].nunique())
            test_meta = {}
        elif best_method == "AggregateMLP_Disagg":
            y_test_pred, test_meta = aggregate_mlp_disagg_predict(train_plus_valid, test_df, DEFAULT_AGGREGATE_MLP_PARAMS)
        elif best_method == "ResidualCorrectionRollingCV":
            y_test_pred, alpha = residual_correction_predict_rolling(train_plus_valid, test_df, feat_cols, tuned_params)
            test_meta = {"alpha": alpha}
        else:  # pragma: no cover
            raise ValueError(f"Unknown method: {best_method}")

        test_mape, n_test = mape_100(test_df["sales"].values, y_test_pred)

        try:
            lgb_baseline_model = fit_signedlog_model_with_params(train_plus_valid, feat_cols, DEFAULT_LGBM_PARAMS)
            y_test_lgb = predict_signedlog_model(lgb_baseline_model, test_df, feat_cols)
            lgbm_base_mape, _ = mape_100(test_df["sales"].values, y_test_lgb)
        except Exception:  # pragma: no cover
            lgbm_base_mape = np.nan

        selection_rows.append(
            {
                "cluster": cluster_id,
                "label": label,
                "selected_model": best_method,
                "valid_mape_selected": float(best_valid_mape) if np.isfinite(best_valid_mape) else np.nan,
                "test_mape_selected": float(test_mape) if np.isfinite(test_mape) else np.nan,
                "test_mape_lgbm_baseline": float(lgbm_base_mape) if np.isfinite(lgbm_base_mape) else np.nan,
                "delta_selected_minus_lgbm": float(test_mape - lgbm_base_mape) if np.isfinite(test_mape) and np.isfinite(lgbm_base_mape) else np.nan,
                "n_train": len(train_df),
                "n_valid": len(valid_df),
                "n_test": len(test_df),
                "n_test_used": int(n_test),
                "select_meta": json.dumps(best_valid_meta, ensure_ascii=False),
                "test_meta": json.dumps(test_meta, ensure_ascii=False),
            }
        )

    selection_df = pd.DataFrame(selection_rows).sort_values("cluster").reset_index(drop=True)
    candidate_df = pd.DataFrame(candidate_rows).sort_values(["cluster", "valid_mape"], na_position="last").reset_index(drop=True)
    tuned_trials_df = pd.DataFrame(tuned_trial_rows)
    if not tuned_trials_df.empty:
        tuned_trials_df = tuned_trials_df.sort_values(["cluster", "valid_mape"], na_position="last").reset_index(drop=True)

    selection_df.to_csv(paths.selection_summary_csv, index=False)
    candidate_df.to_csv(paths.candidate_metrics_csv, index=False)
    tuned_trials_df.to_csv(paths.tuned_lgbm_trials_csv, index=False)

    cluster_configs: list[dict[str, Any]] = []
    for _, row in selection_df.iterrows():
        cid = int(row["cluster"])
        model_name = row["selected_model"]

        if model_name == "RF_C2_BEST":
            params = dict(BEST_C2_RF_PARAMS)
        elif model_name == "RF_Default":
            params = dict(BASELINE_RF_PARAMS)
        elif model_name == "LGBM_Tuned":
            params = dict(tuned_params_by_cluster.get(cid, DEFAULT_LGBM_PARAMS))
        elif model_name == "LGBM_Default":
            params = dict(DEFAULT_LGBM_PARAMS)
        elif model_name == "AggregateMLP_Disagg":
            params = dict(_normalize_aggregate_mlp_params(DEFAULT_AGGREGATE_MLP_PARAMS))
        elif model_name == "DeepAR":
            params = {"input_size": 24, "max_steps": 500}
        elif model_name == "CrostonSBA":
            params = {"alpha": CROSTON_ALPHA}
        elif model_name == "ResidualCorrectionRollingCV":
            test_meta = json.loads(row["test_meta"]) if row["test_meta"] else {}
            params = {
                "base_params": dict(tuned_params_by_cluster.get(cid, DEFAULT_LGBM_PARAMS)),
                "alpha": float(test_meta.get("alpha", 0.0)),
            }
        else:
            params = {}

        cluster_configs.append(
            {
                "cluster": cid,
                "label": row["label"],
                "selected_model": model_name,
                "params": params,
                "test_mape_selected": None if pd.isna(row["test_mape_selected"]) else float(row["test_mape_selected"]),
                "test_mape_lgbm_baseline": None if pd.isna(row["test_mape_lgbm_baseline"]) else float(row["test_mape_lgbm_baseline"]),
            }
        )

    payload = {
        "forecast_horizon": "next_12_weeks",
        "cluster_feature_file": str(paths.product_features_clustered_csv),
        "sales_file": str(paths.cleaned_retail_csv),
        "defaults": {
            "BASELINE_RF_PARAMS": BASELINE_RF_PARAMS,
            "BEST_C2_RF_PARAMS": BEST_C2_RF_PARAMS,
            "DEFAULT_LGBM_PARAMS": DEFAULT_LGBM_PARAMS,
            "DEFAULT_AGGREGATE_MLP_PARAMS": _normalize_aggregate_mlp_params(DEFAULT_AGGREGATE_MLP_PARAMS),
            "CROSTON_ALPHA": CROSTON_ALPHA,
        },
        "cluster_configs": cluster_configs,
    }
    paths.best_model_params_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return ModelingArtifacts(
        selection_df=selection_df,
        candidate_df=candidate_df,
        tuned_trials_df=tuned_trials_df,
        best_model_payload=payload,
    )
