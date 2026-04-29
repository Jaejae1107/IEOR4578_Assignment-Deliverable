"""
Pure-Python chatbot tools for the FortuneTellers Streamlit dashboard.

These functions wrap the existing ``fortunetellers`` package so the
Streamlit UI never has to talk directly to the modelling internals.
No Streamlit imports are allowed in this module - it must stay
framework-agnostic so it can be reused from notebooks or unit tests.

Tool surface
------------
- parse_forecast_request(query, default_country, default_horizon) -> dict
- lookup_product_cluster(product_id) -> dict
- get_best_model(product_id) -> dict
- run_forecast(product_id, country, horizon_weeks) -> dict
- get_last_forecast(product_id) -> dict | None

Model selection is *always* read from the project artifacts
(``best_model_params.json`` and ``cluster_model_selection_summary.csv``)
and never invented by an LLM.
"""

from __future__ import annotations

import json
import re
import sys
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Make the existing fortunetellers/ package importable when this module is
# run from inside the dashboard/ directory (e.g. ``streamlit run app.py``).
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fortunetellers.config import ProjectPaths  # noqa: E402
from fortunetellers.pipeline import ForecastingPipeline, ForecastOutput  # noqa: E402


# ---------------------------------------------------------------------------
# Constants reused by the natural-language parser.
# ---------------------------------------------------------------------------
COUNTRY_ALIASES: Dict[str, str] = {
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "u.k": "United Kingdom",
    "united kingdom": "United Kingdom",
    "great britain": "United Kingdom",
    "britain": "United Kingdom",
    "england": "United Kingdom",
    "all countries": "ALL",
    "all country": "ALL",
    "all markets": "ALL",
    "all market": "ALL",
    "all": "ALL",
    "global": "ALL",
    "worldwide": "ALL",
    "world wide": "ALL",
    "overall": "ALL",
    "everywhere": "ALL",
}

# Plain-English explanation strings used by the chatbot to keep the UI
# consistent regardless of whether an LLM wrapper is in use.
MAPE_EXPLANATION = (
    "MAPE means the average percentage error of the forecast. "
    "A MAPE of 18% means the model's predictions are typically about 18% away "
    "from actual demand. For retail demand forecasting this is usable for "
    "planning, but should be interpreted with caution for volatile or "
    "intermittent products."
)

MODEL_EXPLANATIONS: Dict[str, str] = {
    "RF_Default": (
        "Random Forest with the project's default hyper-parameters. It uses "
        "lag, rolling-window, calendar and product-level features to predict "
        "weekly sales."
    ),
    "RF_C2_BEST": (
        "Random Forest tuned specifically for the steady-regulars cluster. It "
        "uses lag, rolling-window, calendar and product-level features."
    ),
    "LGBM_Default": (
        "LightGBM with default hyper-parameters - a gradient boosted tree "
        "model that uses the same lag/rolling/calendar/product features."
    ),
    "LGBM_Tuned": (
        "LightGBM with hyper-parameters tuned per cluster. It uses the same "
        "lag/rolling/calendar/product features as the other tree models."
    ),
    "AggregateMLP_Disagg": (
        "A small neural network that first forecasts the cluster's aggregate "
        "demand and then disaggregates the result down to the product level "
        "using each SKU's recent share of cluster sales."
    ),
    "CrostonSBA": (
        "Croston's method (Syntetos-Boylan approximation) - the textbook "
        "choice for intermittent or sporadic demand."
    ),
}


# ---------------------------------------------------------------------------
# Pipeline cache.  Building the pipeline reads ~hundreds of MB of CSVs and
# trains cluster models on demand, so we keep one shared instance.
# ---------------------------------------------------------------------------
_PIPELINE_LOCK = threading.Lock()
_PIPELINE_INSTANCE: Optional[ForecastingPipeline] = None
_LAST_FORECAST_BY_PRODUCT: Dict[str, Dict[str, Any]] = {}


def _project_paths() -> ProjectPaths:
    """Return a ``ProjectPaths`` rooted at the existing repo layout."""
    return ProjectPaths(artifact_root=_PROJECT_ROOT / "fortunetellers_artifacts")


def _check_pipeline_inputs(paths: ProjectPaths) -> None:
    """Validate that the files ForecastingPipeline needs are actually on disk.

    The pipeline's first call (``load_or_prepare_transactions``) tries to load
    ``processed/total_retail_cleaned.csv`` and falls back to cleaning the
    raw Excel.  If both are missing the pipeline fails with a generic
    "Raw Excel file not found" - we replace that with an actionable message
    that prints the exact absolute paths for this checkout.
    """
    cleaned_ok = paths.cleaned_retail_csv.exists() and paths.cleaned_products_csv.exists()
    if cleaned_ok:
        return
    if paths.raw_excel.exists():
        return
    raise FileNotFoundError(
        "Cannot run the forecasting pipeline because the cleaned transaction "
        "data is missing AND the raw Excel is missing.\n\n"
        "Provide ONE of the following:\n"
        f"  1. {paths.cleaned_retail_csv}\n"
        f"     and {paths.cleaned_products_csv}\n"
        "     (produced by the existing data cleaning step)\n"
        f"  2. The raw Excel at {paths.raw_excel}\n"
        "     (or set the FORTUNETELLERS_RAW_EXCEL env var to its location)\n\n"
        "Once one of those is present, restart the Streamlit app."
    )


def get_pipeline() -> ForecastingPipeline:
    """Return a process-wide singleton ForecastingPipeline.

    Building the pipeline loads the cleaned retail dataset and clustering
    feature artifacts; we cache it because rebuilding for every chatbot turn
    would be unacceptably slow.
    """
    global _PIPELINE_INSTANCE
    if _PIPELINE_INSTANCE is not None:
        return _PIPELINE_INSTANCE
    with _PIPELINE_LOCK:
        if _PIPELINE_INSTANCE is None:
            paths = _project_paths()
            _check_pipeline_inputs(paths)
            _PIPELINE_INSTANCE = ForecastingPipeline(paths)
    return _PIPELINE_INSTANCE


def reset_pipeline() -> None:
    """Force the next call to ``get_pipeline`` to rebuild from disk."""
    global _PIPELINE_INSTANCE
    with _PIPELINE_LOCK:
        _PIPELINE_INSTANCE = None


# ---------------------------------------------------------------------------
# Artifact loaders (cached so repeated chatbot turns do not re-read CSVs).
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_clustered_features() -> pd.DataFrame:
    paths = _project_paths()
    fpath = paths.product_features_clustered_csv
    if not fpath.exists():
        raise FileNotFoundError(
            f"Clustered product features not found. Expected file: {fpath}\n"
            "Run the clustering / feature-engineering pipeline first."
        )
    df = pd.read_csv(fpath)
    if "StockCode" in df.columns:
        df["StockCode"] = df["StockCode"].astype(str)
        df = df.set_index("StockCode", drop=False)
    return df


@lru_cache(maxsize=1)
def _load_selection_summary() -> pd.DataFrame:
    paths = _project_paths()
    fpath = paths.selection_summary_csv
    if not fpath.exists():
        raise FileNotFoundError(
            f"Cluster model selection summary not found. Expected file: {fpath}"
        )
    return pd.read_csv(fpath)


@lru_cache(maxsize=1)
def _load_candidate_metrics() -> pd.DataFrame:
    paths = _project_paths()
    fpath = paths.candidate_metrics_csv
    if not fpath.exists():
        raise FileNotFoundError(
            f"Cluster candidate metrics not found. Expected file: {fpath}"
        )
    return pd.read_csv(fpath)


@lru_cache(maxsize=1)
def _load_best_model_params() -> Dict[str, Any]:
    paths = _project_paths()
    fpath = paths.best_model_params_json
    if not fpath.exists():
        raise FileNotFoundError(
            f"best_model_params.json not found. Expected file: {fpath}"
        )
    return json.loads(fpath.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _load_cluster_metadata() -> Dict[str, Any]:
    paths = _project_paths()
    fpath = paths.cluster_metadata_json
    if not fpath.exists():
        return {}
    try:
        return json.loads(fpath.read_text(encoding="utf-8"))
    except Exception:
        return {}


def reload_artifacts() -> None:
    """Drop cached artifact dataframes so a UI refresh picks up new files."""
    _load_clustered_features.cache_clear()
    _load_selection_summary.cache_clear()
    _load_candidate_metrics.cache_clear()
    _load_best_model_params.cache_clear()
    _load_cluster_metadata.cache_clear()


# ---------------------------------------------------------------------------
# 1. Natural-language parser
# ---------------------------------------------------------------------------
_HORIZON_PATTERNS = [
    # "next 12 weeks", "12-week forecast", "for 12 weeks"
    (re.compile(r"(\d+)\s*[- ]?\s*week", re.IGNORECASE), "weeks"),
    # "next 3 months"
    (re.compile(r"(\d+)\s*[- ]?\s*month", re.IGNORECASE), "months"),
]


def _extract_horizon(query: str, default_horizon: int) -> int:
    """Pull a forecast horizon (in weeks) out of a free-text query."""
    for pattern, unit in _HORIZON_PATTERNS:
        match = pattern.search(query)
        if not match:
            continue
        try:
            value = int(match.group(1))
        except ValueError:
            continue
        if value <= 0:
            continue
        if unit == "weeks":
            return value
        if unit == "months":
            # Roughly 4 weeks per month.
            return max(1, int(round(value * 4)))
    return int(default_horizon)


def _extract_country(query: str, default_country: str) -> str:
    """Resolve a country alias from the query, falling back to default."""
    lowered = query.lower()
    for alias, canonical in sorted(
        COUNTRY_ALIASES.items(), key=lambda kv: len(kv[0]), reverse=True
    ):
        if alias in lowered:
            return canonical

    # Try to match against any country we have actually seen in the dataset.
    try:
        feats = _load_clustered_features()
    except FileNotFoundError:
        feats = None
    if feats is not None and "country" in feats.columns:
        # Some pipelines store a primary country per product; ignore here.
        pass

    return default_country


def _extract_product_id(query: str) -> Optional[str]:
    """Try to find a StockCode-like token in the query.

    Strategy:
      1. Look for explicit prefixes (product/sku/stockcode/item).
      2. If we have the clustered features artifact, return the first token
         that matches a real StockCode.
      3. Otherwise fall back to the first 4-12 char alphanumeric token.
    """
    cleaned = re.sub(r"\s+", " ", query.strip())

    explicit_pattern = re.compile(
        r"(?:product|sku|stockcode|stock\s*code|item)\s*[:#-]?\s*([A-Za-z0-9]+)",
        flags=re.IGNORECASE,
    )
    explicit_matches = [m.group(1).strip().upper() for m in explicit_pattern.finditer(cleaned)]

    try:
        feats = _load_clustered_features()
        valid_ids = set(feats.index.astype(str).str.upper())
    except FileNotFoundError:
        valid_ids = set()

    for cand in explicit_matches:
        if not valid_ids or cand in valid_ids:
            return cand

    # Fall back: any 4-12 char alphanumeric run that contains at least one digit.
    fallback_pattern = re.compile(r"\b([A-Za-z0-9]{4,12})\b")
    for match in fallback_pattern.finditer(cleaned):
        token = match.group(1).strip().upper()
        if not any(ch.isdigit() for ch in token):
            continue
        if valid_ids and token not in valid_ids:
            continue
        return token

    return None


def parse_forecast_request(
    query: str,
    default_country: str = "United Kingdom",
    default_horizon: int = 12,
) -> Dict[str, Any]:
    """Convert a free-text query into structured forecast arguments.

    Returns a dict with keys ``product_id``, ``country``, ``horizon_weeks``
    plus an ``ok`` flag and an ``error`` string describing what went wrong.
    """
    if not query or not str(query).strip():
        return {
            "ok": False,
            "error": "Empty query.",
            "product_id": None,
            "country": default_country,
            "horizon_weeks": default_horizon,
        }

    product_id = _extract_product_id(query)
    country = _extract_country(query, default_country=default_country)
    horizon = _extract_horizon(query, default_horizon=default_horizon)

    if product_id is None:
        return {
            "ok": False,
            "error": (
                "I could not find a product / SKU / StockCode in your message. "
                "Try something like 'Forecast product 85123A in the UK for the next 12 weeks'."
            ),
            "product_id": None,
            "country": country,
            "horizon_weeks": horizon,
        }

    return {
        "ok": True,
        "error": None,
        "product_id": product_id,
        "country": country,
        "horizon_weeks": horizon,
    }


# ---------------------------------------------------------------------------
# 2. Cluster lookup
# ---------------------------------------------------------------------------
_RELEVANT_FEATURE_COLS = [
    "mean_weekly_sales",
    "median_weekly_sales",
    "total_sales",
    "cv",
    "pct_zero_weeks",
    "q4_pct",
    "q1_pct",
    "seasonal_conc",
    "trend_log_diff",
    "mean_price",
    "n_unique_customers",
    "n_countries",
    "top_country_pct",
    "cancel_rate",
]


def lookup_product_cluster(product_id: str) -> Dict[str, Any]:
    """Return the cluster assignment + key features for a product."""
    pid = str(product_id).strip().upper()
    feats = _load_clustered_features()

    upper_index = feats.index.astype(str).str.upper()
    if pid not in set(upper_index):
        return {
            "ok": False,
            "error": f"Product '{pid}' was not found in product_features_clustered.csv.",
            "product_id": pid,
        }

    row = feats.loc[upper_index == pid].iloc[0]
    cluster_id = int(row["cluster"]) if "cluster" in feats.columns else None
    cluster_label = str(row["cluster_label"]) if "cluster_label" in feats.columns else "unknown"

    selected_features: Dict[str, Any] = {}
    for col in _RELEVANT_FEATURE_COLS:
        if col in feats.columns:
            value = row[col]
            if isinstance(value, (np.floating, float)):
                if not np.isfinite(value):
                    selected_features[col] = None
                else:
                    selected_features[col] = round(float(value), 4)
            elif isinstance(value, (np.integer, int)):
                selected_features[col] = int(value)
            else:
                selected_features[col] = None if pd.isna(value) else value

    return {
        "ok": True,
        "error": None,
        "product_id": pid,
        "cluster": cluster_id,
        "cluster_label": cluster_label,
        "features": selected_features,
    }


# ---------------------------------------------------------------------------
# 3. Best model lookup
# ---------------------------------------------------------------------------
def _candidate_models_for_cluster(cluster_id: int) -> List[Dict[str, Any]]:
    try:
        candidates = _load_candidate_metrics()
    except FileNotFoundError:
        return []
    if "cluster" not in candidates.columns:
        return []
    cluster_rows = candidates[candidates["cluster"] == cluster_id].copy()
    if cluster_rows.empty:
        return []
    if "valid_mape" in cluster_rows.columns:
        cluster_rows = cluster_rows.sort_values("valid_mape")
    out: List[Dict[str, Any]] = []
    for _, r in cluster_rows.iterrows():
        out.append(
            {
                "candidate": str(r.get("candidate")),
                "valid_mape": _to_jsonable(r.get("valid_mape")),
                "n_valid": _to_jsonable(r.get("n_valid")),
            }
        )
    return out


def _to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (pd.Timestamp,)):
        return str(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def get_best_model(product_id: str) -> Dict[str, Any]:
    """Return the artifact-driven best model for the product's cluster.

    The selection is *always* derived from
    ``best_model_params.json`` and ``cluster_model_selection_summary.csv``.
    """
    cluster_info = lookup_product_cluster(product_id)
    if not cluster_info["ok"]:
        return cluster_info

    cluster_id = int(cluster_info["cluster"])

    selection = _load_selection_summary()
    selection_row = selection.loc[selection["cluster"] == cluster_id]
    if selection_row.empty:
        return {
            "ok": False,
            "error": f"No model selection row for cluster {cluster_id} in cluster_model_selection_summary.csv.",
            "product_id": cluster_info["product_id"],
            "cluster": cluster_id,
            "cluster_label": cluster_info["cluster_label"],
        }
    sel = selection_row.iloc[0]

    best_payload = _load_best_model_params()
    cluster_config: Dict[str, Any] = {}
    for cfg in best_payload.get("cluster_configs", []):
        if int(cfg.get("cluster", -999)) == cluster_id:
            cluster_config = cfg
            break

    selected_model = (
        cluster_config.get("selected_model")
        or sel.get("selected_model")
        or "Unknown"
    )

    return {
        "ok": True,
        "error": None,
        "product_id": cluster_info["product_id"],
        "cluster": cluster_id,
        "cluster_label": cluster_info["cluster_label"],
        "selected_model": str(selected_model),
        "selected_params": cluster_config.get("params", {}),
        "test_mape_selected": _to_jsonable(sel.get("test_mape_selected")),
        "valid_mape_selected": _to_jsonable(sel.get("valid_mape_selected")),
        "test_mape_lgbm_baseline": _to_jsonable(sel.get("test_mape_lgbm_baseline")),
        "n_train": _to_jsonable(sel.get("n_train")),
        "n_valid": _to_jsonable(sel.get("n_valid")),
        "n_test": _to_jsonable(sel.get("n_test")),
        "candidate_models": _candidate_models_for_cluster(cluster_id),
        "model_explanation": MODEL_EXPLANATIONS.get(
            str(selected_model),
            f"{selected_model} is the cluster's saved best model.",
        ),
    }


# ---------------------------------------------------------------------------
# 4. Run forecast
# ---------------------------------------------------------------------------
def _forecast_to_dict(forecast: ForecastOutput, horizon_weeks: int) -> Dict[str, Any]:
    payload = forecast.to_dict()
    # The pipeline's natural language is "12 weeks" but it actually emits
    # ``horizon_weeks`` rows, so just expose the requested horizon.
    payload["forecast_horizon_weeks"] = int(horizon_weeks)

    forecast_rows = payload.get("forecast_12_weeks", []) or []
    if forecast_rows:
        values = [float(r.get("forecast_sales", 0.0)) for r in forecast_rows]
        payload["forecast_summary"] = {
            "mean_forecast_sales": round(float(np.mean(values)), 2) if values else 0.0,
            "peak_forecast_sales": round(float(np.max(values)), 2) if values else 0.0,
            "total_forecast_sales": round(float(np.sum(values)), 2) if values else 0.0,
            "n_weeks": len(values),
        }
    else:
        payload["forecast_summary"] = {
            "mean_forecast_sales": 0.0,
            "peak_forecast_sales": 0.0,
            "total_forecast_sales": 0.0,
            "n_weeks": 0,
        }
    return payload


def _try_load_cached_forecast(product_id: str, country: str) -> Optional[Dict[str, Any]]:
    """If ``fortunetellers_artifacts/pipeline/forecast_<pid>_<country>.json``
    exists, hydrate it as a forecast payload.

    Used as a graceful fallback when the raw transaction data is missing -
    the chatbot can still answer for products whose forecasts were saved
    by a previous pipeline run.
    """
    paths = _project_paths()
    safe_country = str(country).replace(" ", "_")
    candidate = paths.pipeline_dir / f"forecast_{product_id}_{safe_country}.json"
    if not candidate.exists():
        return None
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except Exception:
        return None

    forecast_rows = payload.get("forecast_12_weeks") or []
    values = [float(r.get("forecast_sales", 0.0)) for r in forecast_rows]
    payload["forecast_horizon_weeks"] = len(values)
    payload["forecast_summary"] = {
        "mean_forecast_sales": round(float(np.mean(values)), 2) if values else 0.0,
        "peak_forecast_sales": round(float(np.max(values)), 2) if values else 0.0,
        "total_forecast_sales": round(float(np.sum(values)), 2) if values else 0.0,
        "n_weeks": len(values),
    }
    payload["ok"] = True
    payload["error"] = None
    payload["from_cached_artifact"] = str(candidate)
    return payload


def run_forecast(
    product_id: str,
    country: str = "United Kingdom",
    horizon_weeks: int = 12,
) -> Dict[str, Any]:
    """Run the existing ForecastingPipeline and return a JSON-safe dict.

    Errors (missing product, missing country) are returned as
    ``{"ok": False, "error": "..."}`` rather than being raised, so the
    Streamlit UI can render them inline.
    """
    pid = str(product_id).strip().upper()
    horizon_weeks = max(1, int(horizon_weeks))
    country = str(country).strip() if country else "United Kingdom"

    try:
        pipeline = get_pipeline()
    except FileNotFoundError as exc:
        cached = _try_load_cached_forecast(pid, country)
        if cached is not None:
            cached["warning"] = (
                "The raw transaction data is missing, so this forecast was "
                "loaded from a previously saved artifact "
                f"({cached.get('from_cached_artifact')}). Restore the raw "
                "data to run new forecasts."
            )
            _LAST_FORECAST_BY_PRODUCT[pid] = cached
            return cached
        return {"ok": False, "error": str(exc), "product_id": pid}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"Failed to initialise the forecasting pipeline: {exc}", "product_id": pid}

    try:
        forecast: ForecastOutput = pipeline.forecast_product(
            pid, country=country, horizon=horizon_weeks
        )
    except ValueError as exc:
        return {"ok": False, "error": str(exc), "product_id": pid}
    except KeyError as exc:
        return {"ok": False, "error": f"Cluster configuration missing: {exc}", "product_id": pid}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"Forecast pipeline error: {exc}", "product_id": pid}

    payload = _forecast_to_dict(forecast, horizon_weeks)
    payload["ok"] = True
    payload["error"] = None

    # Persist for ``get_last_forecast`` so the UI can re-render charts.
    _LAST_FORECAST_BY_PRODUCT[pid] = payload
    return payload


def get_last_forecast(product_id: str) -> Optional[Dict[str, Any]]:
    """Return the most recent forecast we ran for this product, if any."""
    if not product_id:
        return None
    return _LAST_FORECAST_BY_PRODUCT.get(str(product_id).strip().upper())


# ---------------------------------------------------------------------------
# Convenience helpers consumed by the UI layer.
# ---------------------------------------------------------------------------
def list_available_products() -> List[str]:
    """List every StockCode known to the clustering artifact."""
    feats = _load_clustered_features()
    return sorted(feats.index.astype(str).unique().tolist())


def list_available_clusters() -> List[Dict[str, Any]]:
    """List clusters with their labels and product counts."""
    feats = _load_clustered_features()
    if "cluster" not in feats.columns:
        return []
    grp = feats.groupby("cluster")
    out: List[Dict[str, Any]] = []
    for cluster_id, sub in grp:
        label = "unknown"
        if "cluster_label" in feats.columns:
            mode = sub["cluster_label"].mode()
            if not mode.empty:
                label = str(mode.iloc[0])
        out.append(
            {
                "cluster": int(cluster_id),
                "cluster_label": label,
                "n_products": int(len(sub)),
            }
        )
    out.sort(key=lambda r: r["cluster"])
    return out


_DEFAULT_COUNTRY_LIST = [
    "ALL",
    "United Kingdom",
    "Germany",
    "France",
    "Spain",
    "Netherlands",
    "Belgium",
    "Switzerland",
    "Portugal",
    "Australia",
    "Italy",
    "Sweden",
    "Norway",
    "EIRE",
    "Finland",
    "Denmark",
    "Channel Islands",
    "Iceland",
    "Cyprus",
    "Greece",
    "Israel",
    "USA",
    "Canada",
    "Japan",
    "Hong Kong",
    "Singapore",
]


def list_available_countries(product_id: Optional[str] = None) -> List[str]:
    """List the countries available for the chatbot UI.

    Tries hardest to use the live cleaned-retail dataset, but falls back to
    cached forecast JSONs and finally to a static default list so the
    Streamlit page never crashes when the raw transaction data is missing.
    """
    # 1. Live data path - only works if the pipeline can initialise.
    try:
        pipeline = get_pipeline()
        df = pipeline.total_retail
        if product_id:
            sub = df[df["StockCode"].astype(str).str.upper() == str(product_id).upper()]
            countries = sorted(sub["Country"].astype(str).unique().tolist())
        else:
            countries = sorted(df["Country"].astype(str).unique().tolist())
        if countries:
            return ["ALL", *countries]
    except FileNotFoundError:
        pass
    except Exception:  # noqa: BLE001
        pass

    # 2. Cached-forecast path - inspect filenames in pipeline_dir.
    cached = sorted(
        {
            entry["country"]
            for entry in list_cached_forecasts()
            if not product_id
            or entry["product_id"].upper() == str(product_id).upper()
        }
    )
    if cached:
        ordered = ["ALL"] + [c for c in cached if c != "ALL"]
        return ordered

    # 3. Static fallback so the UI still has a usable selectbox.
    return list(_DEFAULT_COUNTRY_LIST)


def list_cached_forecasts() -> List[Dict[str, str]]:
    """Enumerate forecast JSONs already saved by previous pipeline runs."""
    paths = _project_paths()
    if not paths.pipeline_dir.exists():
        return []
    out: List[Dict[str, str]] = []
    for fpath in sorted(paths.pipeline_dir.glob("forecast_*.json")):
        stem = fpath.stem  # e.g. forecast_85123A_United_Kingdom
        if not stem.startswith("forecast_"):
            continue
        rest = stem[len("forecast_"):]
        if "_" not in rest:
            continue
        product_id, country_token = rest.split("_", 1)
        country = country_token.replace("_", " ")
        out.append(
            {
                "product_id": product_id,
                "country": country,
                "path": str(fpath),
            }
        )
    return out


def is_pipeline_ready() -> bool:
    """Return ``True`` if ForecastingPipeline can be initialised right now.

    Checks just the file prerequisites; does not actually build the pipeline.
    """
    paths = _project_paths()
    cleaned_ok = paths.cleaned_retail_csv.exists() and paths.cleaned_products_csv.exists()
    return cleaned_ok or paths.raw_excel.exists()


def pipeline_setup_message() -> str:
    """One-line, user-friendly description of what is missing for the pipeline."""
    paths = _project_paths()
    return (
        "Live forecasting is currently unavailable because both the cleaned "
        f"transaction data ({paths.cleaned_retail_csv}) and the raw Excel "
        f"({paths.raw_excel}) are missing. The dashboard will run on saved "
        "artifacts where possible, but new forecasts cannot be generated "
        "until one of those files is restored."
    )


def get_cluster_metadata() -> Dict[str, Any]:
    """Return the cluster metadata JSON (label dictionary, counts, etc.)."""
    return _load_cluster_metadata()


def get_selection_summary_df() -> pd.DataFrame:
    return _load_selection_summary().copy()


def get_candidate_metrics_df() -> pd.DataFrame:
    return _load_candidate_metrics().copy()


def get_clustered_features_df() -> pd.DataFrame:
    return _load_clustered_features().copy()


def explain_model(model_name: str) -> str:
    """Return a one-paragraph explanation for the given model name."""
    if not model_name:
        return "No model name provided."
    return MODEL_EXPLANATIONS.get(
        str(model_name),
        f"{model_name} is the project's saved best model for this cluster.",
    )


def explain_mape() -> str:
    return MAPE_EXPLANATION


def build_assistant_payload(
    parsed: Dict[str, Any],
    forecast: Dict[str, Any],
    best_model: Dict[str, Any],
) -> Dict[str, Any]:
    """Compose a structured response for the chatbot UI.

    Combines the parsed query, the artifact-driven model selection rationale
    and the forecast output into a single dict the Streamlit page renders.
    """
    summary = forecast.get("forecast_summary", {})
    mape = forecast.get("test_mape_selected")
    horizon_weeks = forecast.get("forecast_horizon_weeks", parsed.get("horizon_weeks"))

    rationale_parts: List[str] = []
    cluster = forecast.get("cluster")
    cluster_label = forecast.get("cluster_label", "")
    selected_model = forecast.get("selected_model", "Unknown")

    if cluster is not None:
        rationale_parts.append(
            f"Product {forecast.get('product_id')} belongs to cluster {cluster} "
            f"({cluster_label})."
        )
    rationale_parts.append(
        f"For this cluster the saved model selection artifact chose "
        f"{selected_model}."
    )
    rationale_parts.append(explain_model(selected_model))
    if best_model.get("test_mape_selected") is not None:
        rationale_parts.append(
            f"On the held-out test set this model achieves "
            f"{round(float(best_model['test_mape_selected']), 2)}% MAPE."
        )
    rationale_parts.append(
        f"The agent therefore used {selected_model} to generate the "
        f"{horizon_weeks}-week forecast."
    )

    return {
        "headline": (
            f"{horizon_weeks}-week forecast for product "
            f"{forecast.get('product_id')} in {forecast.get('country')}"
        ),
        "rationale": " ".join(rationale_parts),
        "mape_explanation": explain_mape(),
        "fields": {
            "product_id": forecast.get("product_id"),
            "product_description": forecast.get("product_description"),
            "country": forecast.get("country"),
            "cluster": cluster,
            "cluster_label": cluster_label,
            "selected_model": selected_model,
            "test_mape_selected": mape,
            "forecast_horizon_weeks": horizon_weeks,
            "mean_forecast_sales": summary.get("mean_forecast_sales"),
            "peak_forecast_sales": summary.get("peak_forecast_sales"),
            "total_forecast_sales": summary.get("total_forecast_sales"),
            "warning_flag": forecast.get("warning_flag"),
            "recommendation": forecast.get("recommendation"),
        },
    }
