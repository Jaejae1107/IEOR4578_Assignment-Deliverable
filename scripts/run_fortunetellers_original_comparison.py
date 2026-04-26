from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fortunetellers.config import BASELINE_RF_PARAMS, BEST_C2_RF_PARAMS, ProjectPaths, default_raw_excel_path
from fortunetellers.data import load_or_prepare_transactions
from fortunetellers.features import build_cluster_panels, build_or_load_feature_artifacts
from fortunetellers.modeling import _pred_rf_signedlog, mape_100, split_train_valid_time


ORIGINAL_NOTEBOOK_BENCHMARK = [
    {
        "cluster": -2,
        "label": "Truly sporadic",
        "cluster_count": 1328,
        "selected_model": "RF_Default",
        "valid_mape_selected": 78.6146,
        "test_mape_selected": 89.0379,
        "test_mape_lgbm_baseline": 87.1572,
        "n_train": 34528,
        "n_valid": 5312,
        "n_test": 15936,
    },
    {
        "cluster": -1,
        "label": "Intermittent (Croston)",
        "cluster_count": 1627,
        "selected_model": "RF_Default",
        "valid_mape_selected": 86.9088,
        "test_mape_selected": 76.3429,
        "test_mape_lgbm_baseline": 78.7736,
        "n_train": 42302,
        "n_valid": 6508,
        "n_test": 19524,
    },
    {
        "cluster": 0,
        "label": "High cancellation risk",
        "cluster_count": 92,
        "selected_model": "RF_Default",
        "valid_mape_selected": 76.5318,
        "test_mape_selected": 82.8887,
        "test_mape_lgbm_baseline": 102.1911,
        "n_train": 2392,
        "n_valid": 368,
        "n_test": 1104,
    },
    {
        "cluster": 1,
        "label": "Volatile mid-range",
        "cluster_count": 228,
        "selected_model": "RF_Default",
        "valid_mape_selected": 79.1589,
        "test_mape_selected": 75.0641,
        "test_mape_lgbm_baseline": 80.4716,
        "n_train": 5928,
        "n_valid": 912,
        "n_test": 2736,
    },
    {
        "cluster": 2,
        "label": "Steady regulars",
        "cluster_count": 1281,
        "selected_model": "RF_C2_BEST",
        "valid_mape_selected": 84.1653,
        "test_mape_selected": 81.3974,
        "test_mape_lgbm_baseline": 122.6299,
        "n_train": 33306,
        "n_valid": 5124,
        "n_test": 15372,
    },
]


def evaluate_current_rf_benchmark(
    paths: ProjectPaths,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = load_or_prepare_transactions(paths)
    feature_artifacts = build_or_load_feature_artifacts(dataset, paths, rebuild=False)
    panels = build_cluster_panels(feature_artifacts.feat_df_all, dataset)

    current_rows: list[dict[str, float | int | str]] = []
    candidate_rows: list[dict[str, float | int | str]] = []
    cluster_counts = feature_artifacts.feat_df_all["cluster"].value_counts().sort_index()

    for cluster_id in sorted(panels):
        label = panels[cluster_id]["label"]
        full_train = panels[cluster_id]["train"].sort_values(["week", "StockCode"]).copy()
        test_df = panels[cluster_id]["test"].sort_values(["week", "StockCode"]).copy()
        feat_cols = list(panels[cluster_id]["features"])

        train_df, valid_df = split_train_valid_time(full_train, valid_ratio=0.15, min_valid_weeks=4)

        candidates = [("RF_Default", BASELINE_RF_PARAMS)]
        if cluster_id == 2:
            candidates.insert(0, ("RF_C2_BEST", BEST_C2_RF_PARAMS))

        best_method = None
        best_params = None
        best_valid_mape = float("inf")

        for method, params in candidates:
            y_valid_pred = _pred_rf_signedlog(train_df, valid_df, feat_cols, params)
            valid_mape, n_valid_used = mape_100(valid_df["sales"].values, y_valid_pred)
            candidate_rows.append(
                {
                    "cluster": cluster_id,
                    "label": label,
                    "candidate": method,
                    "valid_mape": valid_mape,
                    "n_valid_used": n_valid_used,
                }
            )
            if pd.notna(valid_mape) and n_valid_used > 0 and valid_mape < best_valid_mape:
                best_valid_mape = float(valid_mape)
                best_method = method
                best_params = params

        if best_method is None or best_params is None:
            best_method = "RF_Default"
            best_params = BASELINE_RF_PARAMS
            best_valid_mape = float("nan")

        train_plus_valid = pd.concat([train_df, valid_df], ignore_index=True)
        y_test_pred = _pred_rf_signedlog(train_plus_valid, test_df, feat_cols, best_params)
        test_mape, n_test_used = mape_100(test_df["sales"].values, y_test_pred)

        current_rows.append(
            {
                "cluster": cluster_id,
                "label": label,
                "cluster_count": int(cluster_counts.loc[cluster_id]),
                "selected_model": best_method,
                "valid_mape_selected": float(best_valid_mape) if pd.notna(best_valid_mape) else float("nan"),
                "test_mape_selected": float(test_mape) if pd.notna(test_mape) else float("nan"),
                "n_train": int(len(train_df)),
                "n_valid": int(len(valid_df)),
                "n_test": int(len(test_df)),
                "n_test_used": int(n_test_used),
            }
        )

    return pd.DataFrame(current_rows), pd.DataFrame(candidate_rows)


def build_comparison_report(original_df: pd.DataFrame, current_df: pd.DataFrame) -> pd.DataFrame:
    merged = original_df.merge(
        current_df,
        on="cluster",
        how="inner",
        suffixes=("_original", "_current"),
    )
    merged["label_match"] = merged["label_original"] == merged["label_current"]
    merged["count_match"] = merged["cluster_count_original"] == merged["cluster_count_current"]
    merged["test_mape_delta_current_minus_original"] = (
        merged["test_mape_selected_current"] - merged["test_mape_selected_original"]
    )
    return merged.sort_values("cluster").reset_index(drop=True)


def write_markdown_summary(
    comparison_df: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        "# FortuneTellers Original vs Current Comparison",
        "",
        "This report compares the original notebook benchmark against the current rerun after restoring the original clustering split.",
        "",
        "Notes:",
        "- Original benchmark comes from the source notebook's unified pipeline output.",
        "- Current rerun uses the restored original cluster definitions and a pooled RandomForest benchmark per cluster.",
        "- Current MAPE is therefore useful for directional comparison, not a perfect apples-to-apples replacement for every original candidate family.",
        "",
        "## Cluster Alignment Check",
        "",
        "| Cluster | Original Label | Current Label | Original Count | Current Count | Label Match | Count Match |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in comparison_df.itertuples(index=False):
        lines.append(
            f"| {row.cluster} | {row.label_original} | {row.label_current} | "
            f"{row.cluster_count_original} | {row.cluster_count_current} | "
            f"{'Yes' if row.label_match else 'No'} | {'Yes' if row.count_match else 'No'} |"
        )

    lines.extend(
        [
            "",
            "## Model Comparison",
            "",
            "| Cluster | Original Model | Original Test MAPE | Current Model | Current Test MAPE | Delta (Current - Original) |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )

    for row in comparison_df.itertuples(index=False):
        delta = row.test_mape_delta_current_minus_original
        lines.append(
            f"| {row.cluster} | {row.selected_model_original} | {row.test_mape_selected_original:.2f} | "
            f"{row.selected_model_current} | {row.test_mape_selected_current:.2f} | {delta:.2f} |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare original notebook benchmark with current rerun.")
    parser.add_argument("--raw-excel", default=str(default_raw_excel_path()))
    parser.add_argument("--artifact-root", default=str(ROOT / "fortunetellers_artifacts"))
    args = parser.parse_args()

    paths = ProjectPaths(raw_excel=Path(args.raw_excel), artifact_root=Path(args.artifact_root))
    comparison_dir = paths.artifact_root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    original_df = pd.DataFrame(ORIGINAL_NOTEBOOK_BENCHMARK).sort_values("cluster").reset_index(drop=True)
    current_df, candidate_df = evaluate_current_rf_benchmark(paths)
    comparison_df = build_comparison_report(original_df, current_df)

    reference_json = comparison_dir / "original_notebook_reference.json"
    current_csv = comparison_dir / "current_rf_benchmark.csv"
    candidates_csv = comparison_dir / "current_rf_candidates.csv"
    comparison_csv = comparison_dir / "original_vs_current_comparison.csv"
    comparison_md = comparison_dir / "original_vs_current_comparison.md"

    reference_json.write_text(json.dumps(ORIGINAL_NOTEBOOK_BENCHMARK, indent=2), encoding="utf-8")
    current_df.to_csv(current_csv, index=False)
    candidate_df.to_csv(candidates_csv, index=False)
    comparison_df.to_csv(comparison_csv, index=False)
    write_markdown_summary(comparison_df, comparison_md)

    print("FortuneTellers original-vs-current comparison complete.")
    print(f"  Original benchmark reference: {reference_json}")
    print(f"  Current RF benchmark: {current_csv}")
    print(f"  Current RF candidates: {candidates_csv}")
    print(f"  Comparison CSV: {comparison_csv}")
    print(f"  Comparison report: {comparison_md}")
    print("\nComparison preview:")
    show_cols = [
        "cluster",
        "label_original",
        "cluster_count_original",
        "cluster_count_current",
        "selected_model_original",
        "test_mape_selected_original",
        "selected_model_current",
        "test_mape_selected_current",
        "test_mape_delta_current_minus_original",
    ]
    print(comparison_df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
