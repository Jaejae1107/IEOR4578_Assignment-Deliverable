"""Standalone test-set evaluation for experimental candidates.

Forces each candidate model to predict on the held-out test set
regardless of whether it won model selection on validation. Use this
to get a complete apples-to-apples comparison across experiments.

Usage:
    .venv/bin/python scripts/run_experiment_test_eval.py \\
        --raw-excel /path/to/online_retail_II.xlsx \\
        --clusters -2 -1

Candidates evaluated per cluster:
    RF_Default        — the pipeline baseline
    TwoStageRawLag    — LGBM hurdle model (occurrence + amount)
    DeepAR            — global LSTM with NegBin distribution loss
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from fortunetellers.config import ProjectPaths, default_raw_excel_path
from fortunetellers.data import load_or_prepare_transactions
from fortunetellers.features import build_cluster_panels, build_or_load_feature_artifacts
from fortunetellers.modeling import (
    BASELINE_RF_PARAMS,
    _pred_rf_signedlog,
    deepar_predict,
    mape_100,
    split_train_valid_time,
    two_stage_predict,
)


def evaluate_cluster(
    cluster_id: int,
    panels: dict,
    feat_df_all: pd.DataFrame,
    dataset,
    candidates: list[str],
) -> list[dict]:
    label = panels[cluster_id]["label"]
    full_train = panels[cluster_id]["train"]
    test_df = panels[cluster_id]["test"]
    feat_cols = panels[cluster_id]["features"]

    train_df, valid_df = split_train_valid_time(full_train)
    train_plus_valid = pd.concat([train_df, valid_df], ignore_index=True)
    h_test = test_df["week"].nunique()

    rows = []
    for candidate in candidates:
        print(f"  [{cluster_id}] {candidate} ...", end=" ", flush=True)
        try:
            if candidate == "RF_Default":
                y_pred = _pred_rf_signedlog(train_plus_valid, test_df, feat_cols, BASELINE_RF_PARAMS)
            elif candidate == "TwoStageRawLag":
                y_pred = two_stage_predict(cluster_id, train_plus_valid, test_df, feat_cols, feat_df_all, dataset)
            elif candidate == "DeepAR":
                y_pred = deepar_predict(train_plus_valid, test_df, h=h_test)
            else:
                raise ValueError(f"Unknown candidate: {candidate}")

            test_mape, n_test = mape_100(test_df["sales"].values, y_pred)
            print(f"MAPE={test_mape:.2f}%  (n={n_test})")
        except Exception as exc:
            test_mape, n_test = float("nan"), 0
            print(f"ERROR: {exc}")

        rows.append({
            "cluster": cluster_id,
            "label": label,
            "candidate": candidate,
            "test_mape": round(float(test_mape), 2) if np.isfinite(test_mape) else float("nan"),
            "n_test": n_test,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Force test-set evaluation for experimental candidates.")
    parser.add_argument("--raw-excel", default=str(default_raw_excel_path()))
    parser.add_argument("--artifact-root", default=str(ROOT / "fortunetellers_artifacts"))
    parser.add_argument("--clusters", nargs="+", type=int, default=[-2, -1],
                        help="Cluster IDs to evaluate (default: -2 -1)")
    parser.add_argument("--candidates", nargs="+",
                        default=["RF_Default", "TwoStageRawLag", "DeepAR"],
                        help="Candidates to force-evaluate on test set")
    args = parser.parse_args()

    paths = ProjectPaths(raw_excel=Path(args.raw_excel), artifact_root=Path(args.artifact_root))
    dataset = load_or_prepare_transactions(paths)
    fa = build_or_load_feature_artifacts(dataset, paths)
    panels = build_cluster_panels(fa.feat_df_all, dataset)

    all_rows: list[dict] = []
    for cluster_id in args.clusters:
        if cluster_id not in panels:
            print(f"Cluster {cluster_id} not found — skipping.")
            continue
        print(f"\nCluster {cluster_id} ({panels[cluster_id]['label']}):")
        all_rows.extend(evaluate_cluster(cluster_id, panels, fa.feat_df_all, dataset, args.candidates))

    results = pd.DataFrame(all_rows)
    pivot = results.pivot(index=["cluster", "label"], columns="candidate", values="test_mape")
    if "RF_Default" in pivot.columns:
        for col in [c for c in pivot.columns if c != "RF_Default"]:
            pivot[f"Delta_{col}"] = (pivot[col] - pivot["RF_Default"]).round(2)

    print("\n=== Test MAPE by Candidate ===")
    print(pivot.to_string())

    out_path = paths.modeling_dir / "experiment_test_eval.csv"
    results.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
