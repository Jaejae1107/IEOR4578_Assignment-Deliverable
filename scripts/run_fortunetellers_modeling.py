from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fortunetellers.config import ProjectPaths, default_raw_excel_path
from fortunetellers.data import load_or_prepare_transactions
from fortunetellers.features import build_cluster_panels, build_or_load_feature_artifacts
from fortunetellers.modeling import train_cluster_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FortuneTellers cluster modeling.")
    parser.add_argument("--raw-excel", default=str(default_raw_excel_path()))
    parser.add_argument("--artifact-root", default=str(ROOT / "fortunetellers_artifacts"))
    parser.add_argument("--rebuild-features", action="store_true", help="Rebuild feature artifacts before modeling.")
    args = parser.parse_args()

    paths = ProjectPaths(raw_excel=Path(args.raw_excel), artifact_root=Path(args.artifact_root))
    dataset = load_or_prepare_transactions(paths)
    feature_artifacts = build_or_load_feature_artifacts(dataset, paths, rebuild=args.rebuild_features)
    panels = build_cluster_panels(feature_artifacts.feat_df_all, dataset)
    modeling_artifacts = train_cluster_models(feature_artifacts.feat_df_all, panels, dataset, paths)

    print("FortuneTellers cluster modeling complete.")
    print(f"  Selection summary: {paths.selection_summary_csv}")
    print(f"  Candidate metrics: {paths.candidate_metrics_csv}")
    print(f"  Tuned LGBM trials: {paths.tuned_lgbm_trials_csv}")
    print(f"  Best model params: {paths.best_model_params_json}")
    print("\nSelected models:")
    show_cols = [
        "cluster",
        "label",
        "selected_model",
        "valid_mape_selected",
        "test_mape_selected",
        "test_mape_lgbm_baseline",
    ]
    print(modeling_artifacts.selection_df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
