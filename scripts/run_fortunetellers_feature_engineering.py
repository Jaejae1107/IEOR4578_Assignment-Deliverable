from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fortunetellers.config import ProjectPaths, default_raw_excel_path
from fortunetellers.data import load_or_prepare_transactions
from fortunetellers.features import build_or_load_feature_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FortuneTellers feature engineering.")
    parser.add_argument("--raw-excel", default=str(default_raw_excel_path()))
    parser.add_argument("--artifact-root", default=str(ROOT / "fortunetellers_artifacts"))
    parser.add_argument("--rebuild", action="store_true", help="Rebuild features even if saved artifacts exist.")
    args = parser.parse_args()

    paths = ProjectPaths(raw_excel=Path(args.raw_excel), artifact_root=Path(args.artifact_root))
    dataset = load_or_prepare_transactions(paths)
    artifacts = build_or_load_feature_artifacts(dataset, paths, rebuild=args.rebuild)

    print("FortuneTellers feature engineering complete.")
    print(f"  Raw data: {paths.raw_excel}")
    print(f"  Cleaned sales CSV: {paths.cleaned_retail_csv}")
    print(f"  Product features: {paths.product_features_clustered_csv}")
    print(
        "  Clustering split: "
        f"{len(dataset.clustering_training_weeks)} weeks "
        f"({dataset.clustering_training_weeks.min()} to {dataset.clustering_training_weeks.max()})"
    )
    print(f"  Best k: {artifacts.best_k}")
    print("  Silhouette scores:")
    for k, score in sorted(artifacts.silhouette_scores.items()):
        print(f"    k={k}: {score:.4f}")
    print("  Cluster counts:")
    counts = artifacts.feat_df_all["cluster"].value_counts().sort_index()
    for cluster_id, count in counts.items():
        label = artifacts.feat_df_all.loc[artifacts.feat_df_all["cluster"] == cluster_id, "cluster_label"].iloc[0]
        print(f"    {cluster_id:>2}: {label} ({count})")


if __name__ == "__main__":
    main()
