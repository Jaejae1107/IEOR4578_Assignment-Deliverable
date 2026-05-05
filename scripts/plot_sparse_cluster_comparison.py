"""Generate Figure 11 equivalent: sparse-cluster model comparison bar chart.

Reads experiment_test_eval.csv and produces a grouped bar chart comparing
RF_Default, TwoStageRawLag, and DeepAR (NegBin) test MAPEs for
Cluster -2 (Truly Sporadic) and Cluster -1 (Intermittent).

Usage:
    python scripts/plot_sparse_cluster_comparison.py
    python scripts/plot_sparse_cluster_comparison.py --artifact-root path/to/artifacts --out path/to/output.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


MODEL_LABELS = {
    "RF_Default": "RF Default",
    "TwoStageRawLag": "Two-Stage Raw-Lag",
    "DeepAR": "DeepAR (NegBin)",
}

CLUSTER_LABELS = {
    -2: "Cluster -2\n(Truly Sporadic)",
    -1: "Cluster -1\n(Intermittent)",
}

COLORS = {
    "RF_Default": "#2E75B6",
    "TwoStageRawLag": "#ED7D31",
    "DeepAR": "#A9D18E",
}


def load_data(artifact_root: Path) -> pd.DataFrame:
    csv_path = artifact_root / "modeling" / "experiment_test_eval.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"experiment_test_eval.csv not found at {csv_path}.\n"
            "Run scripts/run_experiment_test_eval.py first."
        )
    df = pd.read_csv(csv_path)
    df = df[df["cluster"].isin([-2, -1])].copy()
    return df


def plot(df: pd.DataFrame, out_path: Path) -> None:
    candidates = ["RF_Default", "TwoStageRawLag", "DeepAR"]
    clusters = [-2, -1]

    n_clusters = len(clusters)
    n_models = len(candidates)
    bar_width = 0.22
    group_gap = 0.75
    x = np.arange(n_clusters) * group_gap

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * bar_width

    for offset, candidate in zip(offsets, candidates):
        values = []
        for cluster_id in clusters:
            row = df[(df["cluster"] == cluster_id) & (df["candidate"] == candidate)]
            values.append(float(row["test_mape"].values[0]) if len(row) else float("nan"))

        bars = ax.bar(
            x + offset,
            values,
            width=bar_width,
            color=COLORS[candidate],
            label=MODEL_LABELS[candidate],
            zorder=3,
            edgecolor="white",
            linewidth=0.5,
        )

        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f"{val:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#333333",
                )

    ax.set_xticks(x)
    ax.set_xticklabels([CLUSTER_LABELS[c] for c in clusters], fontsize=10)
    ax.set_ylabel("Test MAPE (%)", fontsize=10)
    ax.set_ylim(0, max(df["test_mape"].max() * 1.28, 115))
    ax.yaxis.grid(True, color="#e0e0e0", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")

    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        fontsize=9,
        edgecolor="#cccccc",
    )

    ax.set_title(
        "Sparse-Demand Clusters: Three-Model Test MAPE Comparison",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot sparse-cluster model comparison.")
    parser.add_argument(
        "--artifact-root",
        default=str(ROOT / "fortunetellers_artifacts"),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG path (default: <artifact-root>/modeling/sparse_cluster_comparison.png)",
    )
    args = parser.parse_args()

    artifact_root = Path(args.artifact_root)
    out_path = Path(args.out) if args.out else artifact_root / "modeling" / "sparse_cluster_comparison.png"

    df = load_data(artifact_root)
    plot(df, out_path)


if __name__ == "__main__":
    main()
