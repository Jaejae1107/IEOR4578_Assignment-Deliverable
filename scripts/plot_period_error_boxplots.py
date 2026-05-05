"""Generate per-cluster signed-error boxplots by test period (P1/P2/P3)."""
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

PRED_CSV = Path("fortunetellers_artifacts/modeling/test_predictions.csv")
OUT_DIR = Path("fortunetellers_artifacts/modeling")

CLUSTER_ORDER = [-2, -1, 0, 1, 2]
CLUSTER_LABELS = {
    -2: "Cluster -2 — Truly Sporadic",
    -1: "Cluster -1 — Intermittent",
     0: "Cluster 0 — High Cancellation Risk",
     1: "Cluster 1 — Volatile Mid-Range",
     2: "Cluster 2 — Steady Regulars",
}
FIG_NAMES = {
    -2: "period_errors_cluster_m2.png",
    -1: "period_errors_cluster_m1.png",
     0: "period_errors_cluster_0.png",
     1: "period_errors_cluster_1.png",
     2: "period_errors_cluster_2.png",
}

df = pd.read_csv(PRED_CSV)
df = df.dropna(subset=["signed_pct_error"])

for cluster_id in CLUSTER_ORDER:
    cdf = df[df["cluster"] == cluster_id]
    if cdf.empty:
        print(f"No data for cluster {cluster_id}, skipping.")
        continue

    groups = [cdf.loc[cdf["period"] == p, "signed_pct_error"].clip(-300, 300).values for p in ["P1", "P2", "P3"]]

    fig, ax = plt.subplots(figsize=(7, 4))
    bp = ax.boxplot(
        groups,
        labels=["P1 (Weeks 1–4)", "P2 (Weeks 5–8)", "P3 (Weeks 9–12)"],
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
        flierprops=dict(marker="o", markersize=2, alpha=0.3, linestyle="none"),
        whis=1.5,
    )
    colors = ["#AEC6E8", "#7BAFD4", "#4A90C4"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", label="Perfect forecast")
    ax.set_ylabel("Signed Percentage Error (%)")
    ax.set_title(CLUSTER_LABELS[cluster_id])
    ax.legend(fontsize=8)
    fig.tight_layout()

    out_path = OUT_DIR / FIG_NAMES[cluster_id]
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
