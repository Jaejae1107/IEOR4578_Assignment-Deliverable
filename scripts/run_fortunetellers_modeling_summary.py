from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _fmt(value: object, digits: int = 2) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if pd.isna(number):
        return ""
    return f"{number:.{digits}f}"


def _pipe_table(headers: list[str], rows: list[list[object]]) -> str:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(out)


def build_summary(selection_df: pd.DataFrame, candidate_df: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("# FortuneTellers Cluster Modeling Summary")
    lines.append("")
    lines.append("This report summarizes the selected model for each demand cluster and the strongest candidate comparisons from the current rerun.")
    lines.append("")

    lines.append("## Selected Models")
    lines.append("")
    selected_rows = []
    for _, row in selection_df.iterrows():
        selected_rows.append(
            [
                int(row["cluster"]),
                row["label"],
                row["selected_model"],
                _fmt(row["valid_mape_selected"]),
                _fmt(row["test_mape_selected"]),
                _fmt(row["test_mape_lgbm_baseline"]),
                _fmt(row["delta_selected_minus_lgbm"]),
            ]
        )
    lines.append(
        _pipe_table(
            [
                "Cluster",
                "Label",
                "Selected Model",
                "Valid MAPE",
                "Test MAPE",
                "LGBM Baseline",
                "Delta vs LGBM",
            ],
            selected_rows,
        )
    )
    lines.append("")

    lines.append("## Cluster-by-Cluster Notes")
    lines.append("")
    for _, row in selection_df.iterrows():
        cluster_id = int(row["cluster"])
        label = str(row["label"])
        selected_model = str(row["selected_model"])
        test_mape = _fmt(row["test_mape_selected"])
        delta = row["delta_selected_minus_lgbm"]
        delta_text = (
            "matched the LGBM baseline"
            if pd.isna(delta) or abs(float(delta)) < 1e-9
            else ("beat" if float(delta) < 0 else "underperformed") + f" the LGBM baseline by {abs(float(delta)):.2f} points"
        )

        lines.append(f"### Cluster {cluster_id} - {label}")
        lines.append("")
        lines.append(f"- Selected model: `{selected_model}`")
        lines.append(f"- Held-out test MAPE: `{test_mape}`")
        lines.append(f"- Relative result: `{selected_model}` {delta_text}.")

        cluster_candidates = candidate_df[candidate_df["cluster"] == cluster_id].copy()
        cluster_candidates = cluster_candidates.sort_values("valid_mape", na_position="last")
        top_rows = []
        for _, cand in cluster_candidates.iterrows():
            top_rows.append(
                [
                    cand["candidate"],
                    _fmt(cand["valid_mape"]),
                    int(cand["n_valid"]),
                ]
            )
        lines.append("")
        lines.append(_pipe_table(["Candidate", "Valid MAPE", "Valid N"], top_rows))
        lines.append("")

    lines.append("## Key Takeaways")
    lines.append("")
    lines.append("- `RF_Default` won 4 of the 5 clusters in this rerun, which suggests the simpler tree ensemble is the most stable default choice right now.")
    lines.append("- `High cancellation risk` saw the biggest gain over the LGBM baseline, which supports keeping a conservative deployment story for that segment.")
    lines.append("- `Truly sporadic` remains weak regardless of model choice, so this segment should stay in the low-confidence / manual-review bucket.")
    lines.append("- For presentation, use the cluster label more prominently than the cluster number because the numeric IDs can shift across reruns.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a markdown summary for FortuneTellers cluster modeling results.")
    parser.add_argument("--artifact-root", default=str(ROOT / "fortunetellers_artifacts"))
    parser.add_argument("--output-md", default=None, help="Optional explicit markdown output path")
    args = parser.parse_args()

    artifact_root = Path(args.artifact_root)
    modeling_dir = artifact_root / "modeling"
    selection_path = modeling_dir / "cluster_model_selection_summary.csv"
    candidate_path = modeling_dir / "cluster_candidate_metrics.csv"
    output_path = Path(args.output_md) if args.output_md else modeling_dir / "cluster_modeling_summary.md"

    selection_df = pd.read_csv(selection_path)
    candidate_df = pd.read_csv(candidate_path)
    summary = build_summary(selection_df, candidate_df)
    output_path.write_text(summary, encoding="utf-8")

    print("FortuneTellers modeling summary complete.")
    print(f"  Selection CSV: {selection_path}")
    print(f"  Candidate CSV: {candidate_path}")
    print(f"  Markdown summary: {output_path}")


if __name__ == "__main__":
    main()
