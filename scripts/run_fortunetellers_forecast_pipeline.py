from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fortunetellers.config import ProjectPaths, default_raw_excel_path
from fortunetellers.pipeline import ForecastingPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FortuneTellers forecasting pipeline for a product query.")
    parser.add_argument("--product-id", required=True, help="StockCode to forecast, for example 85123A")
    parser.add_argument("--country", default="United Kingdom", help="Country filter or ALL")
    parser.add_argument("--horizon", type=int, default=12, help="Number of future weeks to forecast")
    parser.add_argument("--raw-excel", default=str(default_raw_excel_path()))
    parser.add_argument("--artifact-root", default=str(ROOT / "fortunetellers_artifacts"))
    parser.add_argument("--output-json", default=None, help="Optional explicit path for the forecast JSON output")
    args = parser.parse_args()

    paths = ProjectPaths(raw_excel=Path(args.raw_excel), artifact_root=Path(args.artifact_root))
    pipeline = ForecastingPipeline(paths)
    forecast = pipeline.forecast_product(args.product_id, country=args.country, horizon=args.horizon)
    output_path = pipeline.save_forecast(forecast, args.output_json)

    print("FortuneTellers forecasting pipeline complete.")
    print(f"  Product: {forecast.product_id} - {forecast.product_description}")
    print(f"  Country: {forecast.country}")
    print(f"  Cluster: {forecast.cluster} ({forecast.cluster_label})")
    print(f"  Selected model: {forecast.selected_model}")
    print(f"  Test MAPE: {forecast.test_mape_selected}")
    print(f"  Warning flag: {forecast.warning_flag}")
    print(f"  Recommendation: {forecast.recommendation}")
    print(f"  Output JSON: {output_path}")
    print("\nForecast preview:")
    print(json.dumps(forecast.forecast_12_weeks[:4], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
