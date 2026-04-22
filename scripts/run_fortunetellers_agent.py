from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fortunetellers.agent import ForecastAgent
from fortunetellers.config import ProjectPaths, default_raw_excel_path
from fortunetellers.pipeline import ForecastingPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FortuneTellers agent wrapper on a natural-language query.")
    parser.add_argument("--query", required=True, help='Natural-language query, for example: "Forecast product 85123A in United Kingdom"')
    parser.add_argument("--default-country", default="United Kingdom", help="Fallback country if the query does not name one")
    parser.add_argument("--horizon", type=int, default=12, help="Number of future weeks to forecast")
    parser.add_argument("--raw-excel", default=str(default_raw_excel_path()))
    parser.add_argument("--artifact-root", default=str(ROOT / "fortunetellers_artifacts"))
    parser.add_argument("--output-json", default=None, help="Optional explicit path for the forecast JSON output")
    parser.add_argument("--response-json", default=None, help="Optional explicit path for the saved agent response JSON")
    args = parser.parse_args()

    paths = ProjectPaths(raw_excel=Path(args.raw_excel), artifact_root=Path(args.artifact_root))
    pipeline = ForecastingPipeline(paths)
    agent = ForecastAgent(pipeline)
    response = agent.answer_query(
        query=args.query,
        default_country=args.default_country,
        horizon=args.horizon,
        output_json_path=args.output_json,
    )
    response_path = agent.save_agent_response(response, args.response_json)

    print("FortuneTellers agent response complete.")
    print(f"  Parsed product: {response.parsed_query.product_id}")
    print(f"  Parsed country: {response.parsed_query.country}")
    print(f"  Forecast JSON: {response.forecast_json_path}")
    print(f"  Agent response JSON: {response_path}")
    print("")
    print(response.reply_text)


if __name__ == "__main__":
    main()
