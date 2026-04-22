from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .pipeline import ForecastOutput, ForecastingPipeline


COUNTRY_ALIASES = {
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "united kingdom": "United Kingdom",
    "great britain": "United Kingdom",
    "britain": "United Kingdom",
    "england": "United Kingdom",
    "all countries": "ALL",
    "all markets": "ALL",
    "all market": "ALL",
    "global": "ALL",
    "worldwide": "ALL",
    "overall": "ALL",
}


@dataclass
class ParsedQuery:
    raw_query: str
    product_id: str
    country: str


@dataclass
class AgentResponse:
    parsed_query: ParsedQuery
    forecast: ForecastOutput
    reply_text: str
    forecast_json_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "parsed_query": {
                "raw_query": self.parsed_query.raw_query,
                "product_id": self.parsed_query.product_id,
                "country": self.parsed_query.country,
            },
            "forecast": self.forecast.to_dict(),
            "reply_text": self.reply_text,
            "forecast_json_path": str(self.forecast_json_path),
        }


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _extract_country(query: str, available_countries: list[str], default_country: str) -> str:
    lowered = query.lower()
    for alias, canonical in COUNTRY_ALIASES.items():
        if alias in lowered:
            return canonical

    for country in sorted(available_countries, key=len, reverse=True):
        if country.lower() in lowered:
            return country

    return default_country


def _extract_product_id(query: str, available_product_ids: set[str]) -> str:
    explicit_patterns = [
        r"(?:product|sku|stockcode|stock code|item)\s*[:#-]?\s*([A-Za-z0-9]+)",
        r"\b([A-Za-z0-9]{4,12})\b",
    ]

    seen: list[str] = []
    for pattern in explicit_patterns:
        for match in re.finditer(pattern, query, flags=re.IGNORECASE):
            candidate = match.group(1).strip().upper()
            if candidate in available_product_ids and candidate not in seen:
                seen.append(candidate)

    if not seen:
        raise ValueError("I could not find a valid product ID in the query.")
    return seen[0]


def parse_agent_query(
    query: str,
    available_product_ids: set[str],
    available_countries: list[str],
    default_country: str = "United Kingdom",
) -> ParsedQuery:
    cleaned = _normalize_whitespace(query)
    product_id = _extract_product_id(cleaned, available_product_ids)
    country = _extract_country(cleaned, available_countries, default_country=default_country)
    return ParsedQuery(raw_query=cleaned, product_id=product_id, country=country)


def format_agent_reply(forecast: ForecastOutput) -> str:
    lines: list[str] = []
    lines.append(f"Product {forecast.product_id}: {forecast.product_description}")
    lines.append(f"Country: {forecast.country}")
    lines.append(
        f"Segment: {forecast.cluster_label} (cluster {forecast.cluster}) | Model: {forecast.selected_model} | Test MAPE: {forecast.test_mape_selected}"
    )
    lines.append(
        f"Recent average weekly sales: {forecast.statistics['recent_12week_avg']} | Total sales: {forecast.statistics['total_sales']} | Zero-sales weeks: {forecast.statistics['pct_zero_weeks']}%"
    )
    lines.append(f"Recommendation: {forecast.recommendation}")
    lines.append(f"Warning level: {forecast.warning_flag}")
    lines.append("")
    lines.append("Next 12-week forecast:")

    for row in forecast.forecast_12_weeks:
        lines.append(f"- {row['week_start']}: {row['forecast_sales']}")

    return "\n".join(lines)


class ForecastAgent:
    def __init__(self, pipeline: ForecastingPipeline) -> None:
        self.pipeline = pipeline
        self.available_product_ids = set(self.pipeline.total_retail["StockCode"].astype(str).str.upper().unique().tolist())
        self.available_countries = sorted(self.pipeline.total_retail["Country"].astype(str).unique().tolist())

    def answer_query(
        self,
        query: str,
        default_country: str = "United Kingdom",
        horizon: int = 12,
        output_json_path: str | Path | None = None,
    ) -> AgentResponse:
        parsed = parse_agent_query(
            query=query,
            available_product_ids=self.available_product_ids,
            available_countries=self.available_countries,
            default_country=default_country,
        )
        forecast = self.pipeline.forecast_product(parsed.product_id, country=parsed.country, horizon=horizon)
        forecast_json_path = self.pipeline.save_forecast(forecast, output_json_path)
        reply_text = format_agent_reply(forecast)
        return AgentResponse(
            parsed_query=parsed,
            forecast=forecast,
            reply_text=reply_text,
            forecast_json_path=forecast_json_path,
        )

    def save_agent_response(self, response: AgentResponse, output_path: str | Path | None = None) -> Path:
        if output_path is None:
            safe_country = response.forecast.country.replace(" ", "_")
            output_path = self.pipeline.paths.agent_dir / f"agent_response_{response.forecast.product_id}_{safe_country}.json"
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(response.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return out_path
