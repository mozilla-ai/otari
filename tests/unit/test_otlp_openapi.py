"""Keep the published OTLP export response contract typed and complete."""

import json
from pathlib import Path
from typing import Any

_SPEC_PATH = Path(__file__).resolve().parents[2] / "docs/public/openapi.json"


def test_otlp_export_responses_describe_json_and_protobuf_shapes() -> None:
    spec: dict[str, Any] = json.loads(_SPEC_PATH.read_text())
    schemas = spec["components"]["schemas"]
    expected = {
        "/v1/traces": ("OTLPTraceServiceResponse", "OTLPTracePartialSuccess", "rejectedSpans"),
        "/v1/logs": ("OTLPLogsServiceResponse", "OTLPLogsPartialSuccess", "rejectedLogRecords"),
        "/v1/metrics": ("OTLPMetricsServiceResponse", "OTLPMetricsPartialSuccess", "rejectedDataPoints"),
    }

    for path, (response_name, partial_name, rejected_name) in expected.items():
        response = spec["paths"][path]["post"]["responses"]["200"]
        assert "partial success" in response["description"].lower()
        assert set(response["content"]) == {"application/json", "application/x-protobuf"}
        for media_type in response["content"]:
            assert response["content"][media_type]["schema"] == {
                "$ref": f"#/components/schemas/{response_name}"
            }

        response_schema = schemas[response_name]
        assert set(response_schema["properties"]) == {"partialSuccess"}
        assert response_schema["properties"]["partialSuccess"]["anyOf"][0] == {
            "$ref": f"#/components/schemas/{partial_name}"
        }

        partial_schema = schemas[partial_name]
        assert set(partial_schema["properties"]) == {rejected_name, "errorMessage"}
