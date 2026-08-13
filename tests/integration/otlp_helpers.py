"""Content-free OTLP payload builders shared by telemetry integration tests."""

from typing import Any, cast

from google.protobuf.json_format import ParseDict
from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest
from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import ExportMetricsServiceRequest


def attribute(key: str, value: Any) -> dict[str, Any]:
    payload: dict[str, bool | float | str]
    if isinstance(value, bool):
        payload = {"boolValue": value}
    elif isinstance(value, int):
        payload = {"intValue": str(value)}
    elif isinstance(value, float):
        payload = {"doubleValue": value}
    else:
        payload = {"stringValue": str(value)}
    return {"key": key, "value": payload}


def logs_export(*records: dict[str, Any]) -> dict[str, Any]:
    return {"resourceLogs": [{"scopeLogs": [{"logRecords": list(records)}]}]}


def logs_export_protobuf(*records: dict[str, Any]) -> bytes:
    """Build a protobuf OTLP logs export from the JSON-compatible test shape."""
    request = ExportLogsServiceRequest()
    ParseDict(logs_export(*records), request)
    return cast(bytes, request.SerializeToString())


def log_record(timestamp: int, **attributes: Any) -> dict[str, Any]:
    return {
        "timeUnixNano": str(timestamp),
        "attributes": [attribute(key, value) for key, value in attributes.items()],
    }


def metrics_export(*metrics: dict[str, Any]) -> dict[str, Any]:
    return {"resourceMetrics": [{"scopeMetrics": [{"metrics": list(metrics)}]}]}


def metrics_export_protobuf(*metrics: dict[str, Any]) -> bytes:
    """Build a protobuf OTLP metrics export from the JSON-compatible test shape."""
    request = ExportMetricsServiceRequest()
    ParseDict(metrics_export(*metrics), request)
    return cast(bytes, request.SerializeToString())


def number_point(timestamp: int, value: float, *, start: int | None = None, **attributes: Any) -> dict[str, Any]:
    point: dict[str, Any] = {
        "timeUnixNano": str(timestamp),
        "asDouble": float(value),
        "attributes": [attribute(key, attribute_value) for key, attribute_value in attributes.items()],
    }
    if start is not None:
        point["startTimeUnixNano"] = str(start)
    return point


def sum_metric(name: str, *points: dict[str, Any], temporality: str = "cumulative") -> dict[str, Any]:
    return {
        "name": name,
        "sum": {
            "dataPoints": list(points),
            "aggregationTemporality": f"AGGREGATION_TEMPORALITY_{temporality.upper()}",
            "isMonotonic": True,
        },
    }


def gauge_metric(name: str, *points: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "gauge": {"dataPoints": list(points)}}
