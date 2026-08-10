"""Content-free OTLP payload builders shared by telemetry integration tests."""

from typing import Any, cast

from google.protobuf.json_format import ParseDict
from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest


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
