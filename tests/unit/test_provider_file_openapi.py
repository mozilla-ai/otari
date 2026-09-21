"""Published Files schemas include both native contracts and explicit provider selection."""

import json
import runpy
from pathlib import Path
from typing import Any, cast

from gateway.core.config import API_ROOT


def test_public_file_contracts_are_published() -> None:
    module = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/generate_openapi.py"))
    spec = cast(dict[str, Any], module["generate_openapi_spec"]())
    paths = spec["paths"]
    for method, suffix in (
        ("post", "/files"),
        ("get", "/files"),
        ("get", "/files/{file_id}"),
        ("delete", "/files/{file_id}"),
    ):
        operation = paths[API_ROOT + suffix][method]
        names = {parameter["name"] for parameter in operation["parameters"]}
        assert "X-Otari-Files-Provider" in names
        schema = json.dumps(operation["responses"]["200"]["content"]["application/json"]["schema"])
        assert "AnthropicFile" in schema and "OpenAIFile" in schema
    for suffix in ("/files", "/files/{file_id}", "/files/{file_id}/content"):
        for method, operation in paths[API_ROOT + suffix].items():
            description = operation["description"]
            assert ("pagination" in description) == (suffix == "/files" and method == "get")
    assert "uploads require purpose" in paths[API_ROOT + "/files"]["post"]["description"]
    download_description = paths[API_ROOT + "/files/{file_id}/content"]["get"]["description"]
    assert "raw file bytes, not a JSON envelope" in download_description
    assert "OpenAI response envelope" not in download_description
    listing = paths[API_ROOT + "/files"]["get"]
    assert {"after", "before", "order", "purpose", "page", "ids[]"} <= {
        parameter["name"] for parameter in listing["parameters"]
    }
    upload_schema = paths[API_ROOT + "/files"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
    body_schema = spec["components"]["schemas"][upload_schema["$ref"].split("/")[-1]]
    assert body_schema["properties"]["expires_after[seconds]"] == {
        "type": "integer",
        "minimum": 3600,
        "maximum": 2592000,
        "description": "OpenAI hybrid retention, capped by the control-plane maximum.",
    }
