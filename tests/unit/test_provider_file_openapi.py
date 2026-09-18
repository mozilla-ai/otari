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
    listing = paths[API_ROOT + "/files"]["get"]
    assert {"after", "before", "order", "purpose", "page", "ids[]"} <= {
        parameter["name"] for parameter in listing["parameters"]
    }
