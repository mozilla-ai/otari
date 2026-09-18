#!/usr/bin/env python3
"""Generate OpenAPI specification for Otari.

This script creates the FastAPI application and exports its OpenAPI specification
to a JSON file. It can be run in two modes:
- Generate mode (default): Writes the spec to docs/openapi.json
- Check mode (--check): Compares generated spec with existing file and exits with
  error if they differ (useful for CI/CD)
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastapi import APIRouter, FastAPI, Request

from gateway.api.routes.hybrid_files import router as hybrid_files_router
from gateway.api.routes.provider_files import create_provider_files_router
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.main import create_app
from gateway.services.provider_files.contracts import FileAccount, FileScope, OutputPrepare


def generate_openapi_spec() -> dict[str, object]:
    """Generate OpenAPI specification from FastAPI app.

    Returns:
        OpenAPI specification as a dictionary

    """
    with tempfile.TemporaryDirectory() as tmpdir:
        database_url = f"sqlite:///{Path(tmpdir) / 'openapi.db'}"
        config = GatewayConfig(
            database_url=database_url,
            bootstrap_api_key=False,
            # Placeholders, so the conditionally mounted web-search backend route
            # is in the published contract. It is mounted only where a provider
            # and a backend token are both configured, and the spec has to
            # describe every route this app can serve, not only the ones a bare
            # default config happens to reach. Nothing is called during
            # generation; these values are never used to authenticate anything.
            web_search_provider="tavily",
            web_search_provider_api_key="openapi-generation-placeholder",
            web_search_backend_token="openapi-generation-placeholder",
        )
        app = create_app(config)
        app.include_router(
            create_provider_files_router(
                authenticate=_schema_identity,
                authenticate_gateway=_schema_identity,
                authorize_attempt=_schema_attempt,
            ),
            prefix=API_ROOT,
        )
        spec = app.openapi()
        _merge_hybrid_files(spec)
        return cast(dict[str, object], spec)


async def _schema_identity(request: Request, uow: UnitOfWork) -> FileScope:
    raise RuntimeError("Schema-only authentication dependency")


async def _schema_attempt(scope: FileScope, request: OutputPrepare, uow: UnitOfWork) -> FileAccount:
    raise RuntimeError("Schema-only inference authorization dependency")


def _merge_hybrid_files(spec: dict[str, Any]) -> None:
    """Publish both runtime contracts without replacing standalone storage schemas."""
    hybrid = FastAPI()
    routes = APIRouter(prefix=API_ROOT)
    routes.include_router(hybrid_files_router)
    hybrid.include_router(routes)
    native = hybrid.openapi()
    spec["components"]["schemas"].update(native.get("components", {}).get("schemas", {}))
    for path, methods in native["paths"].items():
        for method, operation in methods.items():
            target = spec["paths"][path][method]
            target["description"] = target.get("description", "") + (
                "\n\nHybrid mode stores files at the authorized provider with uploader/workspace bindings. "
                "X-Otari-Files-Provider selects anthropic (default) or openai. "
                "The Anthropic envelope requires anthropic-version and rejects the legacy Files beta. "
                "OpenAI uses purpose, after/before pagination, and the OpenAI response envelope. "
                "Hosted mode does not serve public file bytes."
            )
            schema = operation["responses"].get("200", {}).get("content", {}).get("application/json", {}).get("schema")
            if schema:
                media = target["responses"]["200"]["content"]["application/json"]
                media["schema"] = {"anyOf": [media["schema"], schema]}
            target.setdefault("parameters", []).append(
                {
                    "name": "anthropic-version",
                    "in": "header",
                    "required": False,
                    "schema": {"type": "string"},
                    "description": "Required for the Anthropic hybrid Files envelope only.",
                }
            )
            target["parameters"].append(
                {
                    "name": "X-Otari-Files-Provider",
                    "in": "header",
                    "required": False,
                    "schema": {"type": "string", "enum": ["anthropic", "openai"], "default": "anthropic"},
                    "description": "Hybrid Files provider selector; credentials remain authority-selected.",
                }
            )
    listing = spec["paths"][f"{API_ROOT}/files"]["get"]
    listing["parameters"].extend(
        [
            {
                "name": "page",
                "in": "query",
                "required": False,
                "schema": {"type": "string"},
                "description": "Hybrid GA cursor.",
            },
            {
                "name": "ids[]",
                "in": "query",
                "required": False,
                "schema": {"type": "array", "items": {"type": "string"}, "maxItems": 100},
                "description": "Hybrid IDs filter; mutually exclusive with page and limit.",
            },
        ]
    )
    existing_names = {parameter["name"] for parameter in listing["parameters"]}
    for name in ("after", "before", "order", "purpose"):
        if name not in existing_names:
            listing["parameters"].append(
                {
                    "name": name,
                    "in": "query",
                    "required": False,
                    "schema": {"type": "string"},
                    "description": "OpenAI hybrid Files listing filter or cursor.",
                }
            )
    upload = spec["paths"][f"{API_ROOT}/files"]["post"]
    body_ref = upload["requestBody"]["content"]["multipart/form-data"]["schema"]["$ref"].split("/")[-1]
    spec["components"]["schemas"][body_ref]["properties"]["expires_in_seconds"] = {
        "type": "integer",
        "minimum": 3600,
        "maximum": 7776000,
        "description": "Anthropic hybrid retention, capped by the control-plane maximum.",
    }
    properties = spec["components"]["schemas"][body_ref]["properties"]
    properties["purpose"]["description"] = (
        "Required for OpenAI hybrid uploads; unsupported for Anthropic hybrid uploads."
    )
    properties["expires_after[anchor]"] = {
        "type": "string",
        "enum": ["created_at"],
        "description": "OpenAI hybrid expiry anchor.",
    }
    properties["expires_after[seconds]"] = {
        "type": "integer",
        "minimum": 1,
        "description": "OpenAI hybrid retention, capped by the control-plane maximum.",
    }


def write_spec(spec: dict[str, object], output_path: Path) -> None:
    """Write OpenAPI spec to file with pretty formatting.

    Args:
        spec: OpenAPI specification dictionary
        output_path: Path to output JSON file

    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, sort_keys=True)
        f.write("\n")


def check_spec(spec: dict[str, object], existing_path: Path) -> bool:
    """Check if generated spec matches existing file.

    Args:
        spec: Generated OpenAPI specification
        existing_path: Path to existing spec file

    Returns:
        True if specs match, False otherwise

    """
    if not existing_path.exists():
        print(f"Error: {existing_path} does not exist", file=sys.stderr)
        return False

    with open(existing_path, encoding="utf-8") as f:
        existing_spec = cast(dict[str, object], json.load(f))

    # Create copies to avoid modifying originals
    spec_copy = spec.copy()
    existing_copy = existing_spec.copy()

    # Remove version from comparison since it's dynamically generated from git
    spec_info = spec_copy.get("info")
    if isinstance(spec_info, dict):
        spec_info_copy = dict(spec_info)
        spec_info_copy.pop("version", None)
        spec_copy["info"] = spec_info_copy

    existing_info = existing_copy.get("info")
    if isinstance(existing_info, dict):
        existing_info_copy = dict(existing_info)
        existing_info_copy.pop("version", None)
        existing_copy["info"] = existing_info_copy

    generated_json = json.dumps(spec_copy, indent=2, sort_keys=True)
    existing_json = json.dumps(existing_copy, indent=2, sort_keys=True)
    if generated_json != existing_json:
        print("Generated spec does not match existing spec", file=sys.stderr)
        print("Generated spec:")
        print(generated_json)
        print("Existing spec:")
        print(existing_json)
        return False

    return generated_json == existing_json


def main() -> int:
    """Generate or check OpenAPI specification.

    Returns:
        Exit code (0 for success, 1 for failure)

    """
    parser = argparse.ArgumentParser(description="Generate OpenAPI specification")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if generated spec matches existing file (for CI/CD)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "docs" / "public" / "openapi.json",
        help="Output path for OpenAPI spec (default: docs/public/openapi.json)",
    )

    args = parser.parse_args()

    print("Generating OpenAPI specification...")
    spec = generate_openapi_spec()

    if args.check:
        print(f"Checking if {args.output} is up to date...")
        if check_spec(spec, args.output):
            print("✓ OpenAPI spec is up to date")
            return 0
        print("✗ OpenAPI spec is out of date", file=sys.stderr)
        print("Run 'python scripts/generate_openapi.py' to update it", file=sys.stderr)
        return 1

    write_spec(spec, args.output)
    print(f"✓ OpenAPI spec written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
