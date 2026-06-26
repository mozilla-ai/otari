#!/usr/bin/env python3
"""Generate a Postman collection for Otari from the OpenAPI spec.

This is a stopgap client for managing a running Otari server (keys, users,
budgets, usage, pricing) and exercising the inference endpoints, until the
streamlined web UI lands. The collection is derived entirely from
``docs/public/openapi.json`` so it stays in sync with the API surface.

Two modes:
- Generate mode (default): writes the collection JSON.
- Check mode (--check): regenerates and compares against the existing file,
  exiting non-zero if they differ (for CI/CD, mirroring generate_openapi.py).

The OpenAPI spec is the source of truth here; regenerate it first with
``python scripts/generate_openapi.py`` if the API changed.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPEC = REPO_ROOT / "docs" / "public" / "openapi.json"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "public" / "otari.postman_collection.json"

# The OpenAPI spec does not declare a security scheme, so we inject it here.
# The gateway accepts the key as a Bearer token via the standard Authorization
# header (it also accepts the legacy Otari-Key header, but only when the value
# is itself prefixed with "Bearer "; see _extract_bearer_token in
# src/gateway/api/deps.py). Using standard Bearer auth keeps the collection
# aligned with the documented curl examples.
DEFAULT_BASE_URL = "http://localhost:8000"

# Methods that carry a JSON request body worth templating.
BODY_METHODS = {"post", "put", "patch"}
HTTP_METHODS = ("get", "post", "put", "patch", "delete")

# A few inference endpoints type their payloads as free-form objects in the
# spec (e.g. ``messages`` is just ``{type: object}``), so a derived example
# collapses to ``[{}]`` and is not runnable. For those, supply a curated,
# working example body keyed by (METHOD, path). Everything else stays derived
# from the schema. Models use the provider-prefixed form Otari expects.
EXAMPLE_BODY_OVERRIDES: dict[tuple[str, str], dict[str, Any]] = {
    ("POST", "/v1/chat/completions"): {
        "model": "openai:gpt-4o-mini",
        "messages": [{"role": "user", "content": "Hello from Otari!"}],
    },
    ("POST", "/v1/messages"): {
        "model": "anthropic:claude-haiku-4-5",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hello from Otari!"}],
    },
    ("POST", "/v1/responses"): {
        "model": "openai:gpt-4o-mini",
        "input": "Hello from Otari!",
    },
}


def load_spec(spec_path: Path) -> dict[str, Any]:
    """Load the OpenAPI spec JSON."""
    with open(spec_path, encoding="utf-8") as f:
        spec: dict[str, Any] = json.load(f)
        return spec


def resolve_ref(ref: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Resolve a local ``#/...`` JSON pointer within the spec."""
    node: Any = spec
    for part in ref.lstrip("#/").split("/"):
        node = node[part]
    return cast(dict[str, Any], node)


def example_for_schema(
    schema: dict[str, Any],
    spec: dict[str, Any],
    *,
    depth: int = 0,
    seen: frozenset[str] = frozenset(),
) -> Any:
    """Build a minimal example value from a JSON schema node.

    Prefers explicit ``example``/``default``; otherwise constructs a value from
    the schema's structure. Objects include only required properties (falling
    back to all properties when none are required) to keep bodies usable rather
    than exhaustive. Recursion is bounded to avoid blowing up on self-referential
    schemas.
    """
    if depth > 6:
        return None

    if "$ref" in schema:
        ref = schema["$ref"]
        if ref in seen:
            return None
        return example_for_schema(
            resolve_ref(ref, spec), spec, depth=depth, seen=seen | {ref}
        )

    if "example" in schema:
        return schema["example"]
    if "default" in schema:
        return schema["default"]

    for combiner in ("allOf", "anyOf", "oneOf"):
        if combiner in schema and schema[combiner]:
            if combiner == "allOf":
                merged: dict[str, Any] = {}
                for sub in schema["allOf"]:
                    value = example_for_schema(sub, spec, depth=depth + 1, seen=seen)
                    if isinstance(value, dict):
                        merged.update(value)
                return merged
            return example_for_schema(
                schema[combiner][0], spec, depth=depth + 1, seen=seen
            )

    if "enum" in schema and schema["enum"]:
        return schema["enum"][0]

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        schema_type = next((t for t in schema_type if t != "null"), None)

    if schema_type == "object" or "properties" in schema:
        props: dict[str, Any] = schema.get("properties", {})
        required = schema.get("required", [])
        keys = required if required else list(props)
        return {
            name: example_for_schema(props[name], spec, depth=depth + 1, seen=seen)
            for name in keys
            if name in props
        }

    if schema_type == "array":
        items = schema.get("items", {})
        return [example_for_schema(items, spec, depth=depth + 1, seen=seen)]

    if schema_type == "string":
        fmt = schema.get("format")
        if fmt == "date-time":
            return "2024-01-01T00:00:00Z"
        if fmt == "date":
            return "2024-01-01"
        if fmt == "uuid":
            return "00000000-0000-0000-0000-000000000000"
        return "string"
    if schema_type == "integer":
        return 0
    if schema_type == "number":
        return 0
    if schema_type == "boolean":
        return False

    return None


def request_body_example(operation: dict[str, Any], spec: dict[str, Any]) -> Any:
    """Return the JSON example for an operation's request body, if any."""
    content = operation.get("requestBody", {}).get("content", {})
    schema = content.get("application/json", {}).get("schema")
    if schema is None:
        return None
    return example_for_schema(schema, spec)


def build_url(path: str, operation: dict[str, Any]) -> dict[str, Any]:
    """Build a Postman URL object with path variables and query params."""
    # Postman represents path params as ``:name`` segments.
    raw_path = path
    segments: list[str] = []
    variables: list[dict[str, Any]] = []
    for seg in path.strip("/").split("/"):
        if seg.startswith("{") and seg.endswith("}"):
            name = seg[1:-1]
            segments.append(f":{name}")
            variables.append({"key": name, "value": "", "description": "path parameter"})
        else:
            segments.append(seg)
    raw_path = "/" + "/".join(segments) if segments else ""

    query: list[dict[str, Any]] = []
    for param in operation.get("parameters", []):
        if param.get("in") != "query":
            continue
        query.append(
            {
                "key": param["name"],
                "value": "",
                "description": param.get("description", ""),
                # Optional params start disabled so the request runs clean.
                "disabled": not param.get("required", False),
            }
        )

    raw = "{{baseUrl}}" + raw_path
    if query:
        raw += "?" + "&".join(f"{q['key']}=" for q in query)

    url: dict[str, Any] = {
        "raw": raw,
        "host": ["{{baseUrl}}"],
        "path": [s for s in segments],
    }
    if query:
        url["query"] = query
    if variables:
        url["variable"] = variables
    return url


def build_item(path: str, method: str, operation: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """Build a single Postman request item from an operation."""
    name = operation.get("summary") or operation.get("operationId") or f"{method.upper()} {path}"

    headers: list[dict[str, Any]] = []
    request: dict[str, Any] = {
        "method": method.upper(),
        "header": headers,
        "url": build_url(path, operation),
    }

    description = operation.get("description")
    if description:
        request["description"] = description

    if method in BODY_METHODS:
        body = EXAMPLE_BODY_OVERRIDES.get(
            (method.upper(), path)
        ) or request_body_example(operation, spec)
        if body is not None:
            headers.append({"key": "Content-Type", "value": "application/json"})
            request["body"] = {
                "mode": "raw",
                "raw": json.dumps(body, indent=2),
                "options": {"raw": {"language": "json"}},
            }

    return {"name": name, "request": request}


def build_collection(spec: dict[str, Any]) -> dict[str, Any]:
    """Build the full Postman collection from an OpenAPI spec."""
    info = spec.get("info", {})
    title = info.get("title", "Otari")

    # Group operations by their first tag, preserving spec tag order.
    declared_tags = [t["name"] for t in spec.get("tags", []) if "name" in t]
    folders: dict[str, list[dict[str, Any]]] = {}
    folder_order: list[str] = list(declared_tags)

    for path in spec.get("paths", {}):
        for method in HTTP_METHODS:
            operation = spec["paths"][path].get(method)
            if operation is None:
                continue
            tags = operation.get("tags") or ["default"]
            tag = tags[0]
            if tag not in folders:
                folders[tag] = []
                if tag not in folder_order:
                    folder_order.append(tag)
            folders[tag].append(build_item(path, method, operation, spec))

    items = [
        {"name": tag, "item": folders[tag]}
        for tag in folder_order
        if tag in folders
    ]

    description = (
        f"{title} management + inference collection, generated from the OpenAPI "
        "spec by scripts/generate_postman.py. Stopgap client for driving a running "
        "Otari server until the web UI lands.\n\n"
        f"Set the `baseUrl` and `otariKey` collection variables (or override them "
        "with a Postman environment). Auth is sent as an "
        "`Authorization: Bearer <otariKey>` header on every request via "
        "collection-level auth."
    )

    return {
        "info": {
            "name": f"{title} API",
            "description": description,
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
        },
        "auth": {
            "type": "bearer",
            "bearer": [
                {"key": "token", "value": "{{otariKey}}", "type": "string"},
            ],
        },
        "variable": [
            {"key": "baseUrl", "value": DEFAULT_BASE_URL, "type": "string"},
            {"key": "otariKey", "value": "", "type": "string"},
        ],
        "item": items,
    }


def serialize(collection: dict[str, Any]) -> str:
    """Serialize a collection deterministically for stable diffs."""
    return json.dumps(collection, indent=2, sort_keys=True) + "\n"


def main() -> int:
    """Generate or check the Postman collection."""
    parser = argparse.ArgumentParser(description="Generate Otari Postman collection")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if the generated collection matches the existing file (for CI/CD)",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=DEFAULT_SPEC,
        help=f"Path to the OpenAPI spec (default: {DEFAULT_SPEC})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output path for the collection (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    if not args.spec.exists():
        print(f"Error: {args.spec} does not exist", file=sys.stderr)
        print("Run 'python scripts/generate_openapi.py' first", file=sys.stderr)
        return 1

    print("Generating Postman collection...")
    collection = build_collection(load_spec(args.spec))
    generated = serialize(collection)

    if args.check:
        print(f"Checking if {args.output} is up to date...")
        if not args.output.exists():
            print(f"✗ {args.output} does not exist", file=sys.stderr)
            return 1
        existing = args.output.read_text(encoding="utf-8")
        if existing == generated:
            print("✓ Postman collection is up to date")
            return 0
        print("✗ Postman collection is out of date", file=sys.stderr)
        print("Run 'python scripts/generate_postman.py' to update it", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generated, encoding="utf-8")
    print(f"✓ Postman collection written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
