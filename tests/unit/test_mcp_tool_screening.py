"""Per-tool admission of a live MCP descriptor (R-SCHEMA-1 to R-SCHEMA-3).

An ``inputSchema`` is untrusted JSON Schema *data*. Otari does not validate it
against a dialect, rewrite it, or drop keywords it does not recognize; it
bounds what it will carry and refuses anything that would make Otari fetch a
schema over the network. A tool that fails screening is omitted from the
catalog with a labeled warning, and its siblings stay available.
"""

from __future__ import annotations

import pytest
from mcp.types import Tool as MCPTool

from gateway.services.mcp_stateless import (
    SCHEMA_MAX_BYTES,
    SCHEMA_MAX_DEPTH,
    TOOL_ANNOTATIONS_MAX_BYTES,
    TOOL_DESCRIPTION_MAX_BYTES,
    TOOL_NAME_MAX_LENGTH,
    screen_tool,
)


def _tool(**overrides: object) -> MCPTool:
    fields: dict[str, object] = {
        "name": "create_issue",
        "description": "Create an issue",
        "inputSchema": {"type": "object", "properties": {"title": {"type": "string"}}},
    }
    fields.update(overrides)
    return MCPTool(**fields)  # type: ignore[arg-type]


def test_an_ordinary_tool_is_admitted() -> None:
    assert screen_tool(_tool()) is None


def test_a_tool_name_at_the_execution_limit_is_admitted() -> None:
    assert screen_tool(_tool(name="x" * TOOL_NAME_MAX_LENGTH)) is None


@pytest.mark.parametrize("name", ["", "x" * (TOOL_NAME_MAX_LENGTH + 1)])
def test_a_tool_name_execution_cannot_accept_is_omitted(name: str) -> None:
    assert screen_tool(_tool(name=name)) == "mcp_tool_name_unsupported"


def test_an_unknown_dialect_or_keyword_is_admitted_unchanged() -> None:
    """Unknown keywords are preserved, not rejected (R-SCHEMA-1, R-SCHEMA-2)."""
    schema = {
        "$schema": "https://example.com/draft/2044-01/schema",
        "type": "object",
        "properties": {"title": {"type": "string", "x-vendor-widget": "textarea"}},
        "x-vendor-policy": {"retries": 3},
    }
    tool = _tool(inputSchema=schema)

    assert screen_tool(tool) is None
    assert tool.inputSchema == schema


def test_a_local_reference_is_admitted() -> None:
    schema = {
        "type": "object",
        "properties": {"labels": {"$ref": "#/$defs/label"}},
        "$defs": {"label": {"type": "string"}},
    }

    assert screen_tool(_tool(inputSchema=schema)) is None


@pytest.mark.parametrize(
    "ref",
    [
        "https://example.com/schema.json",
        "//example.com/schema.json",
        "schema.json#/$defs/label",
        "file:///etc/passwd",
    ],
)
def test_an_external_reference_is_refused_without_network_access(ref: str) -> None:
    schema = {"type": "object", "properties": {"labels": {"$ref": ref}}}

    assert screen_tool(_tool(inputSchema=schema)) == "mcp_tool_schema_unsupported"


def test_a_non_object_schema_is_refused() -> None:
    tool = _tool()
    # MCP's own model types ``inputSchema`` as an object, so a server sending
    # something else can only reach this through the loosely typed wire form.
    object.__setattr__(tool, "inputSchema", ["not", "an", "object"])

    assert screen_tool(tool) == "mcp_tool_schema_unsupported"


def test_an_oversized_schema_is_refused() -> None:
    schema = {"type": "object", "properties": {"title": {"description": "x" * (SCHEMA_MAX_BYTES + 1)}}}

    assert screen_tool(_tool(inputSchema=schema)) == "mcp_tool_schema_unsupported"


def test_a_too_deeply_nested_schema_is_refused() -> None:
    nested: dict[str, object] = {"type": "object"}
    for _ in range(SCHEMA_MAX_DEPTH + 2):
        nested = {"properties": nested}

    assert screen_tool(_tool(inputSchema=nested)) == "mcp_tool_schema_unsupported"


def test_an_oversized_description_is_refused() -> None:
    assert screen_tool(_tool(description="x" * (TOOL_DESCRIPTION_MAX_BYTES + 1))) == "mcp_tool_description_too_large"


def test_an_oversized_annotations_object_is_refused() -> None:
    tool = _tool()
    object.__setattr__(tool, "annotations", {"note": "x" * (TOOL_ANNOTATIONS_MAX_BYTES + 1)})

    assert screen_tool(tool) == "mcp_tool_annotations_too_large"


def test_a_missing_description_is_admitted() -> None:
    assert screen_tool(_tool(description=None)) is None
