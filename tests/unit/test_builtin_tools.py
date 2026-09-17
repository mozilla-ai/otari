"""The built-in tool registry, and the hand-kept tool lists that must agree with it."""

import ast
from pathlib import Path
from typing import Any

import pytest

from gateway.api.routes._tools import Tool
from gateway.api.routes.tools import _managed_tools
from gateway.api.routes.usage import GATEWAY_TOOL_NAMES
from gateway.core.config import GatewayConfig
from gateway.services import code_execution_tool, web_search_tool
from gateway.services._tool_loop import ToolBackend
from gateway.services.builtin_tool import BuiltinTool
from gateway.services.builtin_tool_registry import BUILTIN_TOOLS
from gateway.services.sandbox_backend import SandboxBackend
from gateway.services.web_retrieval_backend import WebRetrievalBackend

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "src" / "gateway" / "services" / "builtin_tool_registry.py"
TOOL_ENV = (
    "OTARI_SANDBOX_URL",
    "OTARI_WEB_SEARCH_URL",
    "OTARI_WEB_SEARCH_PROVIDER",
    "OTARI_WEB_SEARCH_PROVIDER_API_KEY",
)


@pytest.fixture(autouse=True)
def _no_tool_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in TOOL_ENV:
        monkeypatch.delenv(name, raising=False)


def test_the_registry_is_a_literal_tuple() -> None:
    """Nothing scans and nothing computes: the line in the repo is the whole answer."""
    tree = ast.parse(REGISTRY_PATH.read_text(encoding="utf-8"))
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "BUILTIN_TOOLS"
    ]
    assert len(assignments) == 1, "BUILTIN_TOOLS is assigned once, with its annotation"
    value = assignments[0].value
    assert isinstance(value, ast.Tuple), "BUILTIN_TOOLS is a tuple literal, not a computed value"
    assert all(isinstance(element, ast.Attribute | ast.Name) for element in value.elts), (
        "each entry names the BuiltinTool its own module declares; none is constructed in the registry"
    )


def test_listed_tools_have_distinct_names() -> None:
    names = [tool.name for tool in BUILTIN_TOOLS]
    assert len(set(names)) == len(names), "two listed tools share a name"


def test_every_gateway_tool_type_has_exactly_one_listed_tool() -> None:
    """A tool a caller can declare and the registry does not list, or the reverse, fails here."""
    assert {f"otari_{tool.name}" for tool in BUILTIN_TOOLS} == {str(member) for member in Tool}


def test_usage_filters_on_exactly_the_listed_tools() -> None:
    assert set(GATEWAY_TOOL_NAMES) == {tool.name for tool in BUILTIN_TOOLS}


@pytest.mark.parametrize("tool", BUILTIN_TOOLS, ids=lambda tool: tool.name)
def test_a_definition_names_its_tool_and_is_new_on_every_call(tool: BuiltinTool) -> None:
    definition = tool.definition()
    assert definition["type"] == "function"
    assert definition["function"]["name"] == tool.name
    assert tool.definition() is not definition, "a caller mutating one definition must not change the next"


def _backend_for(tool: BuiltinTool) -> ToolBackend:
    """The backend that runs ``tool``, built without opening a connection."""
    if tool is web_search_tool.TOOL:
        return WebRetrievalBackend(base_url="http://search.invalid")
    if tool is code_execution_tool.TOOL:
        return SandboxBackend(sandbox_url="http://sandbox.invalid")
    raise AssertionError(f"no backend case for listed tool {tool.name!r}")


@pytest.mark.parametrize("tool", BUILTIN_TOOLS, ids=lambda tool: tool.name)
def test_the_backend_that_runs_a_tool_advertises_its_listed_definition(tool: BuiltinTool) -> None:
    backend = _backend_for(tool)
    assert tool.definition() in backend.openai_tools
    assert backend.owns_tool(tool.name)
    assert tool.name in dict(backend.purpose_hints())


@pytest.mark.parametrize(
    ("overrides", "env"),
    [
        ({}, {}),
        ({"web_search_url": "http://search.invalid"}, {}),
        ({"web_search_provider": "brave", "web_search_provider_api_key": "key"}, {}),
        ({"web_search_provider": "brave"}, {}),
        ({"sandbox_url": "http://sandbox.invalid"}, {}),
        ({}, {"OTARI_SANDBOX_URL": "http://sandbox.invalid"}),
        ({}, {"OTARI_WEB_SEARCH_URL": "http://search.invalid"}),
    ],
    ids=[
        "nothing",
        "search-url",
        "search-provider",
        "search-provider-without-key",
        "sandbox-url",
        "sandbox-url-from-env",
        "search-url-from-env",
    ],
)
def test_configured_agrees_with_what_the_tools_endpoint_reports(
    monkeypatch: pytest.MonkeyPatch, overrides: dict[str, Any], env: dict[str, str]
) -> None:
    config = GatewayConfig(**overrides)
    # Set after the config is built, so the answer depends on the environment read and not on the field.
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    reported = {tool.id: tool.available for tool in _managed_tools(config)}

    assert reported == {f"otari_{tool.name}": tool.configured(config) for tool in BUILTIN_TOOLS}
