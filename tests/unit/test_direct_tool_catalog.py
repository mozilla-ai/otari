"""Default discovery follows direct selection and validates credentials without making a call."""

from dataclasses import replace
from typing import Any

import pytest

from gateway.core.config import GatewayConfig
from gateway.core.settings.tools import ToolKind
from gateway.models.api_keys import APIKey
from gateway.models.tools import (
    ResolvedWebSearchConfig,
    WebFetchCredential,
    WebSearchCredential,
    WebTool,
    WebToolCredentialSource,
)
from gateway.services.tools import DirectToolCatalog

_POLICY = ResolvedWebSearchConfig(
    enabled=True,
    max_results=None,
    purpose_hint=None,
    allowed_domains=None,
    blocked_domains=None,
    provider_options=None,
    authorized_tools=None,
)


class _Permissions:
    def __init__(self, allowed: set[str] | None = None) -> None:
        self.allowed = allowed
        self.asked: list[tuple[str, str]] = []

    async def tool_refusal(
        self, config: GatewayConfig, api_key: APIKey | None, *, kind: ToolKind, provider: str, tool: str
    ) -> str | None:
        self.asked.append((kind, tool))
        return None if self.allowed is None or tool in self.allowed else "not permitted"


@pytest.mark.parametrize(
    "settings,expected",
    [
        ({}, set()),
        ({"web_search_url": "https://search.example.com"}, set()),
        ({"search_tools": {"news": {"provider": "fake"}}}, {WebTool.SEARCH}),
        ({"search_tools": {"a": {"provider": "fake"}, "b": {"provider": "fake"}}}, set()),
        (
            {"search_tools": {"news": {"provider": "fake"}}, "web_search_default_tool": "none"},
            {WebTool.SEARCH},
        ),
        ({"web_fetch_enabled": True}, {WebTool.FETCH}),
        ({"search_tools": {"exa": {"provider": "exa"}}}, set()),
    ],
)
@pytest.mark.asyncio
async def test_discovery_resolves_the_default_direct_configuration(
    settings: dict[str, Any], expected: set[WebTool], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OTARI_WEB_SEARCH_URL", raising=False)
    catalog = DirectToolCatalog(config=GatewayConfig(**settings), permissions=_Permissions())
    assert await catalog.available_defaults(None, policy=None, workspace_allows=True) == expected


@pytest.mark.asyncio
async def test_a_workspace_veto_prevents_even_the_permission_lookup() -> None:
    permissions = _Permissions()
    catalog = DirectToolCatalog(
        config=GatewayConfig(search_tools={"news": {"provider": "fake"}}), permissions=permissions
    )
    assert await catalog.available_defaults(None, policy=_POLICY, workspace_allows=False) == frozenset()
    assert permissions.asked == []


@pytest.mark.parametrize(
    "provider,key,expected", [("tavily", "k", True), ("exa", "k", True), ("exa", None, False), ("retired", "k", False)]
)
@pytest.mark.asyncio
async def test_the_organization_fallback_is_usable_and_has_a_distinct_grant(
    provider: str, key: str | None, expected: bool
) -> None:
    policy = replace(
        _POLICY, credential=WebSearchCredential(provider=provider, api_key=key, source=WebToolCredentialSource.ORG)
    )
    permissions = _Permissions({"otari:web_search"})
    catalog = DirectToolCatalog(config=GatewayConfig(), permissions=permissions)
    available = await catalog.available_defaults(None, policy=policy, workspace_allows=True)
    assert (WebTool.SEARCH in available) is expected
    assert permissions.asked == ([("search", "otari:web_search")] if expected else [])


@pytest.mark.parametrize("kind", ["search", "fetch"])
@pytest.mark.asyncio
async def test_an_unusable_organization_key_cannot_advertise_a_working_default(kind: str) -> None:
    config = GatewayConfig(search_tools={"news": {"provider": "fake"}}, web_fetch_enabled=True)
    if kind == "search":
        policy = replace(
            _POLICY, credential=WebSearchCredential(provider="exa", source=WebToolCredentialSource.ORG, api_key=None)
        )
    else:
        policy = replace(
            _POLICY,
            fetch_credential=WebFetchCredential(provider="tabstack", source=WebToolCredentialSource.ORG, api_key=None),
        )
    catalog = DirectToolCatalog(config=config, permissions=_Permissions())
    available = await catalog.available_defaults(None, policy=policy, workspace_allows=True)
    assert available == ({WebTool.FETCH} if kind == "search" else {WebTool.SEARCH})


@pytest.mark.asyncio
async def test_an_ambiguous_deployment_default_can_still_use_the_organization_key() -> None:
    config = GatewayConfig(search_tools={"a": {"provider": "fake"}, "b": {"provider": "fake"}})
    policy = replace(
        _POLICY, credential=WebSearchCredential(provider="tavily", source=WebToolCredentialSource.ORG, api_key="k")
    )
    catalog = DirectToolCatalog(config=config, permissions=_Permissions())
    assert await catalog.available_defaults(None, policy=policy, workspace_allows=True) == {WebTool.SEARCH}
