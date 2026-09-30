"""Unit tests for applying a workspace's web search policy to a request's declared web access."""

from __future__ import annotations

from typing import Any

import pytest

from gateway.core.config import GatewayConfig
from gateway.exceptions.tools_exceptions import (
    WebAccessDomainsExcludedError,
    WebAccessNotEnabledError,
    WebAccessToolNotAuthorizedError,
    WebSearchNotEnabledError,
    WorkspaceWebSearchDomainsExcludedError,
)
from gateway.models.tools import ResolvedWebSearchConfig
from gateway.services.tools import apply_web_access_policy


def _policy(**overrides: Any) -> ResolvedWebSearchConfig:
    values: dict[str, Any] = {
        "enabled": True,
        "max_results": None,
        "purpose_hint": None,
        "allowed_domains": None,
        "blocked_domains": None,
        "provider_options": None,
        "authorized_tools": None,
    }
    values.update(overrides)
    return ResolvedWebSearchConfig(**values)


def _config() -> GatewayConfig:
    return GatewayConfig(web_search_max_results=8)


def test_no_policy_narrows_nothing() -> None:
    entry = {"type": "otari_web_search", "max_results": 3}

    grant = apply_web_access_policy(None, requested_tools=["web_search"], search_tool_entry=entry, config=_config())

    assert grant.search_tool_entry == entry
    assert grant.fetch_policy.allowed == ()
    assert grant.fetch_policy.blocked == ()


@pytest.mark.parametrize(
    ("requested_tools", "error"),
    [
        (["web_search"], WebSearchNotEnabledError),
        (["web_fetch"], WebAccessNotEnabledError),
        (["web_search", "web_fetch"], WebAccessNotEnabledError),
    ],
)
def test_a_disabled_policy_refuses_with_the_error_for_what_was_declared(
    requested_tools: list[str], error: type[Exception]
) -> None:
    entry = {"type": "otari_web_search"} if "web_search" in requested_tools else None

    with pytest.raises(error):
        apply_web_access_policy(
            _policy(enabled=False), requested_tools=requested_tools, search_tool_entry=entry, config=_config()
        )


def test_a_tool_the_policy_does_not_authorize_is_refused() -> None:
    with pytest.raises(WebAccessToolNotAuthorizedError):
        apply_web_access_policy(
            _policy(authorized_tools=frozenset({"web_search"})),
            requested_tools=["web_fetch"],
            search_tool_entry=None,
            config=_config(),
        )


def test_search_domains_narrow_the_workspace_fetch_domains() -> None:
    grant = apply_web_access_policy(
        _policy(allowed_domains=("example.com",), blocked_domains=("blocked.example.com",)),
        requested_tools=["web_search", "web_fetch"],
        search_tool_entry={"type": "otari_web_search", "allowed_domains": ["docs.example.com"]},
        config=_config(),
    )

    assert [rule.value for rule in grant.fetch_policy.allowed] == ["docs.example.com"]
    assert [rule.value for rule in grant.fetch_policy.blocked] == ["blocked.example.com"]


def test_fetch_domains_that_share_nothing_with_the_workspace_are_refused() -> None:
    with pytest.raises(WebAccessDomainsExcludedError):
        apply_web_access_policy(
            _policy(allowed_domains=("example.com",)),
            requested_tools=["web_search", "web_fetch"],
            search_tool_entry={"type": "otari_web_search", "allowed_domains": ["other.test"]},
            config=_config(),
        )


def test_search_domains_that_share_nothing_with_the_workspace_are_refused() -> None:
    with pytest.raises(WorkspaceWebSearchDomainsExcludedError):
        apply_web_access_policy(
            _policy(allowed_domains=("example.com",)),
            requested_tools=["web_search"],
            search_tool_entry={"type": "otari_web_search", "allowed_domains": ["other.test"]},
            config=_config(),
        )


def test_the_workspace_ceiling_is_floored_against_the_deployment_default() -> None:
    entry = {"type": "otari_web_search"}

    grant = apply_web_access_policy(
        _policy(max_results=12), requested_tools=["web_search"], search_tool_entry=entry, config=_config()
    )

    assert grant.search_tool_entry is not None
    assert grant.search_tool_entry["max_results"] == 8
    assert "max_results" not in entry
