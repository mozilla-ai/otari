"""Which web search key a workspace uses, and how that key reaches its searches."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from gateway.api.routes._tools import _build_web_retrieval_backend
from gateway.core.config import GatewayConfig
from gateway.models.tools import (
    OrgWebSearchKey,
    ResolvedWebSearchConfig,
    WebSearchCredential,
    WebTool,
    WorkspaceWebSearchKeyOverride,
)
from gateway.repositories.tools import resolve_web_search_key
from gateway.services import search_backend
from gateway.services.search_backend import SearchQuery
from gateway.services.tools import apply_web_access_policy
from gateway.services.web_search_providers import WebSearchProviderError

_START = datetime(2026, 1, 1, tzinfo=UTC)


def _key(name: str, *, provider: str = "tavily", default: bool = False, age: int = 0) -> OrgWebSearchKey:
    """A key created ``age`` days after the first, so a smaller age is older."""
    return OrgWebSearchKey(
        id=uuid.uuid4(),
        organization_id=uuid.uuid4(),
        provider=provider,
        name=name,
        encrypted_api_key="ciphertext",
        is_org_default=default,
        created_at=_START + timedelta(days=age),
    )


def _override(key: OrgWebSearchKey, *, pinned: bool = False, disabled: bool = False) -> WorkspaceWebSearchKeyOverride:
    return WorkspaceWebSearchKeyOverride(
        workspace_id=uuid.uuid4(),
        organization_id=key.organization_id,
        org_web_search_key_id=key.id,
        is_default=pinned,
        disabled=disabled,
    )


def _chosen(
    *candidates: tuple[OrgWebSearchKey, WorkspaceWebSearchKeyOverride | None], unusable: str = ""
) -> str | None:
    chosen = resolve_web_search_key(candidates, lambda key: key.name != unusable)
    return chosen.name if chosen is not None else None


def test_a_workspace_pin_wins_over_every_default() -> None:
    old_default = _key("default", default=True)
    pinned = _key("pinned", provider="brave", age=1)

    assert _chosen((old_default, None), (pinned, _override(pinned, pinned=True))) == "pinned"


def test_without_a_pin_a_default_wins_and_tavily_before_brave() -> None:
    oldest = _key("oldest")
    brave = _key("brave", provider="brave", default=True, age=1)
    tavily = _key("tavily", default=True, age=2)

    assert _chosen((oldest, None), (brave, None), (tavily, None)) == "tavily"


def test_without_a_pin_or_default_the_oldest_key() -> None:
    assert _chosen((_key("oldest"), None), (_key("newer", age=1), None)) == "oldest"


def test_a_key_the_workspace_turned_off_is_passed_over() -> None:
    default = _key("default", default=True)
    other = _key("other", age=1)

    assert _chosen((default, _override(default, disabled=True)), (other, None)) == "other"


def test_a_key_this_deployment_cannot_decrypt_is_passed_over() -> None:
    pinned = _key("pinned")
    other = _key("other", age=1)

    assert _chosen((pinned, _override(pinned, pinned=True)), (other, None), unusable="pinned") == "other"


def test_no_usable_key_means_the_deployments_search() -> None:
    only = _key("only")

    assert _chosen((only, _override(only, disabled=True))) is None
    assert _chosen() is None


def test_a_credential_never_prints_its_key() -> None:
    credential = WebSearchCredential(provider="tavily", api_key="tvly-secret")

    assert "tvly-secret" not in repr(credential)
    assert "tvly-secret" not in str(credential)


def test_the_workspace_key_wins_over_the_deployments_provider() -> None:
    config = GatewayConfig(web_search_provider="brave", web_search_provider_api_key="deployment-key")

    backend = _build_web_retrieval_backend(
        base_url="http://searxng:8080",
        search_tool_entry={},
        credential=WebSearchCredential(provider="tavily", api_key="workspace-key"),
        config=config,
    )

    assert (backend._provider, backend._provider_api_key) == ("tavily", "workspace-key")


def test_without_a_workspace_key_the_deployments_provider_is_used() -> None:
    config = GatewayConfig(web_search_provider="brave", web_search_provider_api_key="deployment-key")

    backend = _build_web_retrieval_backend(base_url=None, search_tool_entry={}, config=config)

    assert (backend._provider, backend._provider_api_key) == ("brave", "deployment-key")


@pytest.mark.parametrize(
    ("search_entry", "carried"),
    [({"type": "otari_web_search"}, True), (None, False)],
    ids=["search-declared", "fetch-only"],
)
def test_admission_carries_the_key_only_for_a_declared_search(
    search_entry: dict[str, str] | None, carried: bool
) -> None:
    credential = WebSearchCredential(provider="tavily", api_key="k")
    policy = ResolvedWebSearchConfig(
        enabled=True,
        max_results=None,
        purpose_hint=None,
        allowed_domains=None,
        blocked_domains=None,
        provider_options=None,
        authorized_tools=None,
        credential=credential,
    )
    tools = [WebTool.SEARCH] if search_entry is not None else [WebTool.FETCH]

    grant = apply_web_access_policy(
        policy, requested_tools=tools, search_tool_entry=search_entry, config=GatewayConfig()
    )

    assert grant.search_credential is (credential if carried else None)


def test_the_choice_decrypts_only_until_a_usable_key() -> None:
    pinned, default, oldest = _key("pinned", age=2), _key("default", default=True, age=1), _key("oldest")
    asked: list[str] = []

    def usable(key: OrgWebSearchKey) -> bool:
        asked.append(key.name)
        return key.name != "pinned"

    chosen = resolve_web_search_key([(oldest, None), (default, None), (pinned, _override(pinned, pinned=True))], usable)

    assert chosen is default
    assert asked == ["pinned", "default"]


@pytest.mark.asyncio
async def test_direct_search_on_a_workspace_key_reads_the_providers_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    async def fake_provider_search(**kwargs: Any) -> list[dict[str, Any]]:
        seen.update(kwargs)
        return [
            {"url": "https://kept.example/a", "title": "A", "content": "first"},
            {"url": "https://dropped.example/b", "title": "B", "content": "second"},
            {"url": "https://kept.example/c", "title": "C", "content": "third"},
        ]

    monkeypatch.setattr(search_backend, "provider_search", fake_provider_search)
    credential = WebSearchCredential(provider="tavily", api_key="workspace-key")

    outcome = await search_backend.run_keyed_search(
        credential, SearchQuery(query="q", max_results=2, domain_filter=("-dropped.example",))
    )

    assert (seen["provider"], seen["api_key"], seen["query"]) == ("tavily", "workspace-key", "q")
    assert seen["options"] == {"max_results": 2}
    assert [hit.url for hit in outcome.results] == ["https://kept.example/a", "https://kept.example/c"]
    assert outcome.cost_usd is None


@pytest.mark.asyncio
async def test_direct_search_on_a_workspace_key_reports_a_provider_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    async def failing_provider_search(**_kwargs: Any) -> list[dict[str, Any]]:
        raise WebSearchProviderError("tavily search returned HTTP 401")

    monkeypatch.setattr(search_backend, "provider_search", failing_provider_search)

    with pytest.raises(search_backend.SearchProviderError) as raised:
        await search_backend.run_keyed_search(
            WebSearchCredential(provider="tavily", api_key="workspace-key"), SearchQuery(query="q")
        )

    assert "workspace-key" not in str(raised.value)
