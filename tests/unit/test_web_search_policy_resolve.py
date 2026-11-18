"""Resolving the web search policy a workspace's requests run under."""

from __future__ import annotations

import uuid
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.models.tools import (
    CodeExecutor,
    WebFetchCredential,
    WebSearchCredential,
    WebTool,
    WebToolCredentialSource,
    WorkspaceWebSearchConfig,
)
from gateway.services.tenancy import workspace_web_search_service
from gateway.services.tenancy.workspace_web_search_service import WorkspaceWebSearchPolicies
from gateway.services.tools import WorkspaceSearchKeys
from gateway.services.tools._web_search_keys import WorkspaceKeys


class _SearchKeys:
    """A resolver that answers one credential of each kind and records which kind each workspace asked for."""

    def __init__(
        self, credential: WebSearchCredential | None, fetch_credential: WebFetchCredential | None = None
    ) -> None:
        self.credential = credential
        self.fetch_credential = fetch_credential
        self.asked: list[tuple[str, uuid.UUID]] = []

    async def keys_for(self, workspace_id: uuid.UUID, *, search: bool, fetch: bool) -> WorkspaceKeys:
        self.asked.extend((kind, workspace_id) for kind, wanted in (("search", search), ("fetch", fetch)) if wanted)
        return WorkspaceKeys(search=self.credential if search else None, fetch=self.fetch_credential if fetch else None)


def _local(
    search_keys: _SearchKeys, monkeypatch: pytest.MonkeyPatch, *, on_without_policy: bool = True
) -> WorkspaceWebSearchPolicies:
    async def no_stored_policy(_session: Any, _workspace_id: uuid.UUID) -> None:
        return None

    monkeypatch.setattr(workspace_web_search_service, "resolve_workspace_web_search_config", no_stored_policy)
    return WorkspaceWebSearchPolicies(
        MagicMock(), search_keys=cast(WorkspaceSearchKeys, search_keys), on_without_policy=on_without_policy
    )


@pytest.mark.asyncio
async def test_a_local_search_carries_the_workspaces_own_key(monkeypatch: pytest.MonkeyPatch) -> None:
    workspace_id = uuid.uuid4()
    credential = WebSearchCredential(provider="brave", source=WebToolCredentialSource.ORG, api_key="brv-own")
    search_keys = _SearchKeys(credential)

    policy = await _local(search_keys, monkeypatch).resolve(workspace_id, [WebTool.SEARCH])

    assert policy is not None
    assert policy.credential == credential
    assert policy.fetch_credential is None
    assert search_keys.asked == [("search", workspace_id), ("fetch", workspace_id)]


@pytest.mark.asyncio
async def test_a_local_search_with_no_key_uses_the_deployments_search(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = await _local(_SearchKeys(None), monkeypatch).resolve(uuid.uuid4(), [WebTool.SEARCH])

    assert policy is None


@pytest.mark.asyncio
async def test_a_search_carries_the_workspaces_fetch_key_too_for_its_enrichment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetch_credential = WebFetchCredential(provider="tabstack", source=WebToolCredentialSource.ORG, api_key="tbs")

    policy = await _local(_SearchKeys(None, fetch_credential), monkeypatch).resolve(uuid.uuid4(), [WebTool.SEARCH])

    assert policy is not None
    assert policy.credential is None
    assert policy.fetch_credential == fetch_credential


@pytest.mark.asyncio
async def test_a_fetch_alone_reads_only_the_fetch_key(monkeypatch: pytest.MonkeyPatch) -> None:
    workspace_id = uuid.uuid4()
    fetch_credential = WebFetchCredential(provider="tabstack", source=WebToolCredentialSource.ORG, api_key="tbs")
    search_keys = _SearchKeys(
        WebSearchCredential(provider="brave", source=WebToolCredentialSource.ORG, api_key="brv-own"), fetch_credential
    )

    policy = await _local(search_keys, monkeypatch).resolve(workspace_id, [WebTool.FETCH])

    assert policy is not None
    assert policy.credential is None
    assert policy.fetch_credential == fetch_credential
    assert search_keys.asked == [("fetch", workspace_id)]


@pytest.mark.asyncio
@pytest.mark.parametrize("tools", [[WebTool.SEARCH], [WebTool.FETCH]], ids=["search", "fetch"])
async def test_no_row_where_the_deployment_makes_web_access_opt_in_is_a_disabled_policy(
    monkeypatch: pytest.MonkeyPatch, tools: list[WebTool]
) -> None:
    search_keys = _SearchKeys(
        WebSearchCredential(provider="brave", source=WebToolCredentialSource.ORG, api_key="brv-own")
    )

    policy = await _local(search_keys, monkeypatch, on_without_policy=False).resolve(uuid.uuid4(), tools)

    assert policy is not None
    assert policy.enabled is False
    assert policy.credential is None
    assert search_keys.asked == []


@pytest.mark.asyncio
async def test_the_pins_read_past_a_stored_domain_rule_that_is_invalid() -> None:
    """A domain rule governs only a tool Otari runs, so it cannot refuse a keyword the provider runs."""
    row = WorkspaceWebSearchConfig(
        workspace_id=uuid.uuid4(),
        enabled=True,
        allowed_domains=["https://example.com/path"],
        search_executor="provider",
        fetch_executor=None,
    )
    db = MagicMock()
    db.get = AsyncMock(return_value=row)
    policies = WorkspaceWebSearchPolicies(
        db, search_keys=cast(WorkspaceSearchKeys, _SearchKeys(None)), on_without_policy=True
    )

    assert await policies.executors(row.workspace_id) == (CodeExecutor.PROVIDER, None)
