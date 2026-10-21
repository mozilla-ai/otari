"""Resolving a workspace's web search policy from this deployment's own rows."""

from __future__ import annotations

import uuid
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from gateway.adapters import web_search_policy_adapter
from gateway.adapters.web_search_policy_adapter import LocalWebSearchPolicy
from gateway.exceptions.tools_exceptions import WebSearchPolicyResolutionFailedError, WebSearchPolicyResolutionFailure
from gateway.models.tools import WebFetchCredential, WebSearchCredential, WebTool, WebToolCredentialSource
from gateway.ports.web_search_policy_port import WebSearchPolicyScope
from gateway.services.tools import WorkspaceSearchKeys
from gateway.services.tools._web_search_keys import WorkspaceKeys


@pytest.mark.asyncio
async def test_a_local_scope_without_a_workspace_is_a_resolution_failure() -> None:
    with pytest.raises(WebSearchPolicyResolutionFailedError) as ei:
        await LocalWebSearchPolicy(MagicMock(), search_keys=MagicMock(), on_without_policy=True).resolve(
            WebSearchPolicyScope(workspace_id=None), [WebTool.SEARCH]
        )

    assert ei.value.reason is WebSearchPolicyResolutionFailure.NO_WORKSPACE


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
) -> LocalWebSearchPolicy:
    async def no_stored_policy(_session: Any, _workspace_id: uuid.UUID) -> None:
        return None

    monkeypatch.setattr(web_search_policy_adapter, "resolve_workspace_web_search_config", no_stored_policy)
    return LocalWebSearchPolicy(
        MagicMock(), search_keys=cast(WorkspaceSearchKeys, search_keys), on_without_policy=on_without_policy
    )


@pytest.mark.asyncio
async def test_a_local_search_carries_the_workspaces_own_key(monkeypatch: pytest.MonkeyPatch) -> None:
    workspace_id = uuid.uuid4()
    credential = WebSearchCredential(provider="brave", source=WebToolCredentialSource.ORG, api_key="brv-own")
    search_keys = _SearchKeys(credential)

    policy = await _local(search_keys, monkeypatch).resolve(
        WebSearchPolicyScope(workspace_id=workspace_id), [WebTool.SEARCH]
    )

    assert policy is not None
    assert policy.credential == credential
    assert policy.fetch_credential is None
    assert search_keys.asked == [("search", workspace_id), ("fetch", workspace_id)]


@pytest.mark.asyncio
async def test_a_local_search_with_no_key_uses_the_deployments_search(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = await _local(_SearchKeys(None), monkeypatch).resolve(
        WebSearchPolicyScope(workspace_id=uuid.uuid4()), [WebTool.SEARCH]
    )

    assert policy is None


@pytest.mark.asyncio
async def test_a_search_carries_the_workspaces_fetch_key_too_for_its_enrichment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetch_credential = WebFetchCredential(provider="tabstack", source=WebToolCredentialSource.ORG, api_key="tbs")

    policy = await _local(_SearchKeys(None, fetch_credential), monkeypatch).resolve(
        WebSearchPolicyScope(workspace_id=uuid.uuid4()), [WebTool.SEARCH]
    )

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

    policy = await _local(search_keys, monkeypatch).resolve(
        WebSearchPolicyScope(workspace_id=workspace_id), [WebTool.FETCH]
    )

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

    policy = await _local(search_keys, monkeypatch, on_without_policy=False).resolve(
        WebSearchPolicyScope(workspace_id=uuid.uuid4()), tools
    )

    assert policy is not None
    assert policy.enabled is False
    assert policy.credential is None
    assert search_keys.asked == []
