"""An organization's own web search and fetch keys: managing them, and the requests using them.

The cases on the older backend go through ``/api/v1/messages`` with it patched out, so
what is asserted is which provider and key it was built with. The cases on the
libraries go through ``/api/v1/chat/completions`` with a scripted model, and answer the
provider's calls through the app's web provider client.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    Choice,
    CompletionUsage,
    Function,
)
from any_llm.types.messages import MessageResponse, MessageUsage, TextBlock
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from gateway.api.deps import get_optional_web_provider_client, get_web_provider_client
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.organizations_exceptions import NotAuthorizedError
from gateway.exceptions.tools_exceptions import WebSearchKeyDefaultConflictError, WebSearchKeyNotFoundError
from gateway.models.tenancy import User, Workspace
from gateway.models.tools import OrgWebSearchKey
from gateway.models.usage import UsageLog
from gateway.repositories.tenancy import (
    OrganizationMemberRepository,
    OrganizationRepository,
    UserRepository,
    WorkspaceMemberRepository,
    WorkspaceRepository,
)
from gateway.repositories.tools import (
    OrgWebSearchKeyRepository,
    WebSearchKeyConflict,
    WorkspaceWebSearchKeyOverrideRepository,
)
from gateway.schemas.tools import OrgWebSearchKeyCreateRequest, WorkspaceWebSearchKeyOverrideRequest
from gateway.services.secret_box import generate_secret_key
from gateway.services.tenancy import OrganizationService
from gateway.services.tenancy.authorization import WorkspaceAccess
from gateway.services.tools import WebSearchKeyService, _web_search_keys

from .conftest import MODEL_NAME

_SEARCH_URL = "http://127.0.0.1:9998/search"
_KEYS = f"{API_ROOT}/organizations/me/web-search-keys"
_REQUEST = {
    "model": "anthropic:claude-3-5-sonnet-20241022",
    "messages": [{"role": "user", "content": "what happened today"}],
    "max_tokens": 100,
    "tools": [{"type": "otari_web_search"}],
}


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    monkeypatch.setenv("OTARI_WEB_SEARCH_URL", _SEARCH_URL)


def _workspace_id(client: TestClient, headers: dict[str, str]) -> str:
    listed = client.get(f"{API_ROOT}/workspaces", headers=headers)
    assert listed.status_code == 200, listed.text
    workspace_id: str = listed.json()["data"][0]["id"]
    return workspace_id


def _add_key(
    client: TestClient,
    headers: dict[str, str],
    *,
    provider: str = "tavily",
    name: str = "ours",
    api_key: str,
    **fields: Any,
) -> dict[str, Any]:
    created = client.post(
        _KEYS, json={"provider": provider, "name": name, "api_key": api_key, **fields}, headers=headers
    )
    assert created.status_code == 201, created.text
    body: dict[str, Any] = created.json()
    return body


def _searched_with(client: TestClient, headers: dict[str, str]) -> dict[str, Any]:
    """The keyword arguments the request's search backend was built with."""
    seen: dict[str, Any] = {}

    async def fake_loop(**kwargs: Any) -> MessageResponse:
        return MessageResponse(
            id="msg_test",
            type="message",
            role="assistant",
            model="claude-3-5-sonnet-20241022",
            content=[TextBlock(type="text", text="ok", citations=None)],
            stop_reason="end_turn",
            stop_sequence=None,
            usage=MessageUsage(input_tokens=5, output_tokens=2),
        )

    def fake_backend(**kwargs: Any) -> Any:
        seen.update(kwargs)
        backend = AsyncMock()
        backend.purpose_hints = lambda: []
        return AsyncMock(__aenter__=AsyncMock(return_value=backend), __aexit__=AsyncMock(return_value=None))

    with (
        patch("gateway.api.routes.messages.anthropic_tool_loop", new=fake_loop),
        patch("gateway.api.routes._tools.WebRetrievalBackend", new=fake_backend),
    ):
        response = client.post(f"{API_ROOT}/messages", json=_REQUEST, headers=headers)
    assert response.status_code == 200, response.text
    return seen


# --- managing keys -----------------------------------------------------------------------


def test_a_key_is_stored_without_its_secret_and_listed(client: TestClient, master_key_header: dict[str, str]) -> None:
    created = _add_key(client, master_key_header, api_key="tvly-secret-1234")

    assert created["provider"] == "tavily"
    assert created["last4"] == "1234"
    assert created["usable"] is True
    assert "tvly-secret-1234" not in str(created)
    listed = client.get(_KEYS, headers=master_key_header).json()
    assert [key["id"] for key in listed["data"]] == [created["id"]]
    assert "tvly-secret-1234" not in str(listed)


def test_a_provider_the_search_tool_cannot_call_is_refused(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    response = client.post(_KEYS, json={"provider": "google", "name": "g", "api_key": "k"}, headers=master_key_header)

    assert response.status_code == 400
    assert "tavily" in response.json()["detail"]


def test_a_second_key_of_the_same_provider_and_name_is_refused(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _add_key(client, master_key_header, api_key="tvly-1")

    response = client.post(
        _KEYS, json={"provider": "tavily", "name": "ours", "api_key": "tvly-2"}, headers=master_key_header
    )

    assert response.status_code == 409


def test_renaming_a_key_onto_another_keys_name_is_refused(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _add_key(client, master_key_header, name="first", api_key="tvly-1")
    second = _add_key(client, master_key_header, name="second", api_key="tvly-2")

    response = client.patch(f"{_KEYS}/{second['id']}", json={"name": "first"}, headers=master_key_header)

    assert response.status_code == 409
    assert "first" in response.json()["detail"]


def test_a_pasted_key_is_trimmed_and_one_that_cannot_be_sent_is_refused(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    trimmed = _add_key(client, master_key_header, api_key="  tvly-secret-1234\n")
    assert trimmed["last4"] == "1234"

    for api_key in ("tvly-secret 1234", "tvly-secret\n1234", "tvly-\x00-1234", "   "):
        refused = client.post(
            _KEYS, json={"provider": "tavily", "name": "bad", "api_key": api_key}, headers=master_key_header
        )
        assert refused.status_code in (400, 422), refused.text
        assert "secret" not in refused.text
    renamed = client.patch(f"{_KEYS}/{trimmed['id']}", json={"api_key": "tvly-new\r\n9999"}, headers=master_key_header)
    assert renamed.status_code == 400, renamed.text


def test_a_restored_key_is_not_a_default_again(client: TestClient, master_key_header: dict[str, str]) -> None:
    first = _add_key(client, master_key_header, name="first", api_key="tvly-1")
    second = _add_key(client, master_key_header, name="second", api_key="tvly-2")
    client.post(f"{_KEYS}/{first['id']}/default", headers=master_key_header)
    client.post(f"{_KEYS}/{first['id']}/archive", headers=master_key_header)
    client.post(f"{_KEYS}/{second['id']}/default", headers=master_key_header)

    restored = client.post(f"{_KEYS}/{first['id']}/restore", headers=master_key_header)

    assert restored.status_code == 200, restored.text
    assert restored.json()["is_org_default"] is False


def test_restoring_a_live_key_changes_nothing(client: TestClient, master_key_header: dict[str, str]) -> None:
    key = _add_key(client, master_key_header, api_key="tvly-1")
    client.post(f"{_KEYS}/{key['id']}/default", headers=master_key_header)

    restored = client.post(f"{_KEYS}/{key['id']}/restore", headers=master_key_header)

    assert restored.status_code == 200, restored.text
    assert restored.json()["is_org_default"] is True


def test_one_default_per_provider_and_a_live_key_cannot_be_deleted(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    first = _add_key(client, master_key_header, name="first", api_key="tvly-1")
    second = _add_key(client, master_key_header, name="second", api_key="tvly-2")

    client.post(f"{_KEYS}/{first['id']}/default", headers=master_key_header)
    made = client.post(f"{_KEYS}/{second['id']}/default", headers=master_key_header)

    assert made.json()["is_org_default"] is True
    defaults = [
        key["name"] for key in client.get(_KEYS, headers=master_key_header).json()["data"] if key["is_org_default"]
    ]
    assert defaults == ["second"]
    assert client.delete(f"{_KEYS}/{second['id']}", headers=master_key_header).status_code == 400
    assert client.post(f"{_KEYS}/{second['id']}/archive", headers=master_key_header).json()["is_org_default"] is False
    assert client.delete(f"{_KEYS}/{second['id']}", headers=master_key_header).status_code == 204
    assert [key["name"] for key in client.get(_KEYS, headers=master_key_header).json()["data"]] == ["first"]


def test_a_workspace_pins_or_turns_off_a_key(client: TestClient, master_key_header: dict[str, str]) -> None:
    workspace_id = _workspace_id(client, master_key_header)
    tavily = _add_key(client, master_key_header, name="t", api_key="tvly-1")
    brave = _add_key(client, master_key_header, provider="brave", name="b", api_key="brave-1")
    view = f"{API_ROOT}/workspaces/{workspace_id}/web-search-keys"

    def effective() -> list[str]:
        return [
            key["name"] for key in client.get(view, headers=master_key_header).json()["data"] if key["is_effective"]
        ]

    # No default and no pin: the oldest key.
    assert effective() == ["t"]
    client.patch(f"{view}/{brave['id']}", json={"is_default": True}, headers=master_key_header)
    assert effective() == ["b"]
    client.patch(f"{view}/{brave['id']}", json={"disabled": True}, headers=master_key_header)
    assert effective() == ["t"]
    client.patch(f"{view}/{tavily['id']}", json={"disabled": True}, headers=master_key_header)
    assert effective() == []
    reset = client.delete(f"{view}/{tavily['id']}", headers=master_key_header)
    assert reset.status_code == 200
    assert effective() == ["t"]


def test_a_workspaces_keys_are_paged_and_the_effective_key_is_chosen_across_pages(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    workspace_id = _workspace_id(client, master_key_header)
    _add_key(client, master_key_header, name="oldest", api_key="tvly-1")
    newest = _add_key(client, master_key_header, name="newest", api_key="tvly-2")
    view = f"{API_ROOT}/workspaces/{workspace_id}/web-search-keys"
    client.patch(f"{view}/{newest['id']}", json={"is_default": True}, headers=master_key_header)

    first = client.get(view, params={"limit": 1}, headers=master_key_header).json()
    second = client.get(view, params={"skip": 1, "limit": 1}, headers=master_key_header).json()

    assert (first["count"], second["count"]) == (2, 2)
    assert [(key["name"], key["is_effective"]) for key in first["data"]] == [("oldest", False)]
    assert [(key["name"], key["is_effective"]) for key in second["data"]] == [("newest", True)]
    assert client.get(view, params={"limit": 1001}, headers=master_key_header).status_code == 422


def test_pinning_and_turning_off_one_key_at_once_is_refused(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    workspace_id = _workspace_id(client, master_key_header)
    key = _add_key(client, master_key_header, api_key="tvly-1")

    response = client.patch(
        f"{API_ROOT}/workspaces/{workspace_id}/web-search-keys/{key['id']}",
        json={"is_default": True, "disabled": True},
        headers=master_key_header,
    )

    assert response.status_code == 400


# --- searching with them ------------------------------------------------------------------


def test_a_workspace_with_a_key_searches_with_it(
    client: TestClient, master_key_header: dict[str, str], api_key_header: dict[str, str]
) -> None:
    _add_key(client, master_key_header, api_key="tvly-org-secret")

    seen = _searched_with(client, api_key_header)

    assert seen["provider"] == "tavily"
    assert seen["provider_api_key"] == "tvly-org-secret"


def test_a_workspace_that_turned_its_keys_off_uses_the_deployments_search(
    client: TestClient, master_key_header: dict[str, str], api_key_header: dict[str, str]
) -> None:
    workspace_id = _workspace_id(client, master_key_header)
    key = _add_key(client, master_key_header, api_key="tvly-org-secret")
    client.patch(
        f"{API_ROOT}/workspaces/{workspace_id}/web-search-keys/{key['id']}",
        json={"disabled": True},
        headers=master_key_header,
    )

    seen = _searched_with(client, api_key_header)

    assert "provider" not in seen
    assert seen["base_url"] == _SEARCH_URL


def test_a_workspace_with_no_key_uses_the_deployments_search_unchanged(
    client: TestClient, api_key_header: dict[str, str]
) -> None:
    seen = _searched_with(client, api_key_header)

    assert "provider" not in seen
    assert seen["base_url"] == _SEARCH_URL


def test_an_archived_key_is_not_searched_with(
    client: TestClient, master_key_header: dict[str, str], api_key_header: dict[str, str]
) -> None:
    key = _add_key(client, master_key_header, api_key="tvly-org-secret")
    client.post(f"{_KEYS}/{key['id']}/archive", headers=master_key_header)

    assert "provider" not in _searched_with(client, api_key_header)


# --- who may do what: built at the service, since the routes always act as the bootstrap owner


def _service(db: AsyncSession) -> WebSearchKeyService:
    organizations = OrganizationService(db, membership_listener=None)
    uow = UnitOfWork(db)
    return WebSearchKeyService(
        uow,
        keys=OrgWebSearchKeyRepository(uow),
        overrides=WorkspaceWebSearchKeyOverrideRepository(uow),
        organizations=organizations,
        workspaces=WorkspaceAccess(db, organizations),
        lock_workspace=WorkspaceRepository(db).lock,
        config=GatewayConfig(),
    )


async def _tenant(db: AsyncSession) -> tuple[User, User, Workspace]:
    """An organization owner, a plain member, and a workspace the member belongs to."""
    organization = await OrganizationRepository(db).create_organization(
        name="Acme", slug="acme", created_by_user_id=None
    )
    users = []
    for role, name in (("owner", "Olive Owner"), ("member", "Mo Member")):
        user = await UserRepository(db).create_local_identity(full_name=name, active_organization_id=organization.id)
        await OrganizationMemberRepository(db).create_membership(
            organization_id=organization.id, user_id=user.id, role=role
        )
        users.append(user)
    owner, member = users
    workspace = await WorkspaceRepository(db).create_workspace(
        name="Default", organization_id=organization.id, created_by_user_id=owner.id
    )
    await WorkspaceMemberRepository(db).create(workspace_id=workspace.id, user_id=member.id, role="member")
    await db.commit()
    return owner, member, workspace


@pytest.mark.asyncio
async def test_a_plain_member_can_neither_list_nor_add_keys(async_db: AsyncSession) -> None:
    _, member, _ = await _tenant(async_db)
    service = _service(async_db)

    with pytest.raises(NotAuthorizedError):
        await service.list_keys(user=member, include_archived=False, skip=0, limit=10)
    with pytest.raises(NotAuthorizedError):
        await service.create_key(
            user=member, request=OrgWebSearchKeyCreateRequest(provider="tavily", name="n", api_key="k")
        )


@pytest.mark.asyncio
async def test_a_workspace_member_sees_the_workspace_key_but_cannot_change_it(async_db: AsyncSession) -> None:
    owner, member, workspace = await _tenant(async_db)
    service = _service(async_db)
    key = await service.create_key(
        user=owner, request=OrgWebSearchKeyCreateRequest(provider="tavily", name="n", api_key="tvly-1")
    )

    seen = await service.list_workspace_keys(user=member, workspace_id=workspace.id)

    assert [(row.org_web_search_key_id, row.is_effective) for row in seen.data] == [(key.id, True)]
    with pytest.raises(NotAuthorizedError):
        await service.set_workspace_override(
            user=member,
            workspace_id=workspace.id,
            key_id=key.id,
            request=WorkspaceWebSearchKeyOverrideRequest(disabled=True),
        )


@pytest.mark.asyncio
async def test_another_organizations_key_reads_as_absent(async_db: AsyncSession) -> None:
    owner, _, _ = await _tenant(async_db)
    other = await OrganizationRepository(async_db).create_organization(
        name="Other", slug="other", created_by_user_id=None
    )
    stranger = await UserRepository(async_db).create_local_identity(
        full_name="Stan Stranger", active_organization_id=other.id
    )
    await OrganizationMemberRepository(async_db).create_membership(
        organization_id=other.id, user_id=stranger.id, role="owner"
    )
    await async_db.commit()
    service = _service(async_db)
    key = await service.create_key(
        user=owner, request=OrgWebSearchKeyCreateRequest(provider="tavily", name="n", api_key="tvly-1")
    )

    with pytest.raises(WebSearchKeyNotFoundError):
        await service.archive_key(user=stranger, key_id=key.id)


# --- keys of two kinds ---------------------------------------------------------------------


def test_a_key_has_a_kind_and_may_carry_its_own_endpoint_and_options(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    search = _add_key(client, master_key_header, provider="exa", name="exa", api_key="exa-1")
    fetch = _add_key(
        client,
        master_key_header,
        provider="tabstack",
        name="pages",
        api_key="tbs-1",
        kind="fetch",
        api_base="https://tabstack.example/v1",
        options={"effort": "standard", "geo_target": {"api_token": "masked-in-the-answer"}},
    )

    assert (search["kind"], search["api_base"], search["options"]) == ("search", None, None)
    assert (fetch["kind"], fetch["api_base"]) == ("fetch", "https://tabstack.example/v1")
    assert fetch["options"]["effort"] == "standard"
    assert "masked-in-the-answer" not in str(fetch)
    cleared = client.patch(f"{_KEYS}/{fetch['id']}", json={"api_base": None}, headers=master_key_header)
    assert cleared.status_code == 200, cleared.text
    assert (cleared.json()["api_base"], cleared.json()["options"]["effort"]) == (None, "standard")


def test_a_keys_kind_is_fixed(client: TestClient, master_key_header: dict[str, str]) -> None:
    key = _add_key(client, master_key_header, provider="exa", api_key="exa-1")

    response = client.patch(f"{_KEYS}/{key['id']}", json={"kind": "fetch"}, headers=master_key_header)

    assert response.status_code == 422


@pytest.mark.parametrize(
    "body",
    [
        {"provider": "fake"},
        {"provider": "fake", "kind": "fetch"},
        {"provider": "builtin", "kind": "fetch"},
        {"provider": "tabstack"},
        {"provider": "mojeek", "kind": "fetch"},
        {"provider": "exa", "api_key": None},
        {"provider": "tavily", "api_base": "https://tavily.example"},
        {"provider": "brave", "options": {"country": "de"}},
        {"provider": "exa", "api_base": "http://exa.example"},
        {"provider": "exa", "options": {"no_such_option": 1}},
    ],
    ids=[
        "fake-search",
        "fake-fetch",
        "builtin-fetch",
        "search-on-a-fetch-provider",
        "fetch-on-a-search-provider",
        "keyless-on-a-keyed-provider",
        "tavily-endpoint",
        "brave-options",
        "key-over-http",
        "unknown-option",
    ],
)
def test_a_key_its_provider_could_not_be_called_with_is_refused(
    client: TestClient, master_key_header: dict[str, str], body: dict[str, Any]
) -> None:
    response = client.post(_KEYS, json={"name": "k", "api_key": "secret-1", **body}, headers=master_key_header)

    assert response.status_code == 400, response.text
    assert client.get(_KEYS, headers=master_key_header).json()["count"] == 0


def test_renaming_a_key_or_replacing_its_secret_does_not_recheck_what_it_stores(
    client: TestClient, master_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    key = _add_key(client, master_key_header, provider="exa", api_key="exa-1", options={"type": "fast"})
    # As if a library upgrade had retired the option the key stores.
    monkeypatch.setattr(
        _web_search_keys, "option_problems", lambda *_args: {"type": "option 'type' is not one the provider knows"}
    )

    rotated = client.patch(
        f"{_KEYS}/{key['id']}", json={"name": "renamed", "api_key": "exa-2"}, headers=master_key_header
    )
    resent = client.patch(f"{_KEYS}/{key['id']}", json={"options": {"type": "fast"}}, headers=master_key_header)

    assert rotated.status_code == 200, rotated.text
    assert (rotated.json()["name"], rotated.json()["last4"]) == ("renamed", "xa-2")
    assert resent.status_code == 400


def test_keys_of_different_kinds_may_share_a_provider_and_a_name(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _add_key(client, master_key_header, provider="exa", name="ours", api_key="exa-1")
    _add_key(client, master_key_header, provider="exa", name="ours", api_key="exa-2", kind="fetch")

    again = client.post(
        _KEYS,
        json={"provider": "exa", "name": "ours", "api_key": "exa-3", "kind": "fetch"},
        headers=master_key_header,
    )

    assert again.status_code == 409
    assert "fetch" in again.json()["detail"]


def test_one_default_per_kind(client: TestClient, master_key_header: dict[str, str]) -> None:
    exa = _add_key(client, master_key_header, provider="exa", name="exa", api_key="exa-1")
    tavily = _add_key(client, master_key_header, provider="tavily", name="tavily", api_key="tvly-1")
    pages = _add_key(client, master_key_header, provider="tabstack", name="pages", api_key="tbs-1", kind="fetch")
    for key in (exa, pages, tavily):
        assert client.post(f"{_KEYS}/{key['id']}/default", headers=master_key_header).status_code == 200

    defaults = {
        key["name"] for key in client.get(_KEYS, headers=master_key_header).json()["data"] if key["is_org_default"]
    }

    assert defaults == {"tavily", "pages"}


@pytest.mark.asyncio
async def test_two_defaults_of_one_kind_at_once_end_in_the_conflict_error(async_db: AsyncSession) -> None:
    """The index arbitrates: a writer that did not clear the other default meets it there."""

    class _NotClearing(OrgWebSearchKeyRepository):
        async def set_org_default(self, key: OrgWebSearchKey) -> OrgWebSearchKey:
            key.is_org_default = True
            try:
                await self.db.flush()
            except Exception:
                raise WebSearchKeyConflict from None
            return key

    owner, _, _ = await _tenant(async_db)
    service = _service(async_db)
    first = await service.create_key(
        user=owner, request=OrgWebSearchKeyCreateRequest(provider="exa", name="a", api_key="exa-1")
    )
    second = await service.create_key(
        user=owner, request=OrgWebSearchKeyCreateRequest(provider="mojeek", name="b", api_key="mjk-1")
    )
    await service.set_default(user=owner, key_id=first.id)
    racing = _service(async_db)
    racing.keys = _NotClearing(racing.uow)

    with pytest.raises(WebSearchKeyDefaultConflictError):
        await racing.set_default(user=owner, key_id=second.id)


def test_pinning_a_key_unpins_only_the_workspaces_other_key_of_its_kind(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    workspace_id = _workspace_id(client, master_key_header)
    view = f"{API_ROOT}/workspaces/{workspace_id}/web-search-keys"
    first = _add_key(client, master_key_header, provider="exa", name="first", api_key="exa-1")
    second = _add_key(client, master_key_header, provider="mojeek", name="second", api_key="mjk-1")
    pages = _add_key(client, master_key_header, provider="tabstack", name="pages", api_key="tbs-1", kind="fetch")
    for key in (first, pages, second):
        assert client.patch(f"{view}/{key['id']}", json={"is_default": True}, headers=master_key_header).is_success

    listed = client.get(view, headers=master_key_header).json()

    assert {key["name"] for key in listed["data"] if key["is_default"]} == {"second", "pages"}
    assert {key["name"] for key in listed["data"] if key["is_effective"]} == {"second", "pages"}
    assert (listed["search_provider"], listed["fetch_provider"]) == ("mojeek", "tabstack")


def test_the_workspace_listing_reports_what_its_calls_run_on(
    client: TestClient, master_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace_id = _workspace_id(client, master_key_header)
    view = f"{API_ROOT}/workspaces/{workspace_id}/web-search-keys"

    on_the_deployment = client.get(view, headers=master_key_header).json()
    monkeypatch.delenv("OTARI_WEB_SEARCH_URL")
    on_nothing = client.get(view, headers=master_key_header).json()
    key = _add_key(client, master_key_header, provider="exa", api_key="exa-1")
    turned_off = client.patch(f"{view}/{key['id']}", json={"disabled": True}, headers=master_key_header).json()
    on_its_key = client.delete(f"{view}/{key['id']}", headers=master_key_header).json()

    # The legacy URL is the deployment's search; the built-in fetcher its fetch.
    assert (on_the_deployment["search_provider"], on_the_deployment["fetch_provider"]) == ("searxng", "builtin")
    assert (on_nothing["search_provider"], on_nothing["fetch_provider"]) == (None, "builtin")
    assert turned_off["search_provider"] is None
    assert on_its_key["search_provider"] == "exa"


# --- serving with them on a deployment with no search of its own ---------------------------------


@pytest.fixture
def no_search_of_its_own(monkeypatch: pytest.MonkeyPatch, test_config: GatewayConfig) -> None:
    monkeypatch.delenv("OTARI_WEB_SEARCH_URL")
    monkeypatch.setattr(test_config, "web_search_extract", False)
    monkeypatch.setattr(test_config, "web_fetch_enabled", True)


def test_a_tavily_key_serves_an_explicit_declaration_where_the_deployment_has_no_search(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    no_search_of_its_own: None,
) -> None:
    _add_key(client, master_key_header, api_key="tvly-org-secret")

    seen = _searched_with(client, api_key_header)

    assert (seen["provider"], seen["provider_api_key"]) == ("tavily", "tvly-org-secret")


def test_the_workspace_page_counts_its_key_where_the_deployment_has_no_search(
    client: TestClient, master_key_header: dict[str, str], no_search_of_its_own: None
) -> None:
    page = f"{API_ROOT}/workspaces/{_workspace_id(client, master_key_header)}/web-search"

    before = client.get(page, headers=master_key_header).json()
    _add_key(client, master_key_header, provider="exa", api_key="exa-1")
    after = client.get(page, headers=master_key_header).json()

    assert (before["web_search_configured"], after["web_search_configured"]) == (False, True)


def _call(name: str, arguments: str) -> ChatCompletion:
    """A model round asking for one ``name`` call, or the answer when ``name`` is empty."""
    calls = [
        ChatCompletionMessageFunctionToolCall(
            id="call_0", type="function", function=Function(name=name, arguments=arguments)
        )
    ]
    return ChatCompletion(
        id="chatcmpl-1",
        object="chat.completion",
        created=0,
        model=MODEL_NAME,
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant", content=None if name else "done", tool_calls=cast(Any, calls if name else None)
                ),
                finish_reason="tool_calls" if name else "stop",
            )
        ],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


@contextmanager
def _scripted(*rounds: ChatCompletion) -> Iterator[None]:
    remaining = iter(rounds)

    async def fake(**_kwargs: Any) -> ChatCompletion:
        return next(remaining)

    with patch("gateway.services.mcp_loop.acompletion", new=fake):
        yield


def _answer_provider_calls(client: TestClient, answer: Callable[[httpx.Request], httpx.Response]) -> None:
    async def web_provider_client() -> AsyncIterator[httpx.AsyncClient]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as answering:
            yield answering

    # The completion routes take the optional client, direct search the required one.
    for dependency in (get_optional_web_provider_client, get_web_provider_client):
        client.app.dependency_overrides[dependency] = web_provider_client  # type: ignore[attr-defined]


def _tool_line(db_session_factory: Callable[[], Session], user_id: str, meter: str) -> dict[str, Any]:
    with db_session_factory() as db:
        row = db.scalars(
            select(UsageLog).where(UsageLog.user_id == user_id).order_by(UsageLog.timestamp.desc())
        ).first()
        assert row is not None
        [line] = [entry for entry in (row.pricing_breakdown or []) if entry["meter"] == meter]
        return line


def _price(client: TestClient, headers: dict[str, str], key: str, per_call: float) -> None:
    response = client.post(
        f"{API_ROOT}/pricing",
        json={"model_key": key, "input_price_per_million": per_call * 1_000_000, "output_price_per_million": 0.0},
        headers=headers,
    )
    assert response.status_code == 200, response.text


@pytest.mark.parametrize(("reported", "billed"), [(0.005, 0.005), (None, 0.002)], ids=["reported", "fallback-row"])
def test_a_pinned_exa_key_searches_in_the_loop_and_is_billed_as_the_organizations(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
    db_session_factory: Callable[[], Session],
    no_search_of_its_own: None,
    reported: float | None,
    billed: float,
) -> None:
    workspace_id = _workspace_id(client, master_key_header)
    key = _add_key(client, master_key_header, provider="exa", api_key="exa-org-secret")
    client.patch(
        f"{API_ROOT}/workspaces/{workspace_id}/web-search-keys/{key['id']}",
        json={"is_default": True},
        headers=master_key_header,
    )
    _price(client, master_key_header, "otari:web_search", 0.002)
    seen: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body: dict[str, Any] = {"results": [{"url": "https://a.example", "title": "A", "text": "Page A."}]}
        if reported is not None:
            body["costDollars"] = {"total": reported}
        return httpx.Response(200, json=body)

    _answer_provider_calls(client, answer)
    with _scripted(_call("web_search", '{"query": "otari"}'), _call("", "")):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "q"}],
                "tools": [{"type": "otari_web_search"}],
            },
            headers=api_key_header,
        )

    assert response.status_code == 200, response.text
    [request] = seen
    assert (request.url.host, request.headers["x-api-key"]) == ("api.exa.ai", "exa-org-secret")
    line = _tool_line(db_session_factory, api_key_obj["user_id"], "web_search_calls")
    assert (line["label"], line["units"]) == ("org:exa", 1)
    assert line["cost"] == pytest.approx(billed)


def test_a_fetch_key_serves_the_fetch_tool(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
    db_session_factory: Callable[[], Session],
    no_search_of_its_own: None,
) -> None:
    _add_key(client, master_key_header, provider="tabstack", name="pages", api_key="tbs-org-secret", kind="fetch")
    _price(client, master_key_header, "otari:web_fetch", 0.001)
    seen: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        page = {"url": "https://example.com/", "content": "Example text.", "metadata": {"title": "Example"}}
        return httpx.Response(200, json=page)

    _answer_provider_calls(client, answer)
    with _scripted(_call("web_fetch", '{"url": "https://example.com/"}'), _call("", "")):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "q"}],
                "tools": [{"type": "otari_web_fetch"}],
            },
            headers=api_key_header,
        )

    assert response.status_code == 200, response.text
    [request] = seen
    assert request.url.host == "api.tabstack.ai"
    assert request.headers["authorization"] == "Bearer tbs-org-secret"
    line = _tool_line(db_session_factory, api_key_obj["user_id"], "web_fetch_calls")
    assert (line["label"], line["units"]) == ("org:tabstack", 1)
    assert line["cost"] == pytest.approx(0.001)
