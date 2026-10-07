"""An organization's own web search keys: managing them, and the request searching with them.

The request cases go through ``/api/v1/messages`` with the search backend patched out,
so what is asserted is which provider and key the backend was built with.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from any_llm.types.messages import MessageResponse, MessageUsage, TextBlock
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import API_ROOT
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.organizations_exceptions import NotAuthorizedError
from gateway.exceptions.tools_exceptions import WebSearchKeyNotFoundError
from gateway.models.tenancy import User, Workspace
from gateway.repositories.tenancy import (
    OrganizationMemberRepository,
    OrganizationRepository,
    UserRepository,
    WorkspaceMemberRepository,
    WorkspaceRepository,
)
from gateway.repositories.tools import OrgWebSearchKeyRepository, WorkspaceWebSearchKeyOverrideRepository
from gateway.schemas.tools import OrgWebSearchKeyCreateRequest, WorkspaceWebSearchKeyOverrideRequest
from gateway.services.secret_box import generate_secret_key
from gateway.services.tenancy import OrganizationService
from gateway.services.tenancy.authorization import WorkspaceAccess
from gateway.services.tools import WebSearchKeyService

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
    client: TestClient, headers: dict[str, str], *, provider: str = "tavily", name: str = "ours", api_key: str
) -> dict[str, Any]:
    created = client.post(_KEYS, json={"provider": provider, "name": name, "api_key": api_key}, headers=headers)
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
