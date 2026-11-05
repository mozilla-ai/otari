"""A workspace's narrowing of the hosted providers: the routes, dispatch and the catalog.

The service's authorization rules are pinned at the service layer below, as
`test_workspace_code_execution_policy.py` does, because the API can only act as
the one operator identity a standalone deployment has. Dispatch and the
catalog are pinned through the API with a real hosted provider, the setup
`test_hosted_provider_dispatch.py` uses: the provider call is captured at the
any-llm boundary and never made.
"""

import uuid
from collections.abc import Iterator
from unittest.mock import AsyncMock, patch

import pytest
from any_llm.types.model import Model
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.organizations_exceptions import NotAuthorizedError, WorkspaceNotFoundError
from gateway.exceptions.providers_exceptions import (
    WorkspaceHostedModelsInvalidError,
    WorkspaceHostedProviderNotFoundError,
)
from gateway.models.tenancy import Organization, User, Workspace
from gateway.ports.model_provider_port import HostedModels
from gateway.repositories.providers import WorkspaceHostedAccessRepository, WorkspaceProviderPolicyRepository
from gateway.repositories.tenancy import (
    OrganizationMemberRepository,
    OrganizationRepository,
    UserRepository,
    WorkspaceMemberRepository,
    WorkspaceRepository,
)
from gateway.schemas.providers import WorkspaceHostedProviderAccessRequest
from gateway.services.model_discovery_service import ProviderDiscovery
from gateway.services.providers import (
    WorkspaceHostedAccessDeniedError,
    WorkspaceHostedAccessService,
    refresh_workspace_hosted_access_cache,
    reset_workspace_hosted_access_cache,
)
from gateway.services.providers._workspace_hosted_access import ensure_workspace_may_use_hosted
from gateway.services.secret_box import generate_secret_key
from gateway.services.tenancy.authorization import WorkspaceAccess
from gateway.services.tenancy.organization_service import OrganizationService

from .conftest import shared_test_config

KEY = "sk-live-hosted-1234"


@pytest.fixture(scope="module")
def test_config(postgres_url: str) -> GatewayConfig:
    """The shared configuration declaring no provider, so nothing configured shadows a hosted one."""
    return shared_test_config(postgres_url, {})


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A developer's own shell key would credential the ladder and skip the port; the cache is per process."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    reset_workspace_hosted_access_cache()
    yield
    reset_workspace_hosted_access_cache()


@pytest.fixture(autouse=True)
def _discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _stub(impl_name: str, **_: object) -> ProviderDiscovery:
        return ProviderDiscovery(
            provider=impl_name,
            models=[
                Model(id=name, object="model", created=0, owned_by=impl_name) for name in ("gpt-4o", "gpt-4o-mini")
            ],
        )

    monkeypatch.setattr("gateway.services.providers._hosted_provider_service.test_provider_credentials", _stub)


class _MockCompletionError(Exception):
    """Raised to short-circuit the mocked provider call after capturing its kwargs."""


def _hosted_openai(client: TestClient, headers: dict[str, str]) -> None:
    for model in ("gpt-4o", "gpt-4o-mini"):
        priced = client.post(
            f"{API_ROOT}/pricing",
            json={"model_key": f"openai:{model}", "input_price_per_million": 2.5, "output_price_per_million": 10},
            headers=headers,
        )
        assert priced.status_code == 200, priced.text
    created = client.post(
        f"{API_ROOT}/hosted-providers",
        json={"provider": "openai", "api_key": KEY, "api_base": "https://hosted.test/v1"},
        headers=headers,
    )
    assert created.status_code == 201, created.text


def _api_key(client: TestClient, headers: dict[str, str]) -> tuple[dict[str, str], str]:
    """A key in the deployment's default workspace, and that workspace's id."""
    created = client.post(f"{API_ROOT}/keys", json={"key_name": "probe"}, headers=headers)
    assert created.status_code == 200, created.text
    workspaces = client.get(f"{API_ROOT}/workspaces", headers=headers)
    assert workspaces.status_code == 200, workspaces.text
    [workspace] = workspaces.json()["data"]
    return {API_KEY_HEADER: f"Bearer {created.json()['key']}"}, workspace["id"]


def _chat(client: TestClient, headers: dict[str, str], model: str) -> tuple[dict[str, object], int]:
    captured: dict[str, object] = {}

    async def fake_acompletion(**kwargs: object) -> None:
        captured.update(kwargs)
        raise _MockCompletionError

    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(side_effect=fake_acompletion)):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "Hi"}]},
            headers=headers,
        )
    return captured, response.status_code


def _listed(client: TestClient, headers: dict[str, str]) -> set[str]:
    response = client.get(f"{API_ROOT}/models", headers=headers)
    assert response.status_code == 200, response.text
    return {model["id"] for model in response.json()["data"]}


def test_a_workspace_lists_the_hosted_providers_it_inherits(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _hosted_openai(client, master_key_header)
    _, workspace_id = _api_key(client, master_key_header)

    response = client.get(f"{API_ROOT}/workspaces/{workspace_id}/hosted-providers", headers=master_key_header)

    assert response.status_code == 200, response.text
    assert response.json()["data"] == [
        {
            "provider": "openai",
            "enabled": True,
            "models": None,
            "customized": False,
            "served_models": ["gpt-4o", "gpt-4o-mini"],
            "served_by_own_key": False,
            "zero_data_retention": False,
        }
    ]


def test_switching_a_provider_off_refuses_dispatch_and_hides_it_until_reset(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _hosted_openai(client, master_key_header)
    key_headers, workspace_id = _api_key(client, master_key_header)
    url = f"{API_ROOT}/workspaces/{workspace_id}/hosted-providers/openai"

    off = client.put(url, json={"enabled": False}, headers=master_key_header)
    assert off.status_code == 200, off.text
    assert off.json()["customized"] is True

    captured, status_code = _chat(client, key_headers, "openai:gpt-4o")
    assert status_code == 403
    assert not captured, "a provider switched off for the workspace must never be called"
    assert not {model for model in _listed(client, key_headers) if model.startswith("openai/")}

    reset = client.delete(url, headers=master_key_header)
    assert reset.status_code == 200, reset.text
    assert reset.json()["customized"] is False

    captured, _ = _chat(client, key_headers, "openai:gpt-4o")
    assert captured["api_key"] == KEY


def test_narrowing_to_some_models_serves_only_those(client: TestClient, master_key_header: dict[str, str]) -> None:
    _hosted_openai(client, master_key_header)
    key_headers, workspace_id = _api_key(client, master_key_header)

    narrowed = client.put(
        f"{API_ROOT}/workspaces/{workspace_id}/hosted-providers/openai",
        json={"enabled": True, "models": ["gpt-4o-mini"]},
        headers=master_key_header,
    )
    assert narrowed.status_code == 200, narrowed.text
    assert narrowed.json()["models"] == ["gpt-4o-mini"]

    refused, refused_status = _chat(client, key_headers, "openai:gpt-4o")
    served, _ = _chat(client, key_headers, "openai:gpt-4o-mini")

    assert refused_status == 403
    assert not refused
    assert served["api_key"] == KEY
    listed = _listed(client, key_headers)
    assert "openai/gpt-4o-mini" in listed
    assert "openai/gpt-4o" not in listed


def test_a_model_the_provider_does_not_serve_is_refused(client: TestClient, master_key_header: dict[str, str]) -> None:
    _hosted_openai(client, master_key_header)
    _, workspace_id = _api_key(client, master_key_header)

    response = client.put(
        f"{API_ROOT}/workspaces/{workspace_id}/hosted-providers/openai",
        json={"enabled": True, "models": ["o3"]},
        headers=master_key_header,
    )

    assert response.status_code == 400, response.text


def test_an_unknown_hosted_provider_is_not_found(client: TestClient, master_key_header: dict[str, str]) -> None:
    _hosted_openai(client, master_key_header)
    _, workspace_id = _api_key(client, master_key_header)

    response = client.put(
        f"{API_ROOT}/workspaces/{workspace_id}/hosted-providers/mistral",
        json={"enabled": False},
        headers=master_key_header,
    )

    assert response.status_code == 404, response.text


SERVED: HostedModels = {"openai": frozenset({"gpt-4o", "gpt-4o-mini"}), "gemini": None}


async def _organization(db: AsyncSession, *, slug: str) -> Organization:
    return await OrganizationRepository(db).create_organization(name=slug.title(), slug=slug, created_by_user_id=None)


async def _member(db: AsyncSession, organization: Organization, *, role: str, full_name: str) -> User:
    user = await UserRepository(db).create_local_identity(full_name=full_name, active_organization_id=organization.id)
    await OrganizationMemberRepository(db).create_membership(
        organization_id=organization.id, user_id=user.id, role=role
    )
    return user


async def _workspace(db: AsyncSession, organization: Organization, *, owner: User) -> Workspace:
    workspace = await WorkspaceRepository(db).create_workspace(
        name="Engineering", organization_id=organization.id, created_by_user_id=owner.id
    )
    await WorkspaceMemberRepository(db).create(workspace_id=workspace.id, user_id=owner.id, role="owner")
    return workspace


def _service(
    db: AsyncSession, *, own_key: frozenset[str] = frozenset(), served: HostedModels = SERVED
) -> WorkspaceHostedAccessService:
    uow = UnitOfWork(db)

    async def hosted_models(_organization_id: uuid.UUID | None) -> HostedModels:
        return served

    async def byo_providers(_workspace_id: uuid.UUID) -> frozenset[str]:
        return own_key

    return WorkspaceHostedAccessService(
        uow,
        rules=WorkspaceHostedAccessRepository(uow),
        policies=WorkspaceProviderPolicyRepository(uow),
        access=WorkspaceAccess(db, OrganizationService(db, membership_listener=None)),
        hosted_models=hosted_models,
        byo_providers=byo_providers,
        refresh_cache=lambda: refresh_workspace_hosted_access_cache(uow),
    )


@pytest.mark.asyncio
async def test_a_write_reaches_the_dispatch_cache_and_a_reset_clears_it(async_db: AsyncSession) -> None:
    org = await _organization(async_db, slug="hosted-cache")
    owner = await _member(async_db, org, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, org, owner=owner)
    service = _service(async_db)

    await service.set_access(
        user=owner,
        workspace_id=workspace.id,
        provider="openai",
        request=WorkspaceHostedProviderAccessRequest(enabled=True, models=[" gpt-4o ", "gpt-4o"]),
    )

    ensure_workspace_may_use_hosted(workspace.id, "openai", "gpt-4o")
    with pytest.raises(WorkspaceHostedAccessDeniedError):
        ensure_workspace_may_use_hosted(workspace.id, "openai", "gpt-4o-mini")
    listed = await service.list_for_workspace(user=owner, workspace_id=workspace.id)
    assert [(row.provider, row.models, row.customized) for row in listed.data] == [
        ("gemini", None, False),
        ("openai", ["gpt-4o"], True),
    ]

    await service.reset_access(user=owner, workspace_id=workspace.id, provider="openai")

    ensure_workspace_may_use_hosted(workspace.id, "openai", "gpt-4o-mini")


@pytest.mark.asyncio
async def test_storing_the_inherited_state_stores_no_row(async_db: AsyncSession) -> None:
    org = await _organization(async_db, slug="hosted-noop")
    owner = await _member(async_db, org, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, org, owner=owner)

    result = await _service(async_db).set_access(
        user=owner,
        workspace_id=workspace.id,
        provider="openai",
        request=WorkspaceHostedProviderAccessRequest(enabled=True, models=None),
    )

    assert result.customized is False
    async with UnitOfWork(async_db) as uow:
        assert list(await WorkspaceHostedAccessRepository(uow).list_for_workspace(workspace.id)) == []


@pytest.mark.asyncio
async def test_a_provider_whose_list_names_a_dropped_model_can_still_be_switched_off(
    async_db: AsyncSession,
) -> None:
    org = await _organization(async_db, slug="hosted-stale")
    owner = await _member(async_db, org, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, org, owner=owner)
    await _service(async_db).set_access(
        user=owner,
        workspace_id=workspace.id,
        provider="openai",
        request=WorkspaceHostedProviderAccessRequest(enabled=True, models=["gpt-4o"]),
    )
    # The deployment has since stopped serving gpt-4o.
    dropped = _service(async_db, served={"openai": frozenset({"gpt-4o-mini"})})

    off = await dropped.set_access(
        user=owner,
        workspace_id=workspace.id,
        provider="openai",
        request=WorkspaceHostedProviderAccessRequest(enabled=False, models=["gpt-4o"]),
    )

    assert off.enabled is False
    with pytest.raises(WorkspaceHostedModelsInvalidError):
        await dropped.set_access(
            user=owner,
            workspace_id=workspace.id,
            provider="openai",
            request=WorkspaceHostedProviderAccessRequest(enabled=True, models=["gpt-4o"]),
        )


@pytest.mark.asyncio
async def test_an_empty_model_list_is_refused(async_db: AsyncSession) -> None:
    org = await _organization(async_db, slug="hosted-empty")
    owner = await _member(async_db, org, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, org, owner=owner)

    with pytest.raises(WorkspaceHostedModelsInvalidError):
        await _service(async_db).set_access(
            user=owner,
            workspace_id=workspace.id,
            provider="openai",
            request=WorkspaceHostedProviderAccessRequest(enabled=True, models=["  "]),
        )


@pytest.mark.asyncio
async def test_a_provider_with_no_roster_accepts_any_model_name(async_db: AsyncSession) -> None:
    org = await _organization(async_db, slug="hosted-roster")
    owner = await _member(async_db, org, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, org, owner=owner)

    result = await _service(async_db).set_access(
        user=owner,
        workspace_id=workspace.id,
        provider="gemini",
        request=WorkspaceHostedProviderAccessRequest(enabled=True, models=["gemini-2.5-pro"]),
    )

    assert result.models == ["gemini-2.5-pro"]


@pytest.mark.asyncio
async def test_a_provider_the_organization_cannot_use_is_not_found(async_db: AsyncSession) -> None:
    org = await _organization(async_db, slug="hosted-unknown")
    owner = await _member(async_db, org, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, org, owner=owner)

    with pytest.raises(WorkspaceHostedProviderNotFoundError):
        await _service(async_db).set_access(
            user=owner,
            workspace_id=workspace.id,
            provider="bedrock",
            request=WorkspaceHostedProviderAccessRequest(enabled=False),
        )


@pytest.mark.asyncio
async def test_a_plain_member_may_read_but_not_write(async_db: AsyncSession) -> None:
    org = await _organization(async_db, slug="hosted-member")
    owner = await _member(async_db, org, role="owner", full_name="Owner")
    plain = await _member(async_db, org, role="member", full_name="Plain")
    workspace = await _workspace(async_db, org, owner=owner)
    await WorkspaceMemberRepository(async_db).create(workspace_id=workspace.id, user_id=plain.id, role="member")
    service = _service(async_db, own_key=frozenset({"openai"}))

    listed = await service.list_for_workspace(user=plain, workspace_id=workspace.id)
    with pytest.raises(NotAuthorizedError):
        await service.set_access(
            user=plain,
            workspace_id=workspace.id,
            provider="openai",
            request=WorkspaceHostedProviderAccessRequest(enabled=False),
        )

    assert {row.provider: row.served_by_own_key for row in listed.data} == {"gemini": False, "openai": True}


@pytest.mark.asyncio
async def test_a_workspace_of_another_organization_is_not_found(async_db: AsyncSession) -> None:
    mine = await _organization(async_db, slug="hosted-mine")
    theirs = await _organization(async_db, slug="hosted-theirs")
    me = await _member(async_db, mine, role="owner", full_name="Me")
    them = await _member(async_db, theirs, role="owner", full_name="Them")
    workspace = await _workspace(async_db, theirs, owner=them)

    with pytest.raises(WorkspaceNotFoundError):
        await _service(async_db).set_access(
            user=me,
            workspace_id=workspace.id,
            provider="openai",
            request=WorkspaceHostedProviderAccessRequest(enabled=False),
        )
