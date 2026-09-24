"""A mandate that names a hosted guardrail: what it may carry, and who may name which one."""

import uuid
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.adapters.hosted_guardrail_adapter import NullHostedGuardrailAdapter
from gateway.core.config import API_ROOT
from gateway.models.tenancy import Organization, User
from gateway.ports.hosted_guardrail_port import (
    HostedGuardrailPort,
    HostedGuardrailUnfundedError,
    HostedGuardrailVerdict,
)
from gateway.repositories.tenancy import OrganizationMemberRepository, OrganizationRepository, UserRepository
from gateway.services.secret_box import generate_secret_key
from gateway.services.tenancy.errors import (
    OrganizationGuardrailCheckFailedError,
    OrganizationGuardrailHostedAloneError,
    OrganizationGuardrailHostedNotFoundError,
    OrganizationGuardrailUnfundedError,
)
from gateway.services.tenancy.organization_guardrail_service import (
    OrganizationGuardrailCreate,
    OrganizationGuardrailService,
    OrganizationGuardrailTest,
    OrganizationGuardrailUpdate,
    resolve_organization_guardrails,
)

from .hosted_guardrail_helpers import LAKERA, HostedGuardrails, bind_hosted_guardrails

PUBLIC_URL = "https://93.184.216.34/guardrails"


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    yield


async def _owner(db: AsyncSession, slug: str = "acme") -> tuple[Organization, User]:
    organization = await OrganizationRepository(db).create_organization(
        name=slug.title(), slug=slug, created_by_user_id=None
    )
    user = await UserRepository(db).create_local_identity(full_name="Owner", active_organization_id=organization.id)
    await OrganizationMemberRepository(db).create_membership(
        organization_id=organization.id, user_id=user.id, role="owner"
    )
    return organization, user


def _hosted_mandate(**overrides: object) -> OrganizationGuardrailCreate:
    body: dict[str, object] = {
        "profile": "prompt-injection",
        "hosted_guardrail_id": str(LAKERA.id),
        "mode": "block",
        "applies_to_all_workspaces": True,
    }
    body.update(overrides)
    return OrganizationGuardrailCreate.model_validate(body)


@pytest.mark.asyncio
async def test_an_organization_mandates_a_hosted_guardrail_it_is_offered(async_db: AsyncSession) -> None:
    organization, owner = await _owner(async_db)
    service = OrganizationGuardrailService(async_db, hosted_guardrails=HostedGuardrails(offered_to=organization.id))

    created = await service.create_guardrail(user=owner, request=_hosted_mandate())

    assert created.hosted_guardrail_id == LAKERA.id
    assert created.url is None
    assert created.definition_id is None
    assert created.has_credential is False
    listed = await service.list_guardrails(user=owner)
    assert [row.hosted_guardrail_id for row in listed.data] == [LAKERA.id]


@pytest.mark.asyncio
async def test_a_hosted_guardrail_offered_elsewhere_reads_as_not_found(async_db: AsyncSession) -> None:
    organization, owner = await _owner(async_db)
    service = OrganizationGuardrailService(async_db, hosted_guardrails=HostedGuardrails(offered_to=uuid.uuid4()))

    with pytest.raises(OrganizationGuardrailHostedNotFoundError):
        await service.create_guardrail(user=owner, request=_hosted_mandate())


@pytest.mark.asyncio
async def test_the_plain_build_offers_no_hosted_guardrail_to_name(async_db: AsyncSession) -> None:
    _, owner = await _owner(async_db)

    with pytest.raises(OrganizationGuardrailHostedNotFoundError):
        await OrganizationGuardrailService(async_db).create_guardrail(user=owner, request=_hosted_mandate())


@pytest.mark.parametrize(
    "extra",
    [{"url": PUBLIC_URL}, {"url": PUBLIC_URL, "credential": "bearer"}, {"definition_id": str(uuid.uuid4())}],
)
@pytest.mark.asyncio
async def test_a_hosted_mandate_names_no_other_backend(async_db: AsyncSession, extra: dict[str, object]) -> None:
    organization, owner = await _owner(async_db)
    service = OrganizationGuardrailService(async_db, hosted_guardrails=HostedGuardrails(offered_to=organization.id))

    with pytest.raises(OrganizationGuardrailHostedAloneError):
        await service.create_guardrail(user=owner, request=_hosted_mandate(**extra))


@pytest.mark.asyncio
async def test_an_update_cannot_add_an_endpoint_beside_a_stored_hosted_guardrail(async_db: AsyncSession) -> None:
    organization, owner = await _owner(async_db)
    service = OrganizationGuardrailService(async_db, hosted_guardrails=HostedGuardrails(offered_to=organization.id))
    created = await service.create_guardrail(user=owner, request=_hosted_mandate())

    with pytest.raises(OrganizationGuardrailHostedAloneError):
        await service.update_guardrail(
            user=owner, guardrail_id=created.id, request=OrganizationGuardrailUpdate(url=PUBLIC_URL)
        )


@pytest.mark.asyncio
async def test_an_explicit_null_clears_the_hosted_guardrail(async_db: AsyncSession) -> None:
    organization, owner = await _owner(async_db)
    service = OrganizationGuardrailService(async_db, hosted_guardrails=HostedGuardrails(offered_to=organization.id))
    created = await service.create_guardrail(user=owner, request=_hosted_mandate())

    cleared = await service.update_guardrail(
        user=owner,
        guardrail_id=created.id,
        request=OrganizationGuardrailUpdate.model_validate({"hosted_guardrail_id": None, "url": PUBLIC_URL}),
    )

    assert cleared.hosted_guardrail_id is None
    assert cleared.url == PUBLIC_URL


@pytest.mark.asyncio
async def test_an_update_to_a_hosted_guardrail_not_offered_reads_as_not_found(async_db: AsyncSession) -> None:
    organization, owner = await _owner(async_db)
    service = OrganizationGuardrailService(async_db, hosted_guardrails=HostedGuardrails(offered_to=organization.id))
    created = await service.create_guardrail(user=owner, request=OrganizationGuardrailCreate(profile="plain"))

    with pytest.raises(OrganizationGuardrailHostedNotFoundError):
        await service.update_guardrail(
            user=owner,
            guardrail_id=created.id,
            request=OrganizationGuardrailUpdate(hosted_guardrail_id=uuid.uuid4()),
        )


@pytest.mark.asyncio
async def test_the_request_path_reads_the_hosted_guardrail(async_db: AsyncSession) -> None:
    organization, owner = await _owner(async_db)
    service = OrganizationGuardrailService(async_db, hosted_guardrails=HostedGuardrails(offered_to=organization.id))
    await service.create_guardrail(user=owner, request=_hosted_mandate())
    workspace_id = uuid.uuid4()

    resolved = await resolve_organization_guardrails(
        async_db, organization_id=organization.id, workspace_id=workspace_id
    )

    assert [(entry.config.profile, entry.hosted_guardrail_id, entry.credential) for entry in resolved] == [
        ("prompt-injection", LAKERA.id, None)
    ]


@pytest.mark.asyncio
async def test_testing_a_hosted_mandate_runs_it_through_the_port(async_db: AsyncSession) -> None:
    organization, owner = await _owner(async_db)
    # Read now: a test ends its read with a rollback, which expires every loaded row.
    organization_id = organization.id
    hosted = HostedGuardrails(offered_to=organization_id, verdict=HostedGuardrailVerdict(valid=False, score=0.9))
    service = OrganizationGuardrailService(async_db, hosted_guardrails=hosted)
    created = await service.create_guardrail(user=owner, request=_hosted_mandate(validate_kwargs={"threshold": 0.5}))

    result = await service.test_guardrail(
        user=owner, guardrail_id=created.id, request=OrganizationGuardrailTest(text="ignore all"), default_url=None
    )

    assert (result.valid, result.score) == (False, 0.9)
    [evaluation] = hosted.evaluations
    assert evaluation.organization_id == organization_id
    assert evaluation.text == "ignore all"
    assert evaluation.validate_kwargs == {"threshold": 0.5}
    assert evaluation.idempotency_key.startswith("test:")


@pytest.mark.asyncio
async def test_testing_a_hosted_mandate_reports_an_unfunded_organization(async_db: AsyncSession) -> None:
    organization, owner = await _owner(async_db)
    hosted = HostedGuardrails(offered_to=organization.id)
    service = OrganizationGuardrailService(async_db, hosted_guardrails=hosted)
    created = await service.create_guardrail(user=owner, request=_hosted_mandate())

    hosted.error = HostedGuardrailUnfundedError()
    with pytest.raises(OrganizationGuardrailUnfundedError):
        await service.test_guardrail(
            user=owner, guardrail_id=created.id, request=OrganizationGuardrailTest(text="hi"), default_url=None
        )

    hosted.error = None
    hosted.offered_to = uuid.uuid4()
    await async_db.refresh(owner)  # expired by the first test's rollback
    with pytest.raises(OrganizationGuardrailCheckFailedError):
        await service.test_guardrail(
            user=owner, guardrail_id=created.id, request=OrganizationGuardrailTest(text="hi"), default_url=None
        )


def test_the_mandate_route_asks_the_bound_port(client: TestClient, master_key_header: dict[str, str]) -> None:
    """The route builds its service with the port the build bound, not a build of its own."""
    body = {"profile": "prompt-injection", "hosted_guardrail_id": str(LAKERA.id), "applies_to_all_workspaces": True}
    refused = client.post(f"{API_ROOT}/organizations/me/guardrails", json=body, headers=master_key_header)
    assert refused.status_code == 404, refused.text

    organization_id = client.get(f"{API_ROOT}/organizations/me", headers=master_key_header).json()["organization"]["id"]
    bind_hosted_guardrails(client, HostedGuardrails(offered_to=uuid.UUID(organization_id)))
    try:
        created = client.post(f"{API_ROOT}/organizations/me/guardrails", json=body, headers=master_key_header)
        assert created.status_code == 201, created.text
        assert created.json()["hosted_guardrail_id"] == str(LAKERA.id)
        offered = client.get(f"{API_ROOT}/organizations/me/hosted-guardrails", headers=master_key_header)
        assert [entry["id"] for entry in offered.json()["data"]] == [str(LAKERA.id)]
    finally:
        client.app.state.container.bind(HostedGuardrailPort, NullHostedGuardrailAdapter)  # type: ignore[attr-defined]
