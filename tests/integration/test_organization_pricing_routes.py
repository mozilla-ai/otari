"""The organization pricing-override endpoints, and what a request settles at.

Three things are covered here that the SQLite unit tests
(`tests/unit/test_organization_pricing_resolution.py`) cannot reach: the HTTP
surface and its statuses, the role gate (at the service layer, for the reason
`test_tenancy_authorization.py` explains), and the end-to-end proof that a
request actually bills at the override rate.
"""

import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from any_llm import LLMProvider
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from sqlmodel import col

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.models.api_keys import APIKey
from gateway.models.pricing import ModelPricing, OrganizationModelPricing
from gateway.models.tenancy import DashboardSession, Organization, OrganizationMember, User, Workspace
from gateway.ports.model_provider_port import HostedAccessDeniedError, HostedCredential, ModelProviderPort
from gateway.repositories.tenancy import (
    OrganizationMemberRepository,
    OrganizationRepository,
    OrgProviderKeyRepository,
    UserRepository,
    WorkspaceProviderKeyOverrideRepository,
    WorkspaceRepository,
)
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, hash_session_token
from gateway.services.organization_pricing_service import (
    OrganizationPricingService,
    PricingOverrideInput,
)
from gateway.services.pricing_service import find_model_pricing
from gateway.services.provider_kwargs import credential_ladder_exhausted, get_provider_kwargs
from gateway.services.secret_box import encrypt_secret, generate_secret_key
from gateway.services.tenancy.errors import (
    NotAuthorizedError,
    OrganizationPricingManagedModelError,
    OrganizationPricingNotFoundError,
    OrganizationPricingOverlapError,
    TenancyValidationError,
)
from gateway.services.tenancy.org_provider_key_service import refresh_org_provider_cache, reset_org_provider_cache
from gateway.services.workspace_scope import (
    organization_for_key_id,
    organization_for_workspace_id,
    reset_key_workspace_cache,
)

_ENDPOINT = f"{API_ROOT}/organizations/me/pricing"
_MODEL_KEY = "openai:gpt-4o"


@pytest.fixture(autouse=True)
def _isolate_the_workspace_cache() -> Iterator[None]:
    """The key-to-workspace memo is process-global, so clear it either side.

    Structural rather than remembered: several tests here delete or re-mint the
    rows the memo caches, and one deliberately deletes a workspace out from under
    it to prove the cache is real. Leaving that entry behind made the next test's
    correctness depend on it calling the reset itself.
    """
    reset_key_workspace_cache()
    yield
    reset_key_workspace_cache()


def _body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model_key": _MODEL_KEY,
        "input_price_per_million": 2.5,
        "output_price_per_million": 5.0,
    }
    body.update(overrides)
    return body


# =============================================================================
# The HTTP surface
# =============================================================================


def test_an_override_is_created_listed_replaced_and_deleted(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    created = client.post(_ENDPOINT, json=_body(), headers=master_key_header)
    assert created.status_code == status.HTTP_201_CREATED, created.text
    override = created.json()
    assert override["model_key"] == _MODEL_KEY
    assert override["input_price_per_million"] == 2.5
    assert override["effective_to"] is None

    listed = client.get(_ENDPOINT, headers=master_key_header)
    assert listed.status_code == status.HTTP_200_OK, listed.text
    assert listed.json()["count"] == 1
    assert listed.json()["data"][0]["id"] == override["id"]

    replaced = client.put(
        f"{_ENDPOINT}/{override['id']}",
        json={
            "input_price_per_million": 1.0,
            "output_price_per_million": 2.0,
            # Required on a replacement, so an omitted start cannot silently move
            # the stored period to the present.
            "effective_from": override["effective_from"],
        },
        headers=master_key_header,
    )
    assert replaced.status_code == status.HTTP_200_OK, replaced.text
    # The period is the one that was sent, not "now".
    assert replaced.json()["effective_from"] == override["effective_from"]
    assert replaced.json()["input_price_per_million"] == 1.0
    # The key is immutable, so a replacement keeps it.
    assert replaced.json()["model_key"] == _MODEL_KEY

    removed = client.delete(f"{_ENDPOINT}/{override['id']}", headers=master_key_header)
    assert removed.status_code == status.HTTP_204_NO_CONTENT, removed.text
    assert client.get(_ENDPOINT, headers=master_key_header).json()["count"] == 0


def test_an_overlapping_period_is_refused_with_a_conflict(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """The definition of done's second line, over HTTP."""
    start = datetime.now(UTC) - timedelta(days=1)
    first = client.post(
        _ENDPOINT,
        json=_body(effective_from=start.isoformat(), effective_to=(start + timedelta(days=10)).isoformat()),
        headers=master_key_header,
    )
    assert first.status_code == status.HTTP_201_CREATED, first.text

    clash = client.post(
        _ENDPOINT,
        json=_body(
            input_price_per_million=9.0,
            output_price_per_million=9.0,
            effective_from=(start + timedelta(days=5)).isoformat(),
            effective_to=(start + timedelta(days=15)).isoformat(),
        ),
        headers=master_key_header,
    )

    assert clash.status_code == status.HTTP_409_CONFLICT, clash.text
    detail = clash.json()["detail"]
    assert _MODEL_KEY in detail
    # The message names the period it collided with, so the operator can act.
    assert "already covers" in detail

    # And nothing was stored: the refusal is not a partial write.
    assert client.get(_ENDPOINT, headers=master_key_header).json()["count"] == 1


def test_two_adjacent_periods_are_accepted(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """Touching is not overlapping, so retiring one rate into the next works."""
    boundary = datetime.now(UTC)
    first = client.post(
        _ENDPOINT,
        json=_body(
            effective_from=(boundary - timedelta(days=10)).isoformat(),
            effective_to=boundary.isoformat(),
        ),
        headers=master_key_header,
    )
    assert first.status_code == status.HTTP_201_CREATED, first.text

    second = client.post(
        _ENDPOINT,
        json=_body(input_price_per_million=3.5, effective_from=boundary.isoformat()),
        headers=master_key_header,
    )

    assert second.status_code == status.HTTP_201_CREATED, second.text
    assert client.get(_ENDPOINT, headers=master_key_header).json()["count"] == 2


def test_a_period_that_ends_before_it_starts_is_refused(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    now = datetime.now(UTC)
    response = client.post(
        _ENDPOINT,
        json=_body(effective_from=now.isoformat(), effective_to=(now - timedelta(days=1)).isoformat()),
        headers=master_key_header,
    )

    assert response.status_code == status.HTTP_400_BAD_REQUEST, response.text
    assert "effective_to" in response.json()["detail"]


def test_a_negative_rate_is_refused(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    response = client.post(_ENDPOINT, json=_body(input_price_per_million=-1.0), headers=master_key_header)

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT, response.text


def test_a_model_key_with_no_provider_prefix_is_refused(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """A bare model name would store a rate resolution could never match."""
    response = client.post(_ENDPOINT, json=_body(model_key="gpt-4o"), headers=master_key_header)

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT, response.text
    # And the prefixed form is accepted, so the rule is not just refusing everything.
    accepted = client.post(_ENDPOINT, json=_body(model_key="openai:gpt-4o"), headers=master_key_header)
    assert accepted.status_code == status.HTTP_201_CREATED, accepted.text


def test_a_replacement_without_a_start_is_refused(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """A replacement states the whole period, so the start is not defaulted.

    Without this the omitted field became ``now``, quietly moving a scheduled
    override into effect today.
    """
    created = client.post(
        _ENDPOINT,
        json=_body(effective_from=(datetime.now(UTC) + timedelta(days=30)).isoformat()),
        headers=master_key_header,
    )
    assert created.status_code == status.HTTP_201_CREATED, created.text

    response = client.put(
        f"{_ENDPOINT}/{created.json()['id']}",
        json={"input_price_per_million": 1.0, "output_price_per_million": 2.0},
        headers=master_key_header,
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT, response.text
    # The stored period did not move.
    listed = client.get(_ENDPOINT, headers=master_key_header).json()["data"]
    assert listed[0]["effective_from"] == created.json()["effective_from"]


def test_the_two_spellings_of_one_model_collapse_to_one_key(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """A slash-form key is normalized on write, so it cannot shadow the canonical one.

    Stored verbatim these were two rows: the overlap rule saw no collision,
    resolution preferred the canonical spelling so the other sat dormant, and
    deleting the canonical one silently promoted it to the live rate.
    """
    canonical = client.post(_ENDPOINT, json=_body(model_key="openai:gpt-4o"), headers=master_key_header)
    assert canonical.status_code == status.HTTP_201_CREATED, canonical.text

    slashed = client.post(
        _ENDPOINT,
        json=_body(model_key="openai/gpt-4o", input_price_per_million=99.0),
        headers=master_key_header,
    )

    # Same key after normalization, same open-ended period, so it is a conflict.
    assert slashed.status_code == status.HTTP_409_CONFLICT, slashed.text
    listed = client.get(_ENDPOINT, headers=master_key_header).json()
    assert listed["count"] == 1
    assert listed["data"][0]["model_key"] == "openai:gpt-4o"
    assert listed["data"][0]["input_price_per_million"] != 99.0


def test_a_slash_form_key_is_stored_canonically(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """The stored key is the one resolution looks for, whichever form was sent."""
    created = client.post(_ENDPOINT, json=_body(model_key="openai/gpt-4o"), headers=master_key_header)

    assert created.status_code == status.HTTP_201_CREATED, created.text
    assert created.json()["model_key"] == "openai:gpt-4o"


def test_repeated_tier_thresholds_are_refused(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """Two rates for one threshold is a question with no answer.

    The cost core resolves a tie by taking the first applicable entry, so which
    rate applied would depend on JSON array order. ``POST /api/v1/pricing`` and
    ``GatewayConfig`` already refuse it; this surface has to as well.
    """
    response = client.post(
        _ENDPOINT,
        json=_body(
            pricing_tiers=[
                {"min_input_tokens": 128000, "input_price_per_million": 5.0},
                {"min_input_tokens": 128000, "input_price_per_million": 9.0},
            ]
        ),
        headers=master_key_header,
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT, response.text
    assert "min_input_tokens" in response.text


def test_distinct_tier_thresholds_are_accepted(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    created = client.post(
        _ENDPOINT,
        json=_body(
            pricing_tiers=[
                {"min_input_tokens": 128000, "input_price_per_million": 5.0},
                {"min_input_tokens": 256000, "input_price_per_million": 9.0},
            ]
        ),
        headers=master_key_header,
    )

    assert created.status_code == status.HTTP_201_CREATED, created.text
    assert len(created.json()["pricing_tiers"]) == 2


def test_an_unknown_override_id_is_a_404(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    missing = uuid.uuid4()

    replaced = client.put(
        f"{_ENDPOINT}/{missing}",
        json={
            "input_price_per_million": 1.0,
            "output_price_per_million": 2.0,
            "effective_from": datetime.now(UTC).isoformat(),
        },
        headers=master_key_header,
    )
    removed = client.delete(f"{_ENDPOINT}/{missing}", headers=master_key_header)

    assert replaced.status_code == status.HTTP_404_NOT_FOUND, replaced.text
    assert removed.status_code == status.HTTP_404_NOT_FOUND, removed.text


def test_the_endpoints_require_the_master_key(client: TestClient) -> None:
    """Every verb, not only the reads.

    The writes are what an unauthenticated caller most wants, so a regression that
    dropped the dependency from just the PUT or DELETE route would pass a test
    that only exercised GET and POST.
    """
    unauthenticated = {status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN}
    # Any id: auth is refused before the route looks for the row, so this must not
    # be a 404.
    absent = f"{_ENDPOINT}/{uuid.uuid4()}"

    assert client.get(_ENDPOINT).status_code in unauthenticated
    assert client.post(_ENDPOINT, json=_body()).status_code in unauthenticated
    assert client.put(absent, json=_body()).status_code in unauthenticated
    assert client.delete(absent).status_code in unauthenticated


def test_the_list_is_paged_and_counts_the_whole_set(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """A growing table is read a page at a time, and the count is the total."""
    start = datetime.now(UTC) - timedelta(days=30)
    for index in range(3):
        created = client.post(
            _ENDPOINT,
            json=_body(
                model_key=f"openai:model-{index}",
                effective_from=(start + timedelta(days=index)).isoformat(),
            ),
            headers=master_key_header,
        )
        assert created.status_code == status.HTTP_201_CREATED, created.text

    first_page = client.get(f"{_ENDPOINT}?skip=0&limit=2", headers=master_key_header)
    second_page = client.get(f"{_ENDPOINT}?skip=2&limit=2", headers=master_key_header)

    assert len(first_page.json()["data"]) == 2
    assert len(second_page.json()["data"]) == 1
    # The total, not the page length, on both pages.
    assert first_page.json()["count"] == 3
    assert second_page.json()["count"] == 3
    # And no row is served twice or skipped.
    ids = [row["id"] for row in first_page.json()["data"] + second_page.json()["data"]]
    assert len(set(ids)) == 3


def test_the_list_refuses_a_limit_past_the_ceiling(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    over = client.get(f"{_ENDPOINT}?limit=1001", headers=master_key_header)
    under = client.get(f"{_ENDPOINT}?limit=0", headers=master_key_header)

    assert over.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT, over.text
    assert under.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT, under.text


def test_the_deployment_price_list_is_untouched_by_an_override(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """The two surfaces are separate: an override is not a deployment price."""
    client.post(
        f"{API_ROOT}/pricing",
        json={
            "model_key": _MODEL_KEY,
            "input_price_per_million": 10.0,
            "output_price_per_million": 20.0,
        },
        headers=master_key_header,
    )

    created = client.post(_ENDPOINT, json=_body(), headers=master_key_header)
    assert created.status_code == status.HTTP_201_CREATED, created.text

    deployment = client.get(f"{API_ROOT}/pricing/{_MODEL_KEY}", headers=master_key_header)
    assert deployment.status_code == status.HTTP_200_OK, deployment.text
    assert deployment.json()["input_price_per_million"] == 10.0


# =============================================================================
# The role gate
#
# Unreachable through the routes: a standalone deployment has one operator
# identity and it is an owner and a superuser, so the interesting cases are
# exercised at the service layer, as `test_tenancy_authorization.py` does.
# =============================================================================


async def _identity(db: AsyncSession, organization: Organization, *, role: str, name: str) -> User:
    user = await UserRepository(db).create_local_identity(
        full_name=name,
        active_organization_id=organization.id,
        is_superuser=False,
    )
    await OrganizationMemberRepository(db).create_membership(
        organization_id=organization.id,
        user_id=user.id,
        role=role,
    )
    return user


def _rates(**overrides: Any) -> PricingOverrideInput:
    fields: dict[str, Any] = {
        "input_price_per_million": 2.5,
        "output_price_per_million": 5.0,
        "cache_read_price_per_million": None,
        "cache_write_price_per_million": None,
        "cache_write_1h_price_per_million": None,
        "pricing_tiers": [],
        "effective_from": datetime.now(UTC),
        "effective_to": None,
    }
    fields.update(overrides)
    return PricingOverrideInput(**fields)


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["owner", "admin"])
async def test_a_management_role_may_write_an_override(async_db: AsyncSession, role: str) -> None:
    organization = await OrganizationRepository(async_db).create_organization(
        name="Acme", slug=f"acme-{role}", created_by_user_id=None
    )
    identity = await _identity(async_db, organization, role=role, name=f"{role} person")

    service = OrganizationPricingService(async_db, GatewayConfig(), model_provider=None)
    created = await service.create_for_caller(identity, _MODEL_KEY, _rates())

    assert created.organization_id == organization.id
    assert created.input_price_per_million == 2.5


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["member", "viewer"])
async def test_a_non_management_role_may_not_write_an_override(async_db: AsyncSession, role: str) -> None:
    """Rates decide what every member is billed, so writing is owner or admin."""
    organization = await OrganizationRepository(async_db).create_organization(
        name="Acme", slug=f"acme-{role}", created_by_user_id=None
    )
    identity = await _identity(async_db, organization, role=role, name=f"{role} person")
    service = OrganizationPricingService(async_db, GatewayConfig(), model_provider=None)

    with pytest.raises(NotAuthorizedError):
        await service.create_for_caller(identity, _MODEL_KEY, _rates())


# =============================================================================
# Whose model is it to re-price (otari-ai#2095)
# =============================================================================

_MANAGED_KEY = "nebius_prod:llama-3"
# One configured instance, so ``nebius_prod:llama-3`` dispatches on the
# deployment's own credential while ``openai:gpt-4o`` stays a bare provider key
# resolved against whatever the organization supplies.
_MANAGED_CONFIG = GatewayConfig(providers={"nebius_prod": {"provider_type": "nebius", "api_key": "x"}})


async def _operator(db: AsyncSession, organization: Organization) -> User:
    """An identity that also operates the deployment, which no role confers."""
    user = await UserRepository(db).create_local_identity(
        full_name="operator person",
        active_organization_id=organization.id,
        is_superuser=True,
    )
    await OrganizationMemberRepository(db).create_membership(
        organization_id=organization.id,
        user_id=user.id,
        role="owner",
    )
    return user


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["owner", "admin"])
async def test_an_organization_may_not_price_a_model_the_deployment_supplies(async_db: AsyncSession, role: str) -> None:
    """The deployment holds that credential, so it settles the bill and sets the rate.

    Without this an organization admin could store a zero for a model the
    deployment pays the upstream for, which makes it free and, because a budget
    counts down the cost, spends nothing either.
    """
    organization = await OrganizationRepository(async_db).create_organization(
        name="Acme", slug=f"acme-managed-{role}", created_by_user_id=None
    )
    identity = await _identity(async_db, organization, role=role, name=f"{role} person")
    service = OrganizationPricingService(async_db, _MANAGED_CONFIG, model_provider=None)

    with pytest.raises(OrganizationPricingManagedModelError) as refused:
        await service.create_for_caller(identity, _MANAGED_KEY, _rates(input_price_per_million=0.0))

    assert _MANAGED_KEY in str(refused.value)


@pytest.mark.asyncio
async def test_an_organization_may_still_price_a_model_it_supplies_the_key_for(async_db: AsyncSession) -> None:
    """A bare ``provider:model`` key resolves against the organization's own BYO credential."""
    organization = await OrganizationRepository(async_db).create_organization(
        name="Acme", slug="acme-byo", created_by_user_id=None
    )
    identity = await _identity(async_db, organization, role="admin", name="admin person")
    service = OrganizationPricingService(async_db, _MANAGED_CONFIG, model_provider=None)

    created = await service.create_for_caller(identity, _MODEL_KEY, _rates())

    assert created.model_key == _MODEL_KEY


@pytest.mark.asyncio
async def test_a_deployment_operator_may_price_a_model_the_deployment_supplies(async_db: AsyncSession) -> None:
    """The exemption that keeps a standalone deployment pricing its own models.

    There the one administrator is also the single organization's owner, so a
    blanket refusal would retire the override surface rather than protect anyone.
    """
    organization = await OrganizationRepository(async_db).create_organization(
        name="Acme", slug="acme-operator", created_by_user_id=None
    )
    identity = await _operator(async_db, organization)
    service = OrganizationPricingService(async_db, _MANAGED_CONFIG, model_provider=None)

    created = await service.create_for_caller(identity, _MANAGED_KEY, _rates())

    assert created.model_key == _MANAGED_KEY


@pytest.mark.asyncio
async def test_an_override_stored_before_the_rule_cannot_be_edited_by_an_organization(
    async_db: AsyncSession,
) -> None:
    """Rows already stored keep resolving; what they may not do is change rate.

    Nothing migrates the existing table, so an override an organization stored
    for a deployment-supplied model survives. Leaving the update path open would
    let it be edited into a rate nobody could create today, which is the same
    bypass by another door.
    """
    organization = await OrganizationRepository(async_db).create_organization(
        name="Acme", slug="acme-legacy", created_by_user_id=None
    )
    identity = await _identity(async_db, organization, role="admin", name="admin person")
    # Stored through a config that knows no instances, which is the deployment as
    # it was before the instance existed.
    stored = await OrganizationPricingService(async_db, GatewayConfig(), model_provider=None).create_for_caller(
        identity, _MANAGED_KEY, _rates()
    )

    service = OrganizationPricingService(async_db, _MANAGED_CONFIG, model_provider=None)
    with pytest.raises(OrganizationPricingManagedModelError):
        await service.replace_for_caller(
            identity,
            stored.id,
            _rates(input_price_per_million=0.0, effective_from=stored.effective_from),
        )


class _FakeHostedModelProvider:
    """A stub ``ModelProviderPort`` that serves or refuses one fixed provider."""

    def __init__(self, *, served: str | None = None, denied: str | None = None) -> None:
        """Name the one provider this stub serves, or the one it refuses; both default to neither."""
        self._served = served
        self._denied = denied

    async def resolve_hosted_credential(
        self,
        *,
        organization_id: uuid.UUID,
        workspace_id: uuid.UUID | None,
        provider: str,
        model: str | None,
    ) -> HostedCredential | None:
        """Answer exactly as configured, ignoring every argument but ``provider``."""
        del organization_id, workspace_id, model
        if provider == self._denied:
            raise HostedAccessDeniedError(f"{provider} is not enabled for this organization")
        if provider == self._served:
            return HostedCredential(api_key="x", api_base=None, response_provider=provider)
        return None

    async def get_hosted_providers(self, *, organization_id: uuid.UUID) -> frozenset[str]:
        del organization_id
        return frozenset({self._served} if self._served is not None else ())


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["owner", "admin"])
async def test_an_organization_may_not_price_a_model_a_hosted_credential_serves(
    async_db: AsyncSession, role: str
) -> None:
    """A bare key the port serves is refused when the organization has no BYO key."""
    organization = await OrganizationRepository(async_db).create_organization(
        name="Acme", slug=f"acme-hosted-{role}", created_by_user_id=None
    )
    identity = await _identity(async_db, organization, role=role, name=f"{role} person")
    service = OrganizationPricingService(
        async_db, GatewayConfig(), model_provider=_FakeHostedModelProvider(served="openai")
    )

    with pytest.raises(OrganizationPricingManagedModelError) as refused:
        await service.create_for_caller(identity, _MODEL_KEY, _rates(input_price_per_million=0.0))

    assert _MODEL_KEY in str(refused.value)


@pytest.mark.asyncio
async def test_an_organization_with_its_own_byo_key_may_still_price_it(async_db: AsyncSession) -> None:
    """An organization with its own BYO key may price the model even when the port also serves it."""
    organization = await OrganizationRepository(async_db).create_organization(
        name="Acme", slug="acme-hosted-byo", created_by_user_id=None
    )
    identity = await _identity(async_db, organization, role="admin", name="admin person")
    # A workspace with no override inherits the organization's key, so it stays covered.
    await WorkspaceRepository(async_db).create_workspace(
        name="Platform", organization_id=organization.id, created_by_user_id=None
    )
    await OrgProviderKeyRepository(async_db).create_key(
        organization_id=organization.id,
        provider="openai",
        name="prod",
        encrypted_api_key=None,
        last4=None,
        # A base URL with no API key still counts as BYO.
        api_base="https://openai.example.test/v1",
        client_args=None,
    )
    service = OrganizationPricingService(
        async_db, GatewayConfig(), model_provider=_FakeHostedModelProvider(served="openai")
    )

    created = await service.create_for_caller(identity, _MODEL_KEY, _rates())

    assert created.model_key == _MODEL_KEY


@pytest.mark.asyncio
async def test_a_workspace_that_disabled_its_only_byo_key_defeats_the_organizations_exemption(
    async_db: AsyncSession,
) -> None:
    """A workspace that disables the organization's only matching key removes the exemption."""
    organization = await OrganizationRepository(async_db).create_organization(
        name="Acme", slug="acme-hosted-disabled", created_by_user_id=None
    )
    identity = await _identity(async_db, organization, role="admin", name="admin person")
    workspace = await WorkspaceRepository(async_db).create_workspace(
        name="Platform", organization_id=organization.id, created_by_user_id=None
    )
    key = await OrgProviderKeyRepository(async_db).create_key(
        organization_id=organization.id,
        provider="openai",
        name="prod",
        encrypted_api_key=None,
        last4=None,
        api_base="https://openai.example.test/v1",
        client_args=None,
    )
    await WorkspaceProviderKeyOverrideRepository(async_db).create(
        workspace_id=workspace.id,
        organization_id=organization.id,
        org_provider_key_id=key.id,
        is_default=False,
        disabled=True,
    )
    service = OrganizationPricingService(
        async_db, GatewayConfig(), model_provider=_FakeHostedModelProvider(served="openai")
    )

    with pytest.raises(OrganizationPricingManagedModelError):
        await service.create_for_caller(identity, _MODEL_KEY, _rates())


@pytest.mark.asyncio
async def test_a_key_row_with_no_credential_material_does_not_exempt_the_organization(
    async_db: AsyncSession,
) -> None:
    """A key row with neither an API key nor a base URL cannot serve a request, so it does not count."""
    organization = await OrganizationRepository(async_db).create_organization(
        name="Acme", slug="acme-hosted-decoy", created_by_user_id=None
    )
    identity = await _identity(async_db, organization, role="admin", name="admin person")
    await OrgProviderKeyRepository(async_db).create_key(
        organization_id=organization.id,
        provider="openai",
        name="decoy",
        encrypted_api_key=None,
        last4=None,
        api_base=None,
        client_args=None,
    )
    service = OrganizationPricingService(
        async_db, GatewayConfig(), model_provider=_FakeHostedModelProvider(served="openai")
    )

    with pytest.raises(OrganizationPricingManagedModelError):
        await service.create_for_caller(identity, _MODEL_KEY, _rates())


@pytest.mark.asyncio
async def test_an_unusable_default_key_does_not_fall_back_to_a_usable_one(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dispatch uses the default key or none, so a usable second key does not exempt the organization."""
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    organization = await OrganizationRepository(async_db).create_organization(
        name="Acme", slug="acme-hosted-bad-default", created_by_user_id=None
    )
    identity = await _identity(async_db, organization, role="admin", name="admin person")
    workspace = await WorkspaceRepository(async_db).create_workspace(
        name="Platform", organization_id=organization.id, created_by_user_id=None
    )
    keys = OrgProviderKeyRepository(async_db)
    usable = await keys.create_key(
        organization_id=organization.id,
        provider="openai",
        name="usable",
        encrypted_api_key=encrypt_secret("sk-usable"),
        last4="able",
        api_base=None,
        client_args=None,
    )
    unusable = await keys.create_key(
        organization_id=organization.id,
        provider="openai",
        name="unusable",
        encrypted_api_key="not-a-ciphertext",
        last4=None,
        api_base=None,
        client_args=None,
    )

    async def dispatch_falls_through_to_the_port() -> bool:
        await refresh_org_provider_cache(async_db)
        kwargs = get_provider_kwargs(GatewayConfig(), LLMProvider.OPENAI, workspace_id=workspace.id)
        return credential_ladder_exhausted(LLMProvider.OPENAI, kwargs)

    try:
        usable.is_org_default = True
        await async_db.flush()
        assert not await dispatch_falls_through_to_the_port()

        usable.is_org_default = False
        await async_db.flush()
        unusable.is_org_default = True
        await async_db.flush()
        assert await dispatch_falls_through_to_the_port()
    finally:
        reset_org_provider_cache()

    service = OrganizationPricingService(
        async_db, GatewayConfig(), model_provider=_FakeHostedModelProvider(served="openai")
    )

    with pytest.raises(OrganizationPricingManagedModelError):
        await service.create_for_caller(identity, _MODEL_KEY, _rates())


@pytest.mark.asyncio
async def test_a_hosted_access_refusal_still_counts_as_deployment_supplied(async_db: AsyncSession) -> None:
    """A port refusal counts as deployment-supplied, because the model still runs on a deployment-owned upstream."""
    organization = await OrganizationRepository(async_db).create_organization(
        name="Acme", slug="acme-hosted-denied", created_by_user_id=None
    )
    identity = await _identity(async_db, organization, role="admin", name="admin person")
    service = OrganizationPricingService(
        async_db, GatewayConfig(), model_provider=_FakeHostedModelProvider(denied="openai")
    )

    with pytest.raises(OrganizationPricingManagedModelError):
        await service.create_for_caller(identity, _MODEL_KEY, _rates())


@pytest.mark.asyncio
async def test_a_deployment_operator_may_price_a_model_a_hosted_credential_serves(async_db: AsyncSession) -> None:
    """The operator exemption also covers a model the port serves."""
    organization = await OrganizationRepository(async_db).create_organization(
        name="Acme", slug="acme-hosted-operator", created_by_user_id=None
    )
    identity = await _operator(async_db, organization)
    service = OrganizationPricingService(
        async_db, GatewayConfig(), model_provider=_FakeHostedModelProvider(served="openai")
    )

    created = await service.create_for_caller(identity, _MODEL_KEY, _rates())

    assert created.model_key == _MODEL_KEY


@pytest.mark.asyncio
async def test_an_organization_may_price_a_model_no_hosted_credential_serves(async_db: AsyncSession) -> None:
    """A port that does not serve the provider leaves the organization free to price it."""
    organization = await OrganizationRepository(async_db).create_organization(
        name="Acme", slug="acme-hosted-unserved", created_by_user_id=None
    )
    identity = await _identity(async_db, organization, role="admin", name="admin person")
    service = OrganizationPricingService(
        async_db, GatewayConfig(), model_provider=_FakeHostedModelProvider(served="anthropic")
    )

    created = await service.create_for_caller(identity, _MODEL_KEY, _rates())

    assert created.model_key == _MODEL_KEY


def test_the_route_asks_the_port_this_build_bound(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    """The route builds the service with the container's bound port.

    Gotcha: the master key acts as the deployment operator, who is exempt, so the caller is an admin session.
    """
    assert client.get(f"{API_ROOT}/organizations/me", headers=master_key_header).status_code == status.HTTP_200_OK
    container: Any = client.app.state.container  # type: ignore[attr-defined]
    container.bind(ModelProviderPort, lambda session: _FakeHostedModelProvider(served="openai"))

    token = "otari-sess-admin@tenant.test"
    session = db_session_factory()
    try:
        organization = Organization(name="Tenant", slug="tenant-hosted-route")
        session.add(organization)
        session.commit()
        session.refresh(organization)
        admin = User(email="admin@tenant.test", full_name="Admin", active_organization_id=organization.id)
        session.add(admin)
        session.commit()
        session.refresh(admin)
        session.add(
            OrganizationMember(organization_id=organization.id, user_id=admin.id, role="admin", status="active")
        )
        session.add(
            DashboardSession(
                token_hash=hash_session_token(token),
                user_id=admin.id,
                created_at=datetime.now(UTC),
                expires_at=datetime.now(UTC) + timedelta(hours=12),
            )
        )
        session.commit()
    finally:
        session.close()

    client.cookies.set(SESSION_COOKIE_NAME, token)
    try:
        unserved = client.post(_ENDPOINT, json=_body(model_key="anthropic:claude-3-5-haiku-latest"))
        served = client.post(_ENDPOINT, json=_body())
    finally:
        client.cookies.clear()

    assert unserved.status_code == status.HTTP_201_CREATED, unserved.text
    assert served.status_code == status.HTTP_403_FORBIDDEN, served.text
    assert _MODEL_KEY in served.json()["detail"]


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["member", "viewer"])
async def test_any_member_may_read_the_overrides(async_db: AsyncSession, role: str) -> None:
    """A member is billed at these rates, so they are not withheld from them."""
    organization = await OrganizationRepository(async_db).create_organization(
        name="Acme", slug=f"acme-read-{role}", created_by_user_id=None
    )
    owner = await _identity(async_db, organization, role="owner", name="owner person")
    reader = await _identity(async_db, organization, role=role, name=f"{role} reader")
    service = OrganizationPricingService(async_db, GatewayConfig(), model_provider=None)
    await service.create_for_caller(owner, _MODEL_KEY, _rates())

    visible, total = await service.list_for_caller(reader)

    assert [row.model_key for row in visible] == [_MODEL_KEY]
    assert total == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field",
    [
        "input_price_per_million",
        "output_price_per_million",
        "cache_read_price_per_million",
        "cache_write_price_per_million",
        "cache_write_1h_price_per_million",
    ],
)
async def test_a_negative_rate_is_refused_at_the_service_boundary(async_db: AsyncSession, field: str) -> None:
    """Named, and a 400, rather than an IntegrityError at flush.

    The route bounds all five with ``Field(ge=0)``, so nothing over HTTP arrives
    negative; this is the boundary a direct caller crosses.
    """
    organization = await OrganizationRepository(async_db).create_organization(
        name="Acme", slug=f"acme-negative-{field}", created_by_user_id=None
    )
    owner = await _identity(async_db, organization, role="owner", name="owner person")
    service = OrganizationPricingService(async_db, GatewayConfig(), model_provider=None)

    with pytest.raises(TenancyValidationError) as caught:
        await service.create_for_caller(owner, _MODEL_KEY, _rates(**{field: -1.0}))

    assert field in caught.value.message
    assert caught.value.status_code == 400


@pytest.mark.asyncio
async def test_another_organizations_override_is_a_404_not_a_403(async_db: AsyncSession) -> None:
    """A 404 rather than a 403, so the id space is not an existence oracle."""
    theirs = await OrganizationRepository(async_db).create_organization(
        name="Theirs", slug="theirs", created_by_user_id=None
    )
    mine = await OrganizationRepository(async_db).create_organization(name="Mine", slug="mine", created_by_user_id=None)
    their_owner = await _identity(async_db, theirs, role="owner", name="their owner")
    my_owner = await _identity(async_db, mine, role="owner", name="my owner")
    service = OrganizationPricingService(async_db, GatewayConfig(), model_provider=None)
    their_override = await service.create_for_caller(their_owner, _MODEL_KEY, _rates())

    with pytest.raises(OrganizationPricingNotFoundError):
        await service.replace_for_caller(my_owner, their_override.id, _rates())
    with pytest.raises(OrganizationPricingNotFoundError):
        await service.delete_for_caller(my_owner, their_override.id)


@pytest.mark.asyncio
async def test_the_overlap_rule_is_scoped_to_one_organization(async_db: AsyncSession) -> None:
    """Two organizations may price the same model over the same period."""
    first = await OrganizationRepository(async_db).create_organization(
        name="First", slug="first", created_by_user_id=None
    )
    second = await OrganizationRepository(async_db).create_organization(
        name="Second", slug="second", created_by_user_id=None
    )
    first_owner = await _identity(async_db, first, role="owner", name="first owner")
    second_owner = await _identity(async_db, second, role="owner", name="second owner")
    service = OrganizationPricingService(async_db, GatewayConfig(), model_provider=None)
    period = _rates(effective_from=datetime.now(UTC) - timedelta(days=1))

    await service.create_for_caller(first_owner, _MODEL_KEY, period)
    also = await service.create_for_caller(second_owner, _MODEL_KEY, period)

    assert also.organization_id == second.id


@pytest.mark.asyncio
async def test_a_second_overlapping_period_is_refused_for_one_organization(
    async_db: AsyncSession,
) -> None:
    organization = await OrganizationRepository(async_db).create_organization(
        name="Acme", slug="acme-overlap", created_by_user_id=None
    )
    owner = await _identity(async_db, organization, role="owner", name="owner person")
    service = OrganizationPricingService(async_db, GatewayConfig(), model_provider=None)
    await service.create_for_caller(owner, _MODEL_KEY, _rates(effective_from=datetime.now(UTC) - timedelta(days=1)))

    with pytest.raises(OrganizationPricingOverlapError):
        await service.create_for_caller(owner, _MODEL_KEY, _rates())


# =============================================================================
# What a request settles at
# =============================================================================


@pytest.mark.asyncio
async def test_a_keys_request_resolves_its_organizations_override(async_db: AsyncSession) -> None:
    """The definition of done's first line, through the resolution a request uses.

    Built from the key outward, exactly as the request path does: the key names a
    workspace, the workspace names the organization, and the organization's rate
    is what prices the request. Nothing here reads a header.
    """
    organization = await OrganizationRepository(async_db).create_organization(
        name="Acme", slug="acme-settles", created_by_user_id=None
    )
    workspace = await WorkspaceRepository(async_db).create_workspace(
        name="Platform", organization_id=organization.id, created_by_user_id=None
    )
    async_db.add(
        ModelPricing(
            model_key=_MODEL_KEY,
            effective_at=datetime.now(UTC) - timedelta(days=30),
            input_price_per_million=10.0,
            output_price_per_million=20.0,
        )
    )
    async_db.add(
        OrganizationModelPricing(
            organization_id=organization.id,
            model_key=_MODEL_KEY,
            input_price_per_million=2.5,
            output_price_per_million=5.0,
            effective_from=datetime.now(UTC) - timedelta(days=1),
            pricing_tiers=[],
        )
    )
    key = APIKey(id="key-settles", key_hash="hash-settles", workspace_id=workspace.id)
    async_db.add(key)
    await async_db.commit()

    resolved_organization = await organization_for_key_id(async_db, key.id)
    pricing = await find_model_pricing(async_db, "openai", "gpt-4o", organization_id=resolved_organization)

    assert resolved_organization == organization.id
    assert pricing is not None
    assert pricing.input_price_per_million == 2.5, "the request must price at the organization's rate"

    # And a key in another organization's workspace is unaffected.
    other = await OrganizationRepository(async_db).create_organization(
        name="Other", slug="other-settles", created_by_user_id=None
    )
    other_workspace = await WorkspaceRepository(async_db).create_workspace(
        name="Other team", organization_id=other.id, created_by_user_id=None
    )
    other_key = APIKey(id="key-other", key_hash="hash-other", workspace_id=other_workspace.id)
    async_db.add(other_key)
    await async_db.commit()

    other_pricing = await find_model_pricing(
        async_db, "openai", "gpt-4o", organization_id=await organization_for_key_id(async_db, other_key.id)
    )
    assert other_pricing is not None
    assert other_pricing.input_price_per_million == 10.0


@pytest.mark.asyncio
async def test_deleting_an_override_returns_the_model_to_the_deployment_list(
    async_db: AsyncSession,
) -> None:
    organization = await OrganizationRepository(async_db).create_organization(
        name="Acme", slug="acme-delete", created_by_user_id=None
    )
    owner = await _identity(async_db, organization, role="owner", name="owner person")
    async_db.add(
        ModelPricing(
            model_key=_MODEL_KEY,
            effective_at=datetime.now(UTC) - timedelta(days=30),
            input_price_per_million=10.0,
            output_price_per_million=20.0,
        )
    )
    service = OrganizationPricingService(async_db, GatewayConfig(), model_provider=None)
    override = await service.create_for_caller(owner, _MODEL_KEY, _rates())
    await async_db.commit()

    await service.delete_for_caller(owner, override.id)
    await async_db.commit()

    pricing = await find_model_pricing(async_db, "openai", "gpt-4o", organization_id=organization.id)
    assert pricing is not None
    assert pricing.input_price_per_million == 10.0

    remaining = (
        (
            await async_db.execute(
                select(OrganizationModelPricing).where(OrganizationModelPricing.organization_id == organization.id)
            )
        )
        .scalars()
        .all()
    )
    assert remaining == []


@pytest.mark.asyncio
async def test_deleting_an_organization_takes_its_overrides(async_db: AsyncSession) -> None:
    """CASCADE, unlike the request-plane tables, which are RESTRICT.

    An override is configuration, and settled usage keeps its own cost, so
    nothing accounting-shaped is lost with it.
    """
    organization = await OrganizationRepository(async_db).create_organization(
        name="Acme", slug="acme-cascade", created_by_user_id=None
    )
    async_db.add(
        OrganizationModelPricing(
            organization_id=organization.id,
            model_key=_MODEL_KEY,
            input_price_per_million=2.5,
            output_price_per_million=5.0,
            effective_from=datetime.now(UTC),
            pricing_tiers=[],
        )
    )
    await async_db.commit()

    await async_db.delete(await async_db.get(Organization, organization.id))
    await async_db.commit()

    remaining = (
        (
            await async_db.execute(
                select(OrganizationModelPricing).where(OrganizationModelPricing.organization_id == organization.id)
            )
        )
        .scalars()
        .all()
    )
    assert remaining == []


@pytest.mark.asyncio
async def test_a_workspace_resolves_to_its_organization_once_and_stays_cached(
    async_db: AsyncSession,
) -> None:
    """The memo is what keeps the per-lookup cost off the request path."""
    organization = await OrganizationRepository(async_db).create_organization(
        name="Acme", slug="acme-cache", created_by_user_id=None
    )
    workspace = await WorkspaceRepository(async_db).create_workspace(
        name="Platform", organization_id=organization.id, created_by_user_id=None
    )
    await async_db.commit()

    first = await organization_for_workspace_id(async_db, workspace.id)
    # Delete the workspace's row out from under the cache; a second call that
    # queried again would now return None.
    await async_db.execute(delete(Workspace).where(col(Workspace.id) == workspace.id))
    await async_db.commit()
    second = await organization_for_workspace_id(async_db, workspace.id)

    assert first == organization.id
    assert second == organization.id


@pytest.mark.asyncio
async def test_an_unknown_workspace_resolves_to_no_organization(async_db: AsyncSession) -> None:
    """None rather than raising: a missing workspace must not fail a priced request."""
    assert await organization_for_workspace_id(async_db, uuid.uuid4()) is None


@pytest.mark.asyncio
async def test_a_racing_duplicate_period_is_a_conflict_not_an_integrity_error(
    async_db: AsyncSession,
) -> None:
    """The unique index refuses the second writer, and that is a 409's business.

    The service's overlap check is enough single-threaded, so this simulates the
    race by disabling it: two writers that both passed the check reach ``flush``,
    where the index refuses the second. That refusal arrives before the route's
    ``commit``, so without mapping it at the flush boundary it escaped as a 500
    and told the caller to retry what was really a conflict.
    """
    organization = await OrganizationRepository(async_db).create_organization(
        name="Acme", slug="acme-race", created_by_user_id=None
    )
    owner = await _identity(async_db, organization, role="owner", name="owner person")
    service = OrganizationPricingService(async_db, GatewayConfig(), model_provider=None)
    rates = _rates()

    await service.create_for_caller(owner, _MODEL_KEY, rates)
    await async_db.commit()

    async def _no_overlap_check(**_kwargs: object) -> None:
        return None

    # Stand in for a second writer whose check passed concurrently with the first.
    service.raise_if_overlapping = _no_overlap_check  # type: ignore[method-assign]

    with pytest.raises(OrganizationPricingOverlapError) as caught:
        await service.create_for_caller(owner, _MODEL_KEY, rates)

    assert caught.value.status_code == 409
    assert _MODEL_KEY in caught.value.message


@pytest.mark.asyncio
async def test_the_race_conflict_names_the_period_it_actually_hit(
    async_db: AsyncSession,
) -> None:
    """The 409 describes the stored row's period, not the candidate's start.

    The first version fabricated an open end from the candidate's own start and
    passed it as ``existing_period``, so the message told the operator about a
    period that was not the one in the way.
    """
    organization = await OrganizationRepository(async_db).create_organization(
        name="Acme", slug="acme-race-message", created_by_user_id=None
    )
    owner = await _identity(async_db, organization, role="owner", name="owner person")
    service = OrganizationPricingService(async_db, GatewayConfig(), model_provider=None)
    start = datetime.now(UTC)
    bounded = _rates(effective_from=start, effective_to=start + timedelta(days=1))

    await service.create_for_caller(owner, _MODEL_KEY, bounded)
    await async_db.commit()

    async def _no_overlap_check(**_kwargs: object) -> None:
        return None

    service.raise_if_overlapping = _no_overlap_check  # type: ignore[method-assign]

    with pytest.raises(OrganizationPricingOverlapError) as caught:
        await service.create_for_caller(owner, _MODEL_KEY, bounded)

    # The stored row is bounded, so the message must not claim "onwards".
    assert "onwards" not in caught.value.message
    assert bounded.effective_to is not None
    assert bounded.effective_to.isoformat() in caught.value.message


@pytest.mark.asyncio
async def test_a_check_violation_is_not_reported_as_a_conflict(
    async_db: AsyncSession,
) -> None:
    """Only the unique-index race is a conflict; other integrity failures are not.

    A broad ``except IntegrityError`` that always mapped to 409 told the caller
    another writer had taken the period when nothing of the kind had happened.
    The row is read back to prove which failure it was.
    """
    organization = await OrganizationRepository(async_db).create_organization(
        name="Acme", slug="acme-check", created_by_user_id=None
    )
    owner = await _identity(async_db, organization, role="owner", name="owner person")
    service = OrganizationPricingService(async_db, GatewayConfig(), model_provider=None)

    # The service's own rate validation would refuse this first, so bypass it to
    # reach the table's CHECK, which is the constraint under test.
    def _no_rate_check(_override: PricingOverrideInput) -> None:
        return None

    import gateway.services.organization_pricing_service as module

    original = module.validate_rates
    module.validate_rates = _no_rate_check  # type: ignore[assignment]
    try:
        with pytest.raises(IntegrityError):
            await service.create_for_caller(owner, _MODEL_KEY, _rates(input_price_per_million=-1.0))
    finally:
        module.validate_rates = original


@pytest.mark.asyncio
async def test_an_update_that_fails_for_another_reason_is_not_reported_as_a_conflict(
    async_db: AsyncSession,
) -> None:
    """The replace path must not find itself in the post-rollback read-back.

    The rollback restores the row being rewritten to its stored period, so an
    update that keeps that period and fails the flush for some other reason (here
    the table's rate CHECK) matched itself and returned a 409 naming the caller's
    own override. Same reachability as the create-path case above; the difference
    is that on this path a row legitimately occupies the period already.
    """
    organization = await OrganizationRepository(async_db).create_organization(
        name="Acme", slug="acme-update-check", created_by_user_id=None
    )
    owner = await _identity(async_db, organization, role="owner", name="owner person")
    service = OrganizationPricingService(async_db, GatewayConfig(), model_provider=None)
    period = _rates()
    stored = await service.create_for_caller(owner, _MODEL_KEY, period)
    await async_db.commit()
    stored_id = stored.id

    def _no_rate_check(_override: PricingOverrideInput) -> None:
        return None

    import gateway.services.organization_pricing_service as module

    original = module.validate_rates
    module.validate_rates = _no_rate_check  # type: ignore[assignment]
    try:
        # Same period, so the read-back would match the row being rewritten; the
        # negative rate is what actually fails the flush.
        with pytest.raises(IntegrityError):
            await service.replace_for_caller(
                owner,
                stored_id,
                _rates(input_price_per_million=-1.0, effective_from=period.effective_from),
            )
    finally:
        module.validate_rates = original
