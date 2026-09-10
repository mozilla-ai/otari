"""Tests for the grouped catalog under /v1/catalog.

The models are priced rather than discovered, so the merge is deterministic
without dialing a provider, and models.dev is a mocked fetch: what is under
test is the fold and the join, not the sources.
"""

import uuid
from collections.abc import Callable, Generator
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.core.config import GatewayConfig
from gateway.models.entities import DashboardSession, OrganizationModelPricing
from gateway.models.tenancy import Organization, OrganizationMember, User
from gateway.services import model_catalog_service as mcs
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, hash_session_token

from .conftest import build_test_client

# Two providers serving one model under two spellings, one of them Fireworks'
# ``p``-for-point and path prefix, plus a second model on one of them.
_NEBIUS_GLM = "nebius:zai-org/GLM-5.3"
_FIREWORKS_GLM = "fireworks:accounts/fireworks/models/glm-5p3"
_NEBIUS_KIMI = "nebius:moonshotai/Kimi-K2.6"

CATALOG: dict[str, Any] = {
    "nebius": {
        "id": "nebius",
        "name": "Nebius",
        "models": {
            "zai-org/GLM-5.3": {
                "id": "zai-org/GLM-5.3",
                "name": "GLM-5.3",
                "family": "glm",
                "description": "Z.ai's flagship, from Nebius.",
                "reasoning": True,
                "tool_call": True,
                "modalities": {"input": ["text"], "output": ["text"]},
                "limit": {"context": 200000, "output": 128000},
                "release_date": "2026-07-01",
                "open_weights": True,
            },
            "moonshotai/Kimi-K2.6": {
                "id": "moonshotai/Kimi-K2.6",
                "name": "Kimi K2.6",
                "tool_call": True,
                "modalities": {"input": ["text", "image"], "output": ["text"]},
                "limit": {"context": 262144, "output": 32768},
            },
        },
    },
    # models.dev spells the provider ``fireworks-ai`` where any-llm says
    # ``fireworks``; the join maps one onto the other.
    "fireworks-ai": {
        "id": "fireworks-ai",
        "name": "Fireworks",
        "models": {
            "accounts/fireworks/models/glm-5p3": {
                "id": "accounts/fireworks/models/glm-5p3",
                "name": "GLM 5.3",
                "family": "glm",
                "description": "GLM 5.3 on Fireworks.",
                "reasoning": True,
                "tool_call": True,
                "structured_output": True,
                "modalities": {"input": ["text"], "output": ["text"]},
                "limit": {"context": 131072, "output": 16384},
                "release_date": "2026-07-01",
            },
        },
    },
    # A provider nobody here has configured, serving the same model.
    "groq": {
        "id": "groq",
        "name": "Groq",
        "models": {
            "glm-5.3": {"id": "glm-5.3", "name": "GLM-5.3", "modalities": {"input": ["text"], "output": ["text"]}},
        },
    },
}


def _config(postgres_url: str, **overrides: Any) -> GatewayConfig:
    return GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        require_pricing=False,
        # Discovery off: the catalog is the priced rows, and no provider is dialed.
        model_discovery=False,
        providers={"nebius": {"api_key": "sk-fake"}, "fireworks": {"api_key": "sk-fake"}},
        **overrides,
    )


@pytest.fixture
def catalog_client(postgres_url: str, clean_database: None) -> Generator[TestClient]:
    mcs.clear_catalog_cache()
    try:
        yield from build_test_client(_config(postgres_url))
    finally:
        mcs.clear_catalog_cache()


@pytest.fixture
def master_header() -> dict[str, str]:
    return {"Authorization": "Bearer test-master-key"}


def _price(client: TestClient, header: dict[str, str], key: str, input_rate: float, output_rate: float) -> None:
    response = client.post(
        "/v1/pricing",
        json={"model_key": key, "input_price_per_million": input_rate, "output_price_per_million": output_rate},
        headers=header,
    )
    assert response.status_code == status.HTTP_200_OK, response.text


@pytest.fixture
def priced(catalog_client: TestClient, master_header: dict[str, str]) -> TestClient:
    _price(catalog_client, master_header, _NEBIUS_GLM, 0.5, 2.0)
    _price(catalog_client, master_header, _FIREWORKS_GLM, 0.7, 2.5)
    _price(catalog_client, master_header, _NEBIUS_KIMI, 0.6, 2.4)
    return catalog_client


def _get(client: TestClient, path: str, **kwargs: Any) -> Any:
    with patch.object(mcs, "_fetch", new=AsyncMock(return_value=CATALOG)):
        response = client.get(path, **kwargs)
    assert response.status_code == status.HTTP_200_OK, response.text
    return response.json()


def test_the_catalog_folds_two_spellings_into_one_model(priced: TestClient, master_header: dict[str, str]) -> None:
    body = _get(priced, "/v1/catalog/models", headers=master_header)

    assert body["metadata_available"] is True
    assert body["defaults_as_of"] is None
    by_id = {model["id"]: model for model in body["models"]}
    assert set(by_id) == {"glm-5-3", "kimi-k2-6"}

    glm = by_id["glm-5-3"]
    # Two spellings, one each: the tie goes to the first selector in order,
    # which is Fireworks'. The vote is decided in the unit tests; this pins that
    # the fold happened and the vendor came from the model, not the provider.
    assert glm["name"] == "GLM 5.3"
    assert glm["vendor"] == "Z.ai"
    assert glm["offering_count"] == 2
    assert glm["provider_count"] == 2
    # The list price is the cheapest offering's; the limits are the largest.
    assert glm["min_input_price_per_million"] == 0.5
    assert glm["context_window"] == 200000
    assert glm["max_output_tokens"] == 128000
    # A capability one offering reports is the model's.
    assert glm["capabilities"]["structured_output"] is True
    assert glm["open_weights"] is True


def test_the_detail_lists_every_offering_cheapest_first(priced: TestClient, master_header: dict[str, str]) -> None:
    body = _get(priced, "/v1/catalog/models/glm-5-3", headers=master_header)

    # The description comes from the offering whose spelling named the model.
    assert body["description"] == "GLM 5.3 on Fireworks."
    assert [offering["selector"] for offering in body["offerings"]] == [_NEBIUS_GLM, _FIREWORKS_GLM]
    nebius, fireworks = body["offerings"]
    assert nebius["provider"] == "nebius"
    assert nebius["provider_type"] == "nebius"
    assert nebius["credential"] == "deployment"
    assert nebius["discovered"] is False
    assert nebius["price_source"] == "deployment"
    assert nebius["price_reference"] == _NEBIUS_GLM
    assert nebius["pricing"]["input_price_per_million"] == 0.5
    assert nebius["pricing"]["unit"] == "tokens"
    assert fireworks["context_window"] == 131072
    assert fireworks["max_output_tokens"] == 16384
    # Groq serves it too and is not configured here.
    assert body["also_available_from"] == [{"provider_type": "groq", "name": "Groq"}]


def test_an_unknown_model_is_404(priced: TestClient, master_header: dict[str, str]) -> None:
    with patch.object(mcs, "_fetch", new=AsyncMock(return_value=CATALOG)):
        response = priced.get("/v1/catalog/models/no-such-model", headers=master_header)
    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_without_metadata_the_catalog_still_groups_by_the_id(priced: TestClient, master_header: dict[str, str]) -> None:
    with patch.object(mcs, "_fetch", new=AsyncMock(return_value=None)):
        response = priced.get("/v1/catalog/models", headers=master_header)
    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body["metadata_available"] is False
    by_id = {model["id"]: model for model in body["models"]}
    # ``glm-5p3`` and ``GLM-5.3`` still meet through the id rung.
    assert by_id["glm-5-3"]["offering_count"] == 2
    assert by_id["glm-5-3"]["description" if "description" in by_id["glm-5-3"] else "name"]


def test_the_catalog_requires_a_credential(catalog_client: TestClient) -> None:
    response = catalog_client.get("/v1/catalog/models")
    assert response.status_code in (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN)


def test_an_api_key_sees_only_the_models_its_allow_list_permits(
    priced: TestClient, master_header: dict[str, str]
) -> None:
    created = priced.post(
        "/v1/keys",
        json={"key_name": "narrow", "allowed_models": [_NEBIUS_KIMI]},
        headers=master_header,
    )
    assert created.status_code == status.HTTP_200_OK, created.text
    key_header = {"Authorization": f"Bearer {created.json()['key']}"}

    body = _get(priced, "/v1/catalog/models", headers=key_header)
    assert [model["id"] for model in body["models"]] == ["kimi-k2-6"]

    with patch.object(mcs, "_fetch", new=AsyncMock(return_value=CATALOG)):
        denied = priced.get("/v1/catalog/models/glm-5-3", headers=key_header)
    # Indistinguishable from a model that does not exist.
    assert denied.status_code == status.HTTP_404_NOT_FOUND


def test_a_session_is_priced_at_its_organizations_override(
    priced: TestClient, master_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    """The one thing the flat listing gets wrong for a tenant.

    ``GET /v1/models`` prices from the deployment list; a member of an
    organization holding a negotiated rate is billed at that rate, and the
    catalog has to say so.
    """
    assert priced.get("/v1/organizations/me", headers=master_header).status_code == status.HTTP_200_OK
    session = db_session_factory()
    try:
        organization = Organization(name="Acme", slug="acme")
        session.add(organization)
        session.commit()
        session.refresh(organization)
        user = User(email="admin@acme.test", full_name="Admin", active_organization_id=organization.id)
        session.add(user)
        session.commit()
        session.refresh(user)
        session.add(OrganizationMember(organization_id=organization.id, user_id=user.id, role="admin", status="active"))
        token = "otari-sess-admin"
        session.add(
            DashboardSession(
                token_hash=hash_session_token(token),
                user_id=user.id,
                created_at=datetime.now(UTC),
                expires_at=datetime.now(UTC) + timedelta(hours=1),
            )
        )
        session.add(
            OrganizationModelPricing(
                id=uuid.uuid4(),
                organization_id=organization.id,
                model_key=_NEBIUS_GLM,
                input_price_per_million=0.1,
                output_price_per_million=0.4,
                pricing_tiers=[],
                effective_from=datetime.now(UTC) - timedelta(days=1),
                effective_to=None,
                unit="tokens",
                origin="api",
            )
        )
        session.commit()
    finally:
        session.close()

    priced.cookies.set(SESSION_COOKIE_NAME, token)
    try:
        body = _get(priced, "/v1/catalog/models/glm-5-3")
    finally:
        priced.cookies.clear()

    # An admin of an organization with no BYO key sees the configured instances,
    # and the Nebius row at the organization's own rate.
    by_selector = {offering["selector"]: offering for offering in body["offerings"]}
    assert by_selector[_NEBIUS_GLM]["price_source"] == "organization"
    assert by_selector[_NEBIUS_GLM]["pricing"]["input_price_per_million"] == 0.1
    assert by_selector[_FIREWORKS_GLM]["price_source"] == "deployment"
    # And the override is now the cheapest, so it leads.
    assert body["offerings"][0]["selector"] == _NEBIUS_GLM
    assert body["min_input_price_per_million"] == 0.1

    # The master key is not that organization, and keeps the deployment rate.
    operator = _get(priced, "/v1/catalog/models/glm-5-3", headers=master_header)
    assert {o["selector"]: o["price_source"] for o in operator["offerings"]} == {
        _NEBIUS_GLM: "deployment",
        _FIREWORKS_GLM: "deployment",
    }


# ---------------------------------------------------------------------------
# The public catalog
# ---------------------------------------------------------------------------


@pytest.fixture
def public_client(postgres_url: str, clean_database: None) -> Generator[TestClient]:
    mcs.clear_catalog_cache()
    try:
        yield from build_test_client(_config(postgres_url, public_catalog=True))
    finally:
        mcs.clear_catalog_cache()


def test_the_list_carries_what_the_filters_and_the_search_need(
    priced: TestClient, master_header: dict[str, str]
) -> None:
    body = _get(priced, "/v1/catalog/models", headers=master_header)
    glm = next(model for model in body["models"] if model["id"] == "glm-5-3")
    assert glm["selectors"] == sorted([_FIREWORKS_GLM, _NEBIUS_GLM])
    assert glm["price_sources"] == ["deployment"]
    assert glm["unpriced_count"] == 0
    assert isinstance(glm["discovered"], bool)


def test_prices_compare_at_the_tier_a_request_size_settles_at(
    priced: TestClient, master_header: dict[str, str]
) -> None:
    # Nebius is the cheaper base rate; past 100K tokens its tier makes it the dearer one.
    response = priced.post(
        "/v1/pricing",
        json={
            "model_key": _NEBIUS_GLM,
            "input_price_per_million": 0.5,
            "output_price_per_million": 2.0,
            "pricing_tiers": [{"min_input_tokens": 100_000, "input_price_per_million": 1.0}],
        },
        headers=master_header,
    )
    assert response.status_code == status.HTTP_200_OK, response.text

    base = next(m for m in _get(priced, "/v1/catalog/models", headers=master_header)["models"] if m["id"] == "glm-5-3")
    assert base["min_input_price_per_million"] == 0.5
    at_8k = _get(priced, "/v1/catalog/models?at_context=8000", headers=master_header)["models"]
    assert next(m for m in at_8k if m["id"] == "glm-5-3")["min_input_price_per_million"] == 0.5
    listed = _get(priced, "/v1/catalog/models?at_context=200000", headers=master_header)["models"]
    at_200k = next(m for m in listed if m["id"] == "glm-5-3")
    # Fireworks' 0.7 is now the floor; the tier's unset output rate falls back to the base.
    assert at_200k["min_input_price_per_million"] == 0.7
    assert at_200k["min_output_price_per_million"] == 2.0
    assert priced.get("/v1/catalog/models?at_context=0", headers=master_header).status_code == 422


def test_a_visitor_reads_the_catalog_only_while_it_is_public(
    catalog_client: TestClient, public_client: TestClient, master_header: dict[str, str]
) -> None:
    # Off (the default): a visitor is refused as before.
    assert catalog_client.get("/v1/catalog/models").status_code in (
        status.HTTP_401_UNAUTHORIZED,
        status.HTTP_403_FORBIDDEN,
    )

    _price(public_client, master_header, _NEBIUS_GLM, 0.5, 2.0)
    body = _get(public_client, "/v1/catalog/models")
    assert [model["id"] for model in body["models"]] == ["glm-5-3"]
    detail = _get(public_client, "/v1/catalog/models/glm-5-3")
    offering = detail["offerings"][0]
    assert offering["credential"] == "deployment"
    assert offering["price_source"] == "deployment"
    # A visitor has no organization, so nothing of a tenant's is rolled up.
    assert offering["usage_30d"] is None

    # A credential that is present and wrong is still a wrong credential.
    with patch.object(mcs, "_fetch", new=AsyncMock(return_value=CATALOG)):
        refused = public_client.get("/v1/catalog/models", headers={"Authorization": "Bearer not-a-key"})
    assert refused.status_code in (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN)


@pytest.fixture
def throttled_public_client(postgres_url: str, clean_database: None) -> Generator[TestClient]:
    mcs.clear_catalog_cache()
    try:
        yield from build_test_client(_config(postgres_url, public_catalog=True, public_catalog_rate_limit_per_minute=2))
    finally:
        mcs.clear_catalog_cache()


def test_a_visitor_is_throttled_on_the_catalogs_own_budget(
    throttled_public_client: TestClient, master_header: dict[str, str]
) -> None:
    # The budget is per client address and counts anonymous reads only.
    with patch.object(mcs, "_fetch", new=AsyncMock(return_value=CATALOG)):
        statuses = [throttled_public_client.get("/v1/catalog/models").status_code for _ in range(3)]
        assert statuses == [status.HTTP_200_OK, status.HTTP_200_OK, status.HTTP_429_TOO_MANY_REQUESTS]
        # The detail shares the budget: it is the same catalog being read.
        detail = throttled_public_client.get("/v1/catalog/models/glm-5-3")
        assert detail.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        # A credentialed caller is not a visitor and is not counted against it.
        signed_in = throttled_public_client.get("/v1/catalog/models", headers=master_header)
        assert signed_in.status_code == status.HTTP_200_OK


def test_a_signed_in_caller_sees_their_own_usage_of_an_offering(
    priced: TestClient, master_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    """The listed rate is what a token costs; this is what the tokens cost."""
    from sqlmodel import select

    from gateway.models.entities import UsageLog
    from gateway.models.tenancy import Workspace

    # The master key acts in the default workspace, which boot provisioned.
    assert priced.get("/v1/organizations/me", headers=master_header).status_code == status.HTTP_200_OK
    session = db_session_factory()
    try:
        workspace_id = session.execute(select(Workspace.id)).scalars().first()
        assert workspace_id is not None
        for cached in (0, 800):
            session.add(
                UsageLog(
                    workspace_id=workspace_id,
                    timestamp=datetime.now(UTC) - timedelta(days=1),
                    model="zai-org/GLM-5.3",
                    provider="nebius",
                    endpoint="/v1/chat/completions",
                    status="success",
                    prompt_tokens=1000,
                    completion_tokens=500,
                    total_tokens=1500,
                    cache_read_tokens=cached,
                    cost=0.0015,
                )
            )
        # Too old to count, and a failure that never billed.
        session.add(
            UsageLog(
                workspace_id=workspace_id,
                timestamp=datetime.now(UTC) - timedelta(days=45),
                model="zai-org/GLM-5.3",
                provider="nebius",
                endpoint="/v1/chat/completions",
                status="success",
                prompt_tokens=1_000_000,
                completion_tokens=0,
                total_tokens=1_000_000,
                cost=1,
            )
        )
        session.commit()
    finally:
        session.close()

    body = _get(priced, "/v1/catalog/models/glm-5-3", headers=master_header)
    by_selector = {offering["selector"]: offering for offering in body["offerings"]}
    usage = by_selector[_NEBIUS_GLM]["usage_30d"]
    assert usage["requests"] == 2
    assert usage["total_tokens"] == 3000
    assert usage["cache_read_tokens"] == 800
    assert usage["cache_hit_rate"] == 0.4
    assert usage["spend_usd"] == 0.003
    assert usage["effective_price_per_million"] == 1.0
    assert by_selector[_FIREWORKS_GLM]["usage_30d"] is None
