"""When a candidate is known before dispatch to have no credential, and what an unpriced request is told."""

import uuid
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from any_llm import LLMProvider
from any_llm.exceptions import MissingApiKeyError

from gateway.api.routes._platform import get_shared_rejection
from gateway.core.config import GatewayConfig
from gateway.core.error_codes import ERROR_CODE_HEADER, PRICING_REQUIRED, PROVIDER_NOT_CONFIGURED
from gateway.ports.model_provider_port import NO_ORGANIZATION
from gateway.services.provider_kwargs import ResolvedProvider
from gateway.services.providers import missing_provider_credential, unpriced_model_refusal

_ORG = uuid.uuid4()
# Declares no provider, so cohere is undeclared unless a test says otherwise.
_CONFIG = GatewayConfig(providers={})


@pytest.fixture(autouse=True)
def _no_cohere_key(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("COHERE_API_KEY", "CO_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def _cohere(kwargs: dict[str, Any]) -> ResolvedProvider:
    return ResolvedProvider(instance="cohere", provider=LLMProvider.COHERE, model="rerank-v3.5", kwargs=kwargs)


def _hosted(*providers: str) -> Any:
    port = AsyncMock()
    port.get_hosted_models.return_value = dict.fromkeys(providers)
    return port


@pytest.mark.asyncio
async def test_an_alias_hides_the_provider_and_its_variable() -> None:
    resolved = ResolvedProvider(
        instance="cohere", provider=LLMProvider.COHERE, model="rerank-v3.5", kwargs={}, alias="house-reranker"
    )
    missing = await missing_provider_credential(_CONFIG, resolved, hosted=None, organization_id=_ORG)
    assert missing is not None
    assert "'house-reranker'" in missing.detail
    assert "cohere" not in missing.detail.lower()
    assert "CO_API_KEY" not in missing.detail


@pytest.mark.asyncio
async def test_an_uncredentialed_provider_is_named_with_its_variable() -> None:
    missing = await missing_provider_credential(_CONFIG, _cohere({}), hosted=None, organization_id=_ORG)
    assert missing is not None
    assert missing.provider == "cohere"
    assert missing.env_var is not None
    assert "COHERE_API_KEY" in missing.env_var


@pytest.mark.asyncio
async def test_a_credential_in_hand_is_not_missing() -> None:
    assert (
        await missing_provider_credential(_CONFIG, _cohere({"api_key": "k"}), hosted=None, organization_id=_ORG) is None
    )


@pytest.mark.asyncio
async def test_the_environment_variable_does_not_count_for_an_undeclared_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COHERE_API_KEY", "k")
    missing = await missing_provider_credential(_CONFIG, _cohere({}), hosted=None, organization_id=_ORG)
    assert missing is not None


@pytest.mark.asyncio
async def test_the_environment_variable_never_serves_a_declared_entry_naming_no_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only an entry's own ``api_key`` (``${COHERE_API_KEY}`` included) credentials it."""
    monkeypatch.setenv("COHERE_API_KEY", "k")
    config = GatewayConfig(providers={"cohere": {"client_args": {"timeout": 60}}})
    assert await missing_provider_credential(config, _cohere({}), hosted=None, organization_id=_ORG) is not None


@pytest.mark.asyncio
async def test_the_refusal_names_how_to_declare_the_provider() -> None:
    missing = await missing_provider_credential(_CONFIG, _cohere({}), hosted=None, organization_id=_ORG)
    assert missing is not None
    assert "providers:" in missing.detail
    assert "${COHERE_API_KEY}" in missing.detail
    assert "Hosted providers page" in missing.detail
    assert "organization provider key" in missing.detail


@pytest.mark.asyncio
async def test_a_provider_the_hosted_port_serves_is_not_missing() -> None:
    port = _hosted("cohere")
    assert await missing_provider_credential(_CONFIG, _cohere({}), hosted=port, organization_id=_ORG) is None
    port.get_hosted_models.assert_awaited_once_with(organization_id=_ORG)


@pytest.mark.asyncio
async def test_without_an_organization_the_hosted_port_is_asked_as_one_no_restriction_names() -> None:
    """A request with no organization is served on what is restricted to none, so that is what it is asked."""
    port = _hosted("cohere")
    assert await missing_provider_credential(_CONFIG, _cohere({}), hosted=port, organization_id=None) is None
    port.get_hosted_models.assert_awaited_once_with(organization_id=NO_ORGANIZATION)


@pytest.mark.asyncio
async def test_a_hosted_port_serving_other_providers_does_not_hide_it() -> None:
    assert (
        await missing_provider_credential(_CONFIG, _cohere({}), hosted=_hosted("openai"), organization_id=_ORG)
        is not None
    )


@pytest.mark.asyncio
async def test_an_unpriced_model_on_an_uncallable_provider_is_refused_as_unconfigured() -> None:
    refusal = await unpriced_model_refusal(
        _CONFIG, "cohere:rerank-v3.5", _cohere({}), plan=None, hosted=None, organization_id=_ORG
    )
    assert (refusal.status_code, refusal.code) == (424, PROVIDER_NOT_CONFIGURED)


@pytest.mark.asyncio
async def test_an_unpriced_model_on_a_callable_provider_is_refused_for_pricing() -> None:
    refusal = await unpriced_model_refusal(
        _CONFIG, "cohere:rerank-typo", _cohere({"api_key": "k"}), plan=None, hosted=None, organization_id=_ORG
    )
    assert (refusal.status_code, refusal.code) == (402, PRICING_REQUIRED)
    assert "'cohere:rerank-typo'" in refusal.detail


@pytest.mark.asyncio
async def test_a_plan_with_several_candidates_keeps_the_pricing_refusal() -> None:
    plan: Any = SimpleNamespace(attempts=[object(), object()])
    refusal = await unpriced_model_refusal(_CONFIG, "fast", _cohere({}), plan=plan, hosted=None, organization_id=_ORG)
    assert refusal.code == PRICING_REQUIRED


@pytest.mark.asyncio
async def test_an_unresolved_selector_keeps_the_pricing_refusal() -> None:
    refusal = await unpriced_model_refusal(_CONFIG, "nonsense", None, plan=None, hosted=None, organization_id=_ORG)
    assert refusal.code == PRICING_REQUIRED


def test_candidates_that_all_lacked_a_credential_are_answered_as_unconfigured() -> None:
    plain = get_shared_rejection([MissingApiKeyError("openai", "OPENAI_API_KEY")] * 2)
    assert plain is not None
    assert plain.status_code == 424
    assert (plain.headers or {})[ERROR_CODE_HEADER] == PROVIDER_NOT_CONFIGURED
