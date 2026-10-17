"""When a candidate is known before dispatch to have no credential, and what an unpriced request is told."""

import uuid
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from any_llm import LLMProvider
from any_llm.exceptions import MissingApiKeyError

from gateway.adapters.model_provider_adapter import HostedProviderModelProviderAdapter
from gateway.api.routes._platform import get_shared_rejection
from gateway.core.config import GatewayConfig
from gateway.core.error_codes import (
    ERROR_CODE_HEADER,
    MANAGED_CREDENTIAL_UNSUPPORTED,
    PRICING_REQUIRED,
    PROVIDER_NOT_CONFIGURED,
)
from gateway.services.provider_kwargs import (
    MANAGED_SURFACE_ROUTING_POLICY,
    ManagedCredentialUnsupported,
    ManagedCredentialUnsupportedError,
    ResolvedProvider,
    missing_credential,
    shared_managed_refusal,
)
from gateway.services.providers import (
    managed_only_candidates,
    missing_provider_credential,
    refuse_managed_candidate,
    refuse_managed_only,
    unpriced_model_refusal,
)
from gateway.types.attempt import Attempt

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
    assert "Providers page" in missing.detail
    assert "organization provider key" in missing.detail


@pytest.mark.asyncio
async def test_a_provider_the_hosted_port_serves_is_not_missing() -> None:
    port = _hosted("cohere")
    assert await missing_provider_credential(_CONFIG, _cohere({}), hosted=port, organization_id=_ORG) is None
    port.get_hosted_models.assert_awaited_once_with(organization_id=_ORG)


@pytest.mark.asyncio
async def test_the_hosted_port_is_not_asked_without_an_organization() -> None:
    port = _hosted("cohere")
    assert await missing_provider_credential(_CONFIG, _cohere({}), hosted=port, organization_id=None) is not None
    port.get_hosted_models.assert_not_awaited()


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


@pytest.mark.asyncio
async def test_a_managed_model_nothing_declares_is_managed_only() -> None:
    managed = await managed_only_candidates(
        _CONFIG, [_cohere({})], hosted=lambda: _hosted("cohere"), organization_id=_ORG
    )
    assert managed == {("cohere", "rerank-v3.5")}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "offered",
    [{}, {"openai": None}, {"cohere": frozenset()}, {"cohere": frozenset({"rerank-v3.0"})}],
    ids=["nothing", "another-provider", "no-models", "another-model"],
)
async def test_a_model_the_deployment_does_not_offer_is_not_managed_only(offered: dict[str, Any]) -> None:
    port = AsyncMock()
    port.get_hosted_models.return_value = offered
    assert not await managed_only_candidates(_CONFIG, [_cohere({})], hosted=lambda: port, organization_id=_ORG)


@pytest.mark.asyncio
async def test_an_advertised_model_set_names_the_model() -> None:
    port = AsyncMock()
    port.get_hosted_models.return_value = {"cohere": frozenset({"rerank-v3.5"})}
    assert await managed_only_candidates(_CONFIG, [_cohere({})], hosted=lambda: port, organization_id=_ORG)


@pytest.mark.asyncio
async def test_the_port_is_not_asked_for_a_credentialed_candidate_or_no_organization() -> None:
    port = _hosted("cohere")
    assert not await managed_only_candidates(
        _CONFIG, [_cohere({"api_key": "k"})], hosted=lambda: port, organization_id=_ORG
    )
    assert not await managed_only_candidates(_CONFIG, [_cohere({})], hosted=lambda: port, organization_id=None)
    port.get_hosted_models.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_core_adapter_with_nothing_hosted_changes_nothing() -> None:
    hosted = HostedProviderModelProviderAdapter(None)
    assert not await managed_only_candidates(_CONFIG, [_cohere({})], hosted=lambda: hosted, organization_id=_ORG)


@pytest.mark.asyncio
async def test_a_port_that_cannot_list_leaves_the_ordinary_refusal() -> None:
    port = AsyncMock()
    port.get_hosted_models.side_effect = RuntimeError("store down")
    assert not await managed_only_candidates(_CONFIG, [_cohere({})], hosted=lambda: port, organization_id=_ORG)


def test_a_managed_refusal_reads_as_its_own_missing_credential() -> None:
    error = ManagedCredentialUnsupportedError(
        "cohere", selector="cohere:rerank-v3.5", surface=MANAGED_SURFACE_ROUTING_POLICY
    )
    missing = missing_credential(error)
    assert isinstance(missing, ManagedCredentialUnsupported)
    assert missing.code == MANAGED_CREDENTIAL_UNSUPPORTED
    assert missing.status_code == 424
    assert missing.detail in str(error)
    assert "not through a routing policy with fallbacks" in missing.detail


def test_a_plan_that_skipped_a_managed_candidate_names_it() -> None:
    managed = ManagedCredentialUnsupportedError(
        "cohere", selector="cohere:rerank-v3.5", surface=MANAGED_SURFACE_ROUTING_POLICY
    )
    other = ManagedCredentialUnsupportedError(
        "cohere", selector="cohere:rerank-v3.5", surface=MANAGED_SURFACE_ROUTING_POLICY
    )
    rejection = get_shared_rejection([managed, other, MissingApiKeyError("openai", "OPENAI_API_KEY")])
    assert rejection is not None
    assert rejection.status_code == 424
    assert (rejection.headers or {})[ERROR_CODE_HEADER] == MANAGED_CREDENTIAL_UNSUPPORTED
    # The name the caller wrote is named once, however many of its candidates were managed.
    assert str(rejection.detail).startswith("'cohere:rerank-v3.5' is offered on this deployment's managed")

    plain = get_shared_rejection([MissingApiKeyError("openai", "OPENAI_API_KEY")] * 2)
    assert plain is not None
    assert (plain.headers or {})[ERROR_CODE_HEADER] == PROVIDER_NOT_CONFIGURED


def test_shared_managed_refusal_names_each_selector_once_and_ignores_plain_misses() -> None:
    def managed(selector: str) -> ManagedCredentialUnsupported:
        return ManagedCredentialUnsupported(
            provider="x", env_var=None, selectors=(selector,), surface=MANAGED_SURFACE_ROUTING_POLICY
        )

    plain = missing_credential(MissingApiKeyError("openai", "OPENAI_API_KEY"))
    assert plain is not None
    assert shared_managed_refusal([plain]) is None
    shared = shared_managed_refusal([managed("fast"), plain, managed("fast")])
    assert shared is not None
    assert shared.selectors == ("fast",)
    assert shared.detail.startswith("'fast' is offered")


def test_a_policy_candidate_is_named_by_what_the_caller_wrote() -> None:
    attempt = Attempt(
        position=1, instance="cohere", provider=LLMProvider.COHERE, model="rerank-v3.5", display_model="fast"
    )
    refuse_managed_candidate(frozenset(), attempt)
    with pytest.raises(ManagedCredentialUnsupportedError) as raised:
        refuse_managed_candidate(frozenset({("cohere", "rerank-v3.5")}), attempt)
    assert raised.value.selector == "fast"
    refused = missing_credential(raised.value)
    assert refused is not None
    assert "cohere" not in refused.detail


@pytest.mark.asyncio
async def test_a_single_candidate_is_named_by_its_alias() -> None:
    resolved = ResolvedProvider(
        instance="cohere", provider=LLMProvider.COHERE, model="rerank-v3.5", kwargs={}, alias="house-reranker"
    )
    with pytest.raises(ManagedCredentialUnsupportedError) as raised:
        await refuse_managed_only(
            _CONFIG, resolved, hosted=lambda: _hosted("cohere"), organization_id=_ORG, surface="/v1/rerank"
        )
    assert raised.value.selector == "house-reranker"
    await refuse_managed_only(
        _CONFIG, resolved, hosted=lambda: _hosted("openai"), organization_id=_ORG, surface="/v1/rerank"
    )
