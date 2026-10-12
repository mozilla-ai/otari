"""When a candidate is known before dispatch to have no credential."""

import uuid
from typing import Any
from unittest.mock import AsyncMock

import pytest
from any_llm import LLMProvider

from gateway.services.provider_kwargs import ResolvedProvider
from gateway.services.providers import missing_provider_credential

_ORG = uuid.uuid4()


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
async def test_an_uncredentialed_provider_is_named_with_its_variable() -> None:
    missing = await missing_provider_credential(_cohere({}), hosted=None, organization_id=_ORG)
    assert missing is not None
    assert missing.provider == "cohere"
    assert missing.env_var is not None
    assert "COHERE_API_KEY" in missing.env_var


@pytest.mark.asyncio
async def test_a_credential_in_hand_is_not_missing() -> None:
    assert await missing_provider_credential(_cohere({"api_key": "k"}), hosted=None, organization_id=_ORG) is None


@pytest.mark.asyncio
async def test_the_environment_variable_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COHERE_API_KEY", "k")
    assert await missing_provider_credential(_cohere({}), hosted=None, organization_id=_ORG) is None


@pytest.mark.asyncio
async def test_a_provider_the_hosted_port_serves_is_not_missing() -> None:
    port = _hosted("cohere")
    assert await missing_provider_credential(_cohere({}), hosted=port, organization_id=_ORG) is None
    port.get_hosted_models.assert_awaited_once_with(organization_id=_ORG)


@pytest.mark.asyncio
async def test_the_hosted_port_is_not_asked_without_an_organization() -> None:
    port = _hosted("cohere")
    assert await missing_provider_credential(_cohere({}), hosted=port, organization_id=None) is not None
    port.get_hosted_models.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_hosted_port_serving_other_providers_does_not_hide_it() -> None:
    assert await missing_provider_credential(_cohere({}), hosted=_hosted("openai"), organization_id=_ORG) is not None
