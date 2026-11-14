"""A rebound ``AccountCountryPort`` is checked by the provider country gate beside the request's country.

The core adapter knows no account's country, so with nothing bound the gate
reads the request's country alone, as before. A build that bills its customers
binds an adapter that answers from the billing address; the probe bound here
answers whatever a test sets, so each test shows one combination of the two
countries: the request's from the trusted header, the account's from the port.
"""

import itertools
import sys
from collections.abc import Generator
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.core.error_codes import COUNTRY_NOT_SUPPORTED, ERROR_CODE_HEADER
from gateway.models.routing import RoutingConfig

from .conftest import build_test_client
from .test_embeddings_endpoint import _mock_embedding_response
from .test_routing_policies import _awaited_model, _completion

MODULE = "probe_account_country"
_ACCOUNT_PROBE = """
from gateway.container import Container
from gateway.ports.account_country_port import AccountCountryPort, AccountCountryUnavailableError

ANSWER = {"country": None, "down": False}
ASKED = []


class ProbeAccountCountry:
    def __init__(self, session):
        pass

    async def country(self, *, organization_id):
        ASKED.append(organization_id)
        if ANSWER["down"]:
            raise AccountCountryUnavailableError("the billing store is down")
        return ANSWER["country"]
"""
PROBE_BOOTSTRAP = (
    _ACCOUNT_PROBE
    + """

def register(container: Container) -> None:
    container.bind(AccountCountryPort, ProbeAccountCountry)
"""
)
# An overlay's own fleet beside the probe: its credentials carry a country list
# the core hosted-provider index never sees, so only the port can say one exists.
FLEET_BOOTSTRAP = (
    _ACCOUNT_PROBE
    + """
from gateway.ports.model_provider_port import HostedCredential, ModelProviderPort

FLEET = {"limits": True}


class FleetModelProvider:
    def __init__(self, session):
        pass

    async def resolve_hosted_credential(self, *, organization_id, workspace_id, provider, model):
        countries = ("US",) if FLEET["limits"] else None
        return HostedCredential(
            api_key="fleet", api_base=None, response_provider=provider, supported_countries=countries
        )

    async def get_hosted_models(self, *, organization_id):
        return {"mistral": None}

    async def limits_countries(self):
        return FLEET["limits"]


def register(container: Container) -> None:
    container.bind(AccountCountryPort, ProbeAccountCountry)
    container.bind(ModelProviderPort, FleetModelProvider)
"""
)
HEADERS = {API_KEY_HEADER: "Bearer test-master-key"}
_COUNTRY_HEADER = "CF-IPCountry"

# One module name per test: the app is cached per distinct config, and a fresh
# name is what gets each test an app whose container imported the probe module
# the test then sets its answer on.
_NAMES = (f"{MODULE}_{n}" for n in itertools.count())

Rebound = tuple[TestClient, ModuleType, dict[str, str]]


def _rebound(
    postgres_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bootstrap: str, **settings: Any
) -> Generator[Rebound]:
    """A client on an app whose container ran ``bootstrap``, with a keyed user ``alice``."""
    name = next(_NAMES)
    (tmp_path / f"{name}.py").write_text(bootstrap)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(name, None)
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        auto_migrate=False,
        require_pricing=False,
        model_discovery=False,
        client_country_header=_COUNTRY_HEADER,
        bootstrap=f"{name}:register",
        **settings,
    )
    try:
        for client in build_test_client(config):
            resp = client.post(f"{API_ROOT}/users", json={"user_id": "alice"}, headers=HEADERS)
            assert resp.status_code == 200, resp.text
            resp = client.post(f"{API_ROOT}/keys", json={"user_id": "alice"}, headers=HEADERS)
            assert resp.status_code == 200, resp.text
            yield client, sys.modules[name], {API_KEY_HEADER: f"Bearer {resp.json()['key']}"}
    finally:
        sys.modules.pop(name, None)


@pytest.fixture
def rebound(postgres_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Rebound]:
    yield from _rebound(
        postgres_url,
        tmp_path,
        monkeypatch,
        PROBE_BOOTSTRAP,
        providers={"openai": {"api_key": "sk-openai"}, "anthropic": {"api_key": "sk-ant"}},
        provider_supported_countries={"openai": ["US"]},
        routing=RoutingConfig.model_validate(
            {
                "policies": {
                    "fast": {"select": [{"default": "openai:gpt-5-mini"}], "on_failure": ["anthropic:claude-haiku-4-5"]}
                }
            }
        ),
    )


@pytest.fixture
def fleet(postgres_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Rebound]:
    """No list the core knows of: no ``provider_supported_countries`` and no hosted provider, only the fleet's."""
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    yield from _rebound(postgres_url, tmp_path, monkeypatch, FLEET_BOOTSTRAP)


def _chat(client: TestClient, model: str, country: str, *, auth: dict[str, str]) -> Any:
    return client.post(
        f"{API_ROOT}/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": "hi"}], "user": "alice"},
        headers={**auth, _COUNTRY_HEADER: country},
    )


@pytest.mark.parametrize("account_country", [None, "US"], ids=["unknown", "listed"])
def test_an_account_the_list_holds_or_nobody_knows_is_served(rebound: Rebound, account_country: str | None) -> None:
    client, probe, key = rebound
    probe.ANSWER["country"] = account_country

    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(return_value=_completion("gpt-5-mini"))) as mock:
        resp = _chat(client, "openai:gpt-5-mini", "US", auth=key)

    assert resp.status_code == 200, resp.text
    assert _awaited_model(mock) == "openai:gpt-5-mini"
    assert len(probe.ASKED) == 1


def test_an_account_outside_the_list_is_refused_from_a_listed_country(rebound: Rebound) -> None:
    client, probe, key = rebound
    probe.ANSWER["country"] = "FR"

    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock()) as mock:
        resp = _chat(client, "openai:gpt-5-mini", "US", auth=key)

    assert resp.status_code == 403, resp.text
    assert resp.headers[ERROR_CODE_HEADER] == COUNTRY_NOT_SUPPORTED
    mock.assert_not_awaited()
    # The caller's own country is named; neither the account's nor which half refused is.
    assert resp.json()["detail"] == "Model 'openai:gpt-5-mini' is not available to requests from country 'US'."
    rows = client.get(f"{API_ROOT}/usage", params={"status": "error"}, headers=HEADERS).json()
    assert [(row["provider"], row["status_code"]) for row in rows] == [("openai", 403)]


def test_a_policy_falls_over_past_a_provider_the_account_may_not_use(rebound: Rebound) -> None:
    client, probe, key = rebound
    probe.ANSWER["country"] = "FR"

    with patch(
        "gateway.api.routes.chat.acompletion", new=AsyncMock(return_value=_completion("claude-haiku-4-5"))
    ) as mock:
        resp = _chat(client, "fast", "US", auth=key)

    assert resp.status_code == 200, resp.text
    assert _awaited_model(mock) == "anthropic:claude-haiku-4-5"


def test_the_master_key_has_no_account_and_is_gated_on_its_own_country(rebound: Rebound) -> None:
    client, probe, _ = rebound
    probe.ANSWER["country"] = "FR"

    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(return_value=_completion("gpt-5-mini"))) as mock:
        resp = _chat(client, "openai:gpt-5-mini", "US", auth=HEADERS)

    assert resp.status_code == 200, resp.text
    assert _awaited_model(mock) == "openai:gpt-5-mini"
    assert probe.ASKED == []


def test_an_account_the_port_cannot_resolve_is_refused_where_a_list_applies(rebound: Rebound) -> None:
    client, probe, key = rebound
    probe.ANSWER["down"] = True

    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock()) as mock:
        refused = _chat(client, "openai:gpt-5-mini", "US", auth=key)
    with patch(
        "gateway.api.routes.chat.acompletion", new=AsyncMock(return_value=_completion("claude-haiku-4-5"))
    ) as mock_unlisted:
        served = _chat(client, "anthropic:claude-haiku-4-5", "US", auth=key)

    assert refused.status_code == 403, refused.text
    assert refused.headers[ERROR_CODE_HEADER] == COUNTRY_NOT_SUPPORTED
    assert refused.json()["detail"] == "Model 'openai:gpt-5-mini' is not available to requests from country 'US'."
    mock.assert_not_awaited()
    assert served.status_code == 200, served.text
    assert _awaited_model(mock_unlisted) == "anthropic:claude-haiku-4-5"


def test_a_pass_through_call_is_refused_on_the_accounts_country(rebound: Rebound) -> None:
    client, probe, key = rebound
    probe.ANSWER["country"] = "FR"

    with patch(
        "gateway.api.routes.embeddings.aembedding", new=AsyncMock(return_value=_mock_embedding_response())
    ) as mock:
        resp = client.post(
            f"{API_ROOT}/embeddings",
            json={"model": "openai:text-embedding-3-small", "input": "hello"},
            headers={**key, _COUNTRY_HEADER: "US"},
        )

    assert resp.status_code == 403, resp.text
    assert resp.headers[ERROR_CODE_HEADER] == COUNTRY_NOT_SUPPORTED
    mock.assert_not_awaited()


def test_a_batch_lifecycle_call_is_refused_on_the_accounts_country(rebound: Rebound) -> None:
    client, probe, key = rebound
    probe.ANSWER["country"] = "FR"

    with patch("gateway.api.routes.batches.aretrieve_batch", new=AsyncMock()) as mock:
        resp = client.get(
            f"{API_ROOT}/batches/batch_abc123", params={"provider": "openai"}, headers={**key, _COUNTRY_HEADER: "US"}
        )

    assert resp.status_code == 403, resp.text
    assert resp.headers[ERROR_CODE_HEADER] == COUNTRY_NOT_SUPPORTED
    assert resp.json()["detail"] == "Provider 'openai' is not available to requests from country 'US'."
    mock.assert_not_awaited()


def test_a_list_only_the_bound_port_knows_still_refuses_the_accounts_country(fleet: Rebound) -> None:
    """The core index holds no list, so only the port's own answer gets the account looked up."""
    client, probe, key = fleet
    probe.ANSWER["country"] = "FR"

    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock()) as mock:
        resp = _chat(client, "mistral:mistral-small-latest", "US", auth=key)

    assert resp.status_code == 403, resp.text
    assert resp.headers[ERROR_CODE_HEADER] == COUNTRY_NOT_SUPPORTED
    assert (
        resp.json()["detail"] == "Model 'mistral:mistral-small-latest' is not available to requests from country 'US'."
    )
    mock.assert_not_awaited()
    assert len(probe.ASKED) == 1


def test_the_account_is_not_looked_up_where_no_list_could_refuse_it(fleet: Rebound) -> None:
    client, probe, key = fleet
    probe.FLEET["limits"] = False
    probe.ANSWER["country"] = "FR"

    with patch(
        "gateway.api.routes.chat.acompletion", new=AsyncMock(return_value=_completion("mistral-small-latest"))
    ) as mock:
        resp = _chat(client, "mistral:mistral-small-latest", "US", auth=key)

    assert resp.status_code == 200, resp.text
    assert _awaited_model(mock) == "mistral:mistral-small-latest"
    assert probe.ASKED == []
