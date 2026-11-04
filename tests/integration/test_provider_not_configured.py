"""A request for a provider the gateway holds no credential for.

Nothing leaves the gateway for such a call, so the failure is the deployment's
configuration rather than the provider's. It is answered as a client error that
names the provider and the remedy, with a stable code, and never as a 5xx: an
SDK retries a 5xx, and no retry can supply a key. A provider nothing declares is
refused this way even when its SDK variable is set, because any-llm would
otherwise authenticate from the variable and serve every caller on it.
"""

from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from any_llm.exceptions import MissingApiKeyError
from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig, _resolve_env_vars
from gateway.core.error_codes import ERROR_CODE_HEADER, PROVIDER_NOT_CONFIGURED
from gateway.models.routing import RoutingConfig
from gateway.models.usage import UsageLog
from gateway.services.model_discovery_service import ProviderDiscovery
from gateway.services.secret_box import generate_secret_key

from .conftest import build_test_client

# any-llm's own sentence, which the response must not copy.
_ANY_LLM_WORDING = "Please provide it in the config"


@pytest.fixture(autouse=True)
def _no_provider_keys_in_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The providers named below must be unconfigured whatever the developer's shell holds."""
    for name in ("ANTHROPIC_API_KEY", "COHERE_API_KEY", "MISTRAL_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def client(postgres_url: str) -> Generator[TestClient]:
    """A client on a deployment that declares no provider, unlike the shared config."""
    yield from build_test_client(
        GatewayConfig(
            database_url=postgres_url,
            master_key="test-master-key",
            host="127.0.0.1",
            port=8000,
            auto_migrate=False,
            require_pricing=False,
            model_discovery=False,
            providers={},
        )
    )


def _assert_provider_not_configured(response: Any, provider: str) -> None:
    assert response.status_code == 424, response.text
    assert response.headers[ERROR_CODE_HEADER] == PROVIDER_NOT_CONFIGURED
    body = response.json()
    assert body["code"] == PROVIDER_NOT_CONFIGURED
    assert f"'{provider}'" in body["detail"]
    assert "hosted provider" in body["detail"]
    assert _ANY_LLM_WORDING not in body["detail"]


def test_chat_names_the_provider_and_what_to_configure(
    client: TestClient,
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
    test_user: dict[str, Any],
    db_session: Session,
) -> None:
    response = client.post(
        f"{API_ROOT}/chat/completions",
        json={"model": "anthropic:claude-sonnet-4-5", "messages": [{"role": "user", "content": "Hello"}]},
        headers=api_key_header,
    )
    _assert_provider_not_configured(response, "anthropic")

    # Recorded as the refusal it is, so the error rate can tell it from a provider outage.
    row = db_session.query(UsageLog).filter(UsageLog.api_key_id == api_key_obj["id"]).one()
    assert row.status == "error"
    assert row.status_code == 424


def test_a_streaming_request_is_refused_before_the_stream_opens(
    client: TestClient,
    api_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    response = client.post(
        f"{API_ROOT}/chat/completions",
        json={
            "model": "anthropic:claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        },
        headers=api_key_header,
    )
    _assert_provider_not_configured(response, "anthropic")
    assert response.headers["content-type"].startswith("application/json")


def test_a_passthrough_route_answers_the_same_way(
    client: TestClient,
    api_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    response = client.post(
        f"{API_ROOT}/embeddings",
        json={"model": "cohere:embed-english-v3.0", "input": "hello"},
        headers=api_key_header,
    )
    _assert_provider_not_configured(response, "cohere")


def test_the_messages_route_answers_in_its_own_envelope(
    client: TestClient,
    api_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    response = client.post(
        f"{API_ROOT}/messages",
        json={
            "model": "anthropic:claude-sonnet-4-5",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "Hello"}],
        },
        headers=api_key_header,
    )
    assert response.status_code == 424, response.text
    assert response.headers[ERROR_CODE_HEADER] == PROVIDER_NOT_CONFIGURED
    message = response.json()["error"]["message"]
    assert "'anthropic'" in message
    assert "hosted provider" in message
    assert _ANY_LLM_WORDING not in message


# ---------------------------------------------------------------------------
# A routing policy whose head has no credential falls over to one that has
# ---------------------------------------------------------------------------

_MASTER = {API_KEY_HEADER: "Bearer test-master-key"}


@pytest.fixture
def failover_config(postgres_url: str) -> GatewayConfig:
    return GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        require_pricing=False,
        model_discovery=False,
        providers={"openai": {"api_key": "sk-openai"}},
        routing=RoutingConfig.model_validate(
            {
                "policies": {
                    "fast": {
                        "select": [{"default": "mistral:mistral-small-latest"}],
                        "on_failure": ["openai:gpt-5-mini"],
                    },
                    "solo": {"select": [{"default": "mistral:mistral-small-latest"}]},
                    "unconfigured_pair": {
                        "select": [{"default": "mistral:mistral-small-latest"}],
                        "on_failure": ["cohere:command-r"],
                    },
                }
            }
        ),
    )


@pytest.fixture
def failover_client(failover_config: GatewayConfig) -> Generator[TestClient]:
    yield from build_test_client(failover_config)


def _completion(model: str) -> ChatCompletion:
    return ChatCompletion(
        id="cmpl-1",
        choices=[Choice(finish_reason="stop", index=0, message=ChatCompletionMessage(role="assistant", content="hi"))],
        created=0,
        model=model,
        object="chat.completion",
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


async def _provider_with_only_an_openai_key(**kwargs: Any) -> ChatCompletion:
    """What any-llm does with the candidates: refuses mistral for want of a key, serves openai."""
    if kwargs["model"].startswith("mistral"):
        raise MissingApiKeyError("mistral", "MISTRAL_API_KEY")
    if kwargs["model"].startswith("cohere"):
        raise MissingApiKeyError("cohere", "COHERE_API_KEY")
    return _completion(kwargs["model"])


def _chat(client: TestClient, model: str) -> Any:
    return client.post(
        f"{API_ROOT}/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": "hi"}], "user": "test-user"},
        headers=_MASTER,
    )


def test_a_candidate_without_a_credential_is_passed_over(failover_client: TestClient) -> None:
    assert failover_client.post(f"{API_ROOT}/users", json={"user_id": "test-user"}, headers=_MASTER).status_code == 200
    with patch(
        "gateway.api.routes.chat.acompletion", new=AsyncMock(side_effect=_provider_with_only_an_openai_key)
    ) as mock:
        response = _chat(failover_client, "fast")
    assert response.status_code == 200, response.text
    dispatched = [call.kwargs["model"] for call in mock.await_args_list]
    # Refused before dispatch, so mistral is never called at all.
    assert [model.split(":")[0] for model in dispatched] == ["openai"]


def test_a_policy_with_no_credentialed_candidate_is_refused_as_unconfigured(failover_client: TestClient) -> None:
    assert failover_client.post(f"{API_ROOT}/users", json={"user_id": "test-user"}, headers=_MASTER).status_code == 200
    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(side_effect=_provider_with_only_an_openai_key)):
        response = _chat(failover_client, "solo")
    # A one-target policy resolves as an alias does, so the refusal names it rather than its target.
    assert response.status_code == 424, response.text
    assert response.json()["code"] == PROVIDER_NOT_CONFIGURED
    assert "'solo'" in response.json()["detail"]
    assert "mistral" not in response.text.lower()


def test_a_policy_whose_candidates_all_lack_a_credential_is_refused_as_unconfigured(
    failover_client: TestClient,
) -> None:
    assert failover_client.post(f"{API_ROOT}/users", json={"user_id": "test-user"}, headers=_MASTER).status_code == 200
    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(side_effect=_provider_with_only_an_openai_key)):
        response = _chat(failover_client, "unconfigured_pair")
    assert response.status_code == 424, response.text
    assert response.headers[ERROR_CODE_HEADER] == PROVIDER_NOT_CONFIGURED
    body = response.json()
    assert body["code"] == PROVIDER_NOT_CONFIGURED
    assert "cohere, mistral" in body["detail"]
    assert _ANY_LLM_WORDING not in body["detail"]


# ---------------------------------------------------------------------------
# A provider's own SDK variable does not serve a provider nothing declares
# ---------------------------------------------------------------------------


def test_an_undeclared_provider_is_refused_although_its_variable_is_set(
    client: TestClient,
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
    test_user: dict[str, Any],
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-the-environment")
    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(side_effect=AssertionError)) as mock:
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": "anthropic:claude-sonnet-4-5", "messages": [{"role": "user", "content": "Hello"}]},
            headers=api_key_header,
        )
    _assert_provider_not_configured(response, "anthropic")
    assert "sk-ant-from-the-environment" not in response.text
    mock.assert_not_awaited()
    row = db_session.query(UsageLog).filter(UsageLog.api_key_id == api_key_obj["id"]).one()
    assert row.status_code == 424


@pytest.fixture
def alias_client(postgres_url: str) -> Generator[TestClient]:
    """A deployment declaring no provider, with an alias onto an undeclared one."""
    yield from build_test_client(
        GatewayConfig(
            database_url=postgres_url,
            master_key="test-master-key",
            auto_migrate=False,
            require_pricing=False,
            model_discovery=False,
            providers={},
            aliases={"house-model": "anthropic:claude-sonnet-4-5"},
        )
    )


def test_an_aliased_refusal_does_not_name_the_target(alias_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-the-environment")
    assert alias_client.post(f"{API_ROOT}/users", json={"user_id": "test-user"}, headers=_MASTER).status_code == 200
    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(side_effect=AssertionError)) as mock:
        response = alias_client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": "house-model", "messages": [{"role": "user", "content": "Hello"}], "user": "test-user"},
            headers=_MASTER,
        )
    assert response.status_code == 424, response.text
    assert response.headers[ERROR_CODE_HEADER] == PROVIDER_NOT_CONFIGURED
    assert "'house-model'" in response.text
    assert "anthropic" not in response.text.lower()
    mock.assert_not_awaited()


def test_an_undeclared_passthrough_provider_is_refused_although_its_variable_is_set(
    client: TestClient,
    api_key_header: dict[str, str],
    test_user: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COHERE_API_KEY", "co-from-the-environment")
    with patch("gateway.api.routes.embeddings.aembedding", new=AsyncMock(side_effect=AssertionError)) as mock:
        response = client.post(
            f"{API_ROOT}/embeddings",
            json={"model": "cohere:embed-english-v3.0", "input": "hello"},
            headers=api_key_header,
        )
    _assert_provider_not_configured(response, "cohere")
    mock.assert_not_awaited()


def test_an_undeclared_batch_provider_is_refused_although_its_variable_is_set(
    client: TestClient,
    api_key_header: dict[str, str],
    test_user: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-the-environment")
    response = client.get(f"{API_ROOT}/batches?provider=anthropic", headers=api_key_header)
    _assert_provider_not_configured(response, "anthropic")


def test_an_undeclared_keyless_local_provider_is_refused(
    client: TestClient,
    api_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    # The local server is the deployment's own, so only a declaration opens it to callers.
    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(side_effect=AssertionError)) as mock:
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": "ollama:llama3", "messages": [{"role": "user", "content": "Hello"}]},
            headers=api_key_header,
        )
    assert response.status_code == 424, response.text
    assert response.headers[ERROR_CODE_HEADER] == PROVIDER_NOT_CONFIGURED
    assert "'ollama'" in response.json()["detail"]
    mock.assert_not_awaited()


@pytest.fixture
def keyless_client(postgres_url: str) -> Generator[TestClient]:
    """A deployment declaring a local backend and a cloud-role provider, neither with a key."""
    yield from build_test_client(
        GatewayConfig(
            database_url=postgres_url,
            master_key="test-master-key",
            auto_migrate=False,
            require_pricing=False,
            model_discovery=False,
            providers={"ollama": {}, "bedrock": {}},
        )
    )


@pytest.mark.parametrize("model", ["ollama:llama3", "bedrock:anthropic.claude-sonnet-4-20250514-v1:0"])
def test_a_declared_keyless_or_ambient_provider_serves_with_no_key(keyless_client: TestClient, model: str) -> None:
    assert keyless_client.post(f"{API_ROOT}/users", json={"user_id": "test-user"}, headers=_MASTER).status_code == 200
    with patch(
        "gateway.api.routes.chat.acompletion", new=AsyncMock(side_effect=lambda **kw: _completion(kw["model"]))
    ) as mock:
        response = _chat(keyless_client, model)
    assert response.status_code == 200, response.text
    assert mock.await_args is not None
    assert "api_key" not in mock.await_args.kwargs


def test_a_declared_entry_naming_no_key_is_refused_although_its_variable_is_set(
    postgres_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-the-environment")
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        auto_migrate=False,
        require_pricing=False,
        model_discovery=False,
        providers={"anthropic": {}},
    )
    for bare_client in build_test_client(config):
        assert bare_client.post(f"{API_ROOT}/users", json={"user_id": "test-user"}, headers=_MASTER).status_code == 200
        with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(side_effect=AssertionError)) as mock:
            response = _chat(bare_client, "anthropic:claude-sonnet-4-5")
        _assert_provider_not_configured(response, "anthropic")
        assert "sk-ant-from-the-environment" not in response.text
        mock.assert_not_awaited()


def test_a_hosted_provider_serves(
    client: TestClient,
    api_key_header: dict[str, str],
    test_user: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())

    async def _lists_nothing(impl_name: str, **_: object) -> ProviderDiscovery:
        return ProviderDiscovery(provider=impl_name, models=[])

    monkeypatch.setattr("gateway.services.providers._hosted_provider_service.test_provider_credentials", _lists_nothing)
    stored = client.post(
        f"{API_ROOT}/hosted-providers",
        json={"provider": "anthropic", "api_key": "sk-ant-stored", "serves_unlisted_models": True},
        headers=_MASTER,
    )
    assert stored.status_code == 201, stored.text
    with patch(
        "gateway.api.routes.chat.acompletion", new=AsyncMock(side_effect=lambda **kw: _completion(kw["model"]))
    ) as mock:
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": "anthropic:claude-sonnet-4-5", "messages": [{"role": "user", "content": "Hello"}]},
            headers=api_key_header,
        )
    assert response.status_code == 200, response.text
    assert mock.await_args is not None
    assert mock.await_args.kwargs["api_key"] == "sk-ant-stored"


@pytest.fixture
def env_failover_client(postgres_url: str, monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient]:
    """A deployment declaring openai through interpolation, with mistral present only in the environment."""
    monkeypatch.setenv("MISTRAL_API_KEY", "mistral-from-the-environment")
    monkeypatch.setenv("OTARI_TEST_OPENAI_KEY", "sk-openai-interpolated")
    # Interpolated as config.yml is, so the entry is the ``${VAR}`` declaration an operator writes.
    config = GatewayConfig.model_validate(
        _resolve_env_vars(
            {
                "database_url": postgres_url,
                "master_key": "test-master-key",
                "auto_migrate": False,
                "require_pricing": False,
                "model_discovery": False,
                "providers": {"openai": {"api_key": "${OTARI_TEST_OPENAI_KEY}"}},
                "routing": {
                    "policies": {
                        "fast": {
                            "select": [{"default": "mistral:mistral-small-latest"}],
                            "on_failure": ["openai:gpt-5-mini"],
                        }
                    }
                },
            }
        )
    )
    yield from build_test_client(config)


def test_a_routed_plan_skips_an_environment_only_candidate(env_failover_client: TestClient) -> None:
    assert (
        env_failover_client.post(f"{API_ROOT}/users", json={"user_id": "test-user"}, headers=_MASTER).status_code == 200
    )
    with patch(
        "gateway.api.routes.chat.acompletion", new=AsyncMock(side_effect=lambda **kw: _completion(kw["model"]))
    ) as mock:
        response = _chat(env_failover_client, "fast")
    assert response.status_code == 200, response.text
    dispatched = [call.kwargs for call in mock.await_args_list]
    assert [kwargs["model"].split(":")[0] for kwargs in dispatched] == ["openai"]
    assert dispatched[0]["api_key"] == "sk-openai-interpolated"
