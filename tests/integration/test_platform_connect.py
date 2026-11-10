"""Connecting a deployment to otari.ai, over HTTP, against a mocked otari.ai.

The mock answers the two endpoints `docs/platform-connect-protocol.md`
specifies and records every request it receives, which is also what the trust
rule is checked against. Model discovery is stubbed, so no case dials a provider.
"""

import json
import logging
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage
from any_llm.types.model import Model
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session
from sqlmodel import col

from gateway.api.deps import get_platform_client
from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.models.tenancy import DashboardSession, Organization, OrganizationMember, User
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, hash_session_token
from gateway.services.model_discovery_service import ProviderDiscovery
from gateway.services.providers import PlatformClient
from gateway.services.secret_box import generate_secret_key

from .conftest import build_test_client

PATH = f"{API_ROOT}/platform/connect"
HOSTED = f"{API_ROOT}/hosted-providers"
PLATFORM = "https://platform.test"
DEVICE_CODE = "device-code-never-shown"
ISSUED_KEY = "tk-issued-once-9876"
API_BASE = "https://inference.platform.test/api/v1"


class FakePlatform:
    """otari.ai's two connect endpoints, answering what a test queued."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.token_answers: list[tuple[int, dict[str, Any]]] = []
        self.interval = 5
        self.expires_in = 600
        # The shape otari.ai sends: a query and a fragment route, opened as given.
        self.verification = "https://otari.test/dashboard/?region=us#/connect?code=WXYZ-1234"

        self.token_headers: dict[str, str] = {}
        self.device_status = 200

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append((request.url.path, body))
        assert "authorization" not in request.headers
        if request.url.path == f"{API_ROOT}/connect/device":
            if self.device_status != 200:
                return httpx.Response(self.device_status, json={"detail": "otari.ai's own words"})
            return httpx.Response(
                200,
                json={
                    "device_code": DEVICE_CODE,
                    "user_code": "WXYZ-1234",
                    "verification_uri": "https://otari.test/connect",
                    "verification_uri_complete": self.verification,
                    "expires_in": self.expires_in,
                    "interval": self.interval,
                },
            )
        if request.url.path == f"{API_ROOT}/connect/token":
            status, answer = (
                self.token_answers.pop(0) if self.token_answers else (400, {"error": "authorization_pending"})
            )
            return httpx.Response(status, json=answer, headers=self.token_headers if status == 429 else {})
        return httpx.Response(404)

    def polls(self) -> int:
        return sum(1 for path, _ in self.requests if path.endswith("/connect/token"))


class Clock:
    def __init__(self) -> None:
        self.now = datetime.now(UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())


@pytest.fixture(autouse=True)
def _discovery(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Stub the model listing, recording the credential each listing was asked on."""
    dials: list[dict[str, Any]] = []

    async def _stub(impl_name: str, **kwargs: Any) -> ProviderDiscovery:
        dials.append({"implementation": impl_name, **kwargs})
        models = [Model(id=name, object="model", created=0, owned_by=impl_name) for name in ("openai:gpt-4o", "x:new")]
        return ProviderDiscovery(provider=impl_name, models=models)

    monkeypatch.setattr("gateway.services.providers._hosted_provider_service.test_provider_credentials", _stub)
    return dials


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    clock = Clock()
    monkeypatch.setattr("gateway.services.providers._platform_connect._now", clock)
    return clock


@pytest.fixture
def platform(client: TestClient) -> Iterator[FakePlatform]:
    fake = FakePlatform()
    app: Any = client.app
    app.dependency_overrides[get_platform_client] = lambda: PlatformClient(
        PLATFORM, transport=httpx.MockTransport(fake.handler)
    )
    yield fake
    app.dependency_overrides.pop(get_platform_client, None)


def _approve(platform: FakePlatform, api_key: str = ISSUED_KEY) -> None:
    platform.token_answers.append((200, {"api_key": api_key, "api_base": API_BASE}))


def _start(client: TestClient, headers: dict[str, str]) -> dict[str, Any]:
    response = client.post(PATH, headers=headers)
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


def _status(client: TestClient, headers: dict[str, str]) -> dict[str, Any]:
    response = client.get(PATH, headers=headers)
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


def _provider(client: TestClient, headers: dict[str, str]) -> dict[str, Any] | None:
    rows = client.get(HOSTED, headers=headers).json()["data"]
    return next((row for row in rows if row["name"] == "otari-ai"), None)


def test_nothing_is_sent_to_the_platform_until_an_operator_starts(
    client: TestClient, master_key_header: dict[str, str], platform: FakePlatform
) -> None:
    """The trust rule: booting, the bootstrap and reading the status all stay on this gateway."""
    assert client.get(f"{API_ROOT}/bootstrap").json()["platform_connect"] is True
    assert _status(client, master_key_header) == {
        "status": "idle",
        "user_code": None,
        "verification_uri_complete": None,
        "expires_in": None,
    }
    client.get(HOSTED, headers=master_key_header)
    client.delete(PATH, headers=master_key_header)
    assert platform.requests == []

    _start(client, master_key_header)
    assert [path for path, _ in platform.requests] == [f"{API_ROOT}/connect/device"]


def test_start_returns_the_code_and_link_and_never_the_device_code(
    client: TestClient, master_key_header: dict[str, str], platform: FakePlatform, clock: Clock
) -> None:
    response = client.post(PATH, headers=master_key_header)

    assert response.json() == {
        "user_code": "WXYZ-1234",
        "verification_uri_complete": "https://otari.test/dashboard/?region=us#/connect?code=WXYZ-1234",
        "expires_in": 600,
    }
    assert DEVICE_CODE not in response.text
    [(_, sent)] = platform.requests
    assert sent == {"gateway_name": "testserver"}

    pending = client.get(PATH, headers=master_key_header)
    assert pending.json()["status"] == "pending"
    assert pending.json()["user_code"] == "WXYZ-1234"
    assert DEVICE_CODE not in pending.text


def test_the_device_code_is_stored_encrypted(
    client: TestClient, master_key_header: dict[str, str], platform: FakePlatform, test_db: Session
) -> None:
    _start(client, master_key_header)

    stored = test_db.execute(text("SELECT encrypted_device_code FROM platform_connect_attempts")).scalar_one()
    assert stored and DEVICE_CODE not in stored


def test_a_second_start_while_pending_returns_the_same_attempt_without_asking_again(
    client: TestClient, master_key_header: dict[str, str], platform: FakePlatform, clock: Clock
) -> None:
    _start(client, master_key_header)
    clock.advance(100)

    again = _start(client, master_key_header)

    assert again["user_code"] == "WXYZ-1234"
    assert again["expires_in"] == 500
    assert len(platform.requests) == 1


def test_a_pending_read_polls_at_most_once_per_interval(
    client: TestClient, master_key_header: dict[str, str], platform: FakePlatform, clock: Clock
) -> None:
    _start(client, master_key_header)

    # Not due yet: the first poll waits one interval, as RFC 8628 has it.
    assert _status(client, master_key_header)["status"] == "pending"
    assert platform.polls() == 0

    clock.advance(5)
    assert _status(client, master_key_header)["status"] == "pending"
    assert _status(client, master_key_header)["status"] == "pending"
    assert platform.polls() == 1
    assert platform.requests[-1][1] == {"device_code": DEVICE_CODE}

    clock.advance(5)
    _status(client, master_key_header)
    assert platform.polls() == 2


def test_slow_down_adds_five_seconds_to_the_interval(
    client: TestClient, master_key_header: dict[str, str], platform: FakePlatform, clock: Clock
) -> None:
    _start(client, master_key_header)
    platform.token_answers.append((400, {"error": "slow_down"}))
    clock.advance(5)
    assert _status(client, master_key_header)["status"] == "pending"
    assert platform.polls() == 1

    clock.advance(9)
    _status(client, master_key_header)
    assert platform.polls() == 1

    clock.advance(1)
    _status(client, master_key_header)
    assert platform.polls() == 2


def test_a_rate_limited_poll_waits_out_retry_after_and_stays_pending(
    client: TestClient, master_key_header: dict[str, str], platform: FakePlatform, clock: Clock
) -> None:
    _start(client, master_key_header)
    platform.token_answers.append((429, {"detail": "Too many requests"}))
    platform.token_headers = {"Retry-After": "30"}
    clock.advance(5)
    assert _status(client, master_key_header)["status"] == "pending"
    assert platform.polls() == 1

    clock.advance(29)
    assert _status(client, master_key_header)["status"] == "pending"
    assert platform.polls() == 1

    _approve(platform)
    clock.advance(1)
    assert _status(client, master_key_header)["status"] == "connected"


@pytest.mark.parametrize("status", [422, 429, 503, 500])
def test_a_platform_that_refuses_to_start_is_reported_as_unavailable_without_its_body(
    client: TestClient, master_key_header: dict[str, str], platform: FakePlatform, status: int
) -> None:
    platform.device_status = status

    response = client.post(PATH, headers=master_key_header)

    assert response.status_code == 503
    assert response.json() == {"detail": "otari.ai could not be reached. Try again in a moment."}
    assert _status(client, master_key_header)["status"] == "idle"


def test_the_key_is_asked_for_once_and_a_long_gateway_name_is_cut_to_the_limit(
    client: TestClient, master_key_header: dict[str, str], platform: FakePlatform, clock: Clock
) -> None:
    app: Any = client.app
    app.state.config.public_base_url = f"https://{'g' * 150}.example.com"
    try:
        _start(client, master_key_header)
    finally:
        app.state.config.public_base_url = None
    assert len(platform.requests[0][1]["gateway_name"]) == 100

    _approve(platform)
    clock.advance(5)
    assert _status(client, master_key_header)["status"] == "connected"
    polls = platform.polls()
    for _ in range(3):
        clock.advance(30)
        _status(client, master_key_header)
    assert platform.polls() == polls


def test_approval_stores_the_provider_for_the_operators_organization_and_lists_its_models(
    client: TestClient,
    master_key_header: dict[str, str],
    platform: FakePlatform,
    clock: Clock,
    test_db: Session,
    _discovery: list[dict[str, Any]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    _start(client, master_key_header)
    _approve(platform)
    clock.advance(5)

    connected = client.get(PATH, headers=master_key_header)

    assert connected.json()["status"] == "connected"
    assert ISSUED_KEY not in connected.text
    provider = _provider(client, master_key_header)
    assert provider is not None
    assert provider["provider"] == "otari"
    # The origin: any-llm's otari implementation appends the API root itself.
    assert provider["api_base"] == "https://inference.platform.test"
    assert provider["api_key_last4"] == ISSUED_KEY[-4:]
    organization_id = test_db.query(Organization).filter(col(Organization.slug) == "default").one().id
    assert provider["restricted_to_organization_id"] == str(organization_id)
    assert _discovery[-1]["api_key"] == ISSUED_KEY
    assert _discovery[-1]["api_base"] == "https://inference.platform.test"

    models = client.get(f"{HOSTED}/otari-ai/models", headers=master_key_header).json()["data"]
    assert sorted(model["model"] for model in models) == ["openai:gpt-4o", "x:new"]

    row = test_db.execute(text("SELECT status, encrypted_device_code FROM platform_connect_attempts")).one()
    assert tuple(row) == ("connected", None)
    # Connected stays connected, and otari.ai is not asked again.
    polls = platform.polls()
    clock.advance(60)
    assert _status(client, master_key_header)["status"] == "connected"
    assert platform.polls() == polls
    for secret in (ISSUED_KEY, DEVICE_CODE):
        assert secret not in caplog.text


def test_connecting_again_replaces_the_credential_of_the_existing_provider(
    client: TestClient, master_key_header: dict[str, str], platform: FakePlatform, clock: Clock
) -> None:
    _start(client, master_key_header)
    _approve(platform)
    clock.advance(5)
    assert _status(client, master_key_header)["status"] == "connected"
    first = _provider(client, master_key_header)
    assert first is not None

    _start(client, master_key_header)
    _approve(platform, "tk-issued-again-5555")
    clock.advance(5)
    assert _status(client, master_key_header)["status"] == "connected"

    second = _provider(client, master_key_header)
    assert second is not None
    assert second["id"] == first["id"]
    assert second["api_key_last4"] == "5555"


def test_a_connected_status_reads_idle_once_the_provider_is_removed(
    client: TestClient, master_key_header: dict[str, str], platform: FakePlatform, clock: Clock
) -> None:
    _start(client, master_key_header)
    _approve(platform)
    clock.advance(5)
    assert _status(client, master_key_header)["status"] == "connected"

    assert client.delete(f"{HOSTED}/otari-ai", headers=master_key_header).status_code == 200

    assert _status(client, master_key_header)["status"] == "idle"


@pytest.mark.parametrize(("error", "status"), [("expired_token", "expired"), ("access_denied", "denied")])
def test_a_refusal_ends_the_attempt_and_a_new_start_asks_again(
    client: TestClient,
    master_key_header: dict[str, str],
    platform: FakePlatform,
    clock: Clock,
    error: str,
    status: str,
) -> None:
    _start(client, master_key_header)
    platform.token_answers.append((400, {"error": error}))
    clock.advance(5)

    assert _status(client, master_key_header) == {
        "status": status,
        "user_code": None,
        "verification_uri_complete": None,
        "expires_in": None,
    }
    assert _status(client, master_key_header)["status"] == status
    assert platform.polls() == 1
    assert _provider(client, master_key_header) is None

    _start(client, master_key_header)
    assert _status(client, master_key_header)["status"] == "pending"
    assert [path for path, _ in platform.requests].count(f"{API_ROOT}/connect/device") == 2


def test_an_attempt_past_its_lifetime_expires_without_asking(
    client: TestClient, master_key_header: dict[str, str], platform: FakePlatform, clock: Clock
) -> None:
    _start(client, master_key_header)
    clock.advance(601)

    assert _status(client, master_key_header)["status"] == "expired"
    assert platform.polls() == 0


def test_cancel_forgets_the_attempt_and_stops_polling(
    client: TestClient, master_key_header: dict[str, str], platform: FakePlatform, clock: Clock
) -> None:
    _start(client, master_key_header)

    assert client.delete(PATH, headers=master_key_header).status_code == 204

    clock.advance(5)
    assert _status(client, master_key_header)["status"] == "idle"
    assert platform.polls() == 0


def test_a_platform_that_fails_mid_poll_leaves_the_attempt_pending(
    client: TestClient, master_key_header: dict[str, str], platform: FakePlatform, clock: Clock
) -> None:
    _start(client, master_key_header)
    platform.token_answers.append((502, {}))
    platform.token_answers.append((400, {"error": "something_else"}))
    clock.advance(5)
    assert _status(client, master_key_header)["status"] == "pending"
    clock.advance(5)
    assert _status(client, master_key_header)["status"] == "pending"


@pytest.mark.parametrize(
    "verification",
    ["http://otari.test/connect", "javascript:alert(1)", "https://user:pass@otari.test/connect"],
)
def test_a_start_the_platform_answers_unusably_is_refused_without_upsell(
    client: TestClient, master_key_header: dict[str, str], platform: FakePlatform, verification: str
) -> None:
    platform.verification = verification

    response = client.post(PATH, headers=master_key_header)

    assert response.status_code == 503
    assert response.json() == {"detail": "otari.ai could not be reached. Try again in a moment."}
    assert _status(client, master_key_header)["status"] == "idle"


def test_a_key_for_an_address_that_is_not_https_is_dropped(
    client: TestClient, master_key_header: dict[str, str], platform: FakePlatform, clock: Clock
) -> None:
    _start(client, master_key_header)
    platform.token_answers.append((200, {"api_key": ISSUED_KEY, "api_base": "http://inference.platform.test/v1"}))
    clock.advance(5)

    response = client.get(PATH, headers=master_key_header)

    assert response.status_code == 503
    assert ISSUED_KEY not in response.text
    assert _provider(client, master_key_header) is None
    assert _status(client, master_key_header)["status"] == "expired"


def test_start_is_refused_without_a_secret_key_before_the_platform_is_asked(
    client: TestClient, master_key_header: dict[str, str], platform: FakePlatform, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OTARI_SECRET_KEY")

    response = client.post(PATH, headers=master_key_header)

    assert response.status_code == 500
    assert platform.requests == []


def test_start_is_rate_limited(client: TestClient, master_key_header: dict[str, str], platform: FakePlatform) -> None:
    for _ in range(10):
        assert client.post(PATH, headers=master_key_header).status_code == 200

    limited = client.post(PATH, headers=master_key_header)

    assert limited.status_code == 429
    assert "Retry-After" in limited.headers


def _member_cookie(session: Session) -> str:
    """A signed-in member of the default organization who is no operator."""
    organization_id = session.query(Organization).filter(col(Organization.slug) == "default").one().id
    user = User(email="member@example.com", full_name="Member", active_organization_id=organization_id)
    session.add(user)
    session.commit()
    session.refresh(user)
    session.add(OrganizationMember(organization_id=organization_id, user_id=user.id, role="member", status="active"))
    token = "otari-sess-member"
    session.add(
        DashboardSession(
            token_hash=hash_session_token(token),
            user_id=user.id,
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(hours=12),
        )
    )
    session.commit()
    return token


@pytest.mark.parametrize("method", ["POST", "GET", "DELETE"])
def test_only_an_operator_reaches_the_routes(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    platform: FakePlatform,
    test_db: Session,
    method: str,
) -> None:
    # Provisions the default organization.
    client.get(HOSTED, headers=master_key_header)

    assert client.request(method, PATH).status_code == 401
    assert client.request(method, PATH, headers=api_key_header).status_code == 401
    client.cookies.set(SESSION_COOKIE_NAME, _member_cookie(test_db))
    try:
        assert client.request(method, PATH).status_code == 404
    finally:
        client.cookies.clear()
    assert platform.requests == []


def test_an_empty_platform_url_removes_the_routes_and_the_bootstrap_flag(
    test_config: GatewayConfig, clean_database: None, master_key_header: dict[str, str]
) -> None:
    config = test_config.model_copy(update={"platform_url": ""})
    for client in build_test_client(config):
        assert client.get(f"{API_ROOT}/bootstrap").json()["platform_connect"] is False
        for method in ("POST", "GET", "DELETE"):
            assert client.request(method, PATH, headers=master_key_header).status_code == 404


def test_on_a_default_configuration_a_connected_model_serves_a_request_at_once(
    postgres_url: str, clean_database: None, clock: Clock
) -> None:
    """Zero configuration: pricing is required, the models.dev defaults are off, and no setting is touched."""
    config = GatewayConfig(database_url=postgres_url, master_key="test-master-key", auto_migrate=False)
    assert (config.require_pricing, config.default_pricing) == (True, False)
    headers = {API_KEY_HEADER: "Bearer test-master-key"}
    fake = FakePlatform()
    for client in build_test_client(config):
        app: Any = client.app
        app.dependency_overrides[get_platform_client] = lambda: PlatformClient(
            PLATFORM, transport=httpx.MockTransport(fake.handler)
        )
        _start(client, headers)
        _approve(fake)
        clock.advance(5)
        assert _status(client, headers)["status"] == "connected"

        models = client.get(f"{HOSTED}/otari-ai/models", headers=headers).json()["data"]
        by_name = {model["model"]: model for model in models}
        assert by_name["openai:gpt-4o"]["enabled"] is True
        assert by_name["openai:gpt-4o"]["price_source"] == "defaults"
        assert by_name["openai:gpt-4o"]["input_price_per_million"] is not None
        # No vendor entry prices it, so it waits for a rate as on any hosted provider.
        assert (by_name["x:new"]["enabled"], by_name["x:new"]["disabled_reason"]) == (False, "unpriced")

        assert client.post(f"{API_ROOT}/users", json={"user_id": "u1"}, headers=headers).status_code == 200
        completion = ChatCompletion(
            id="cmpl-1",
            choices=[
                Choice(finish_reason="stop", index=0, message=ChatCompletionMessage(role="assistant", content="hi"))
            ],
            created=0,
            model="openai:gpt-4o",
            object="chat.completion",
            usage=CompletionUsage(prompt_tokens=1000, completion_tokens=1000, total_tokens=2000),
        )
        provider = AsyncMock(return_value=completion)
        with patch("gateway.api.routes.chat.acompletion", new=provider):
            response = client.post(
                f"{API_ROOT}/chat/completions",
                json={"model": "otari-ai:openai:gpt-4o", "messages": [{"role": "user", "content": "hi"}], "user": "u1"},
                headers=headers,
            )

        assert response.status_code == 200, response.text
        assert float(response.json()["usage"]["cost_usd"]) > 0
        assert provider.await_args is not None
        assert provider.await_args.kwargs["api_key"] == ISSUED_KEY


def test_the_toggle_still_decides_for_every_other_hosted_provider(
    postgres_url: str, clean_database: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _listing(impl_name: str, **_: object) -> ProviderDiscovery:
        return ProviderDiscovery(
            provider=impl_name, models=[Model(id="gpt-4o", object="model", created=0, owned_by=impl_name)]
        )

    monkeypatch.setattr("gateway.services.providers._hosted_provider_service.test_provider_credentials", _listing)
    config = GatewayConfig(database_url=postgres_url, master_key="test-master-key", auto_migrate=False)
    headers = {API_KEY_HEADER: "Bearer test-master-key"}
    for client in build_test_client(config):
        created = client.post(HOSTED, json={"provider": "openai", "api_key": "sk-live-1234"}, headers=headers)
        assert created.status_code == 201, created.text
        [model] = client.get(f"{HOSTED}/openai/models", headers=headers).json()["data"]
        assert (model["enabled"], model["disabled_reason"]) == (False, "unpriced")
