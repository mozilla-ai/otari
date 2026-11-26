"""``cors_inference_origins`` opens the model, web tool and OAuth app token endpoints to a browser, and nothing else.

A page on any origin may call a model with a key it holds, the way curl already
can, and an app's page may exchange the code it was sent for that key. What
these tests pin is what it may not do: send credentials, reach any other route,
approve its own grant, or take over the dashboard's own CORS policy.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute, iter_route_contexts
from fastapi.testclient import TestClient
from pydantic import ValidationError

from gateway.core.config import API_ROOT, DEPLOYMENT_PAID_HEADER, GatewayConfig
from gateway.main import _INFERENCE_CORS_PATHS, create_app

MASTER_KEY = "sk-test-master"
DASHBOARD = "https://dashboard.example.com"
PAGE = "https://hackathon-team.example.net"
CHAT = f"{API_ROOT}/chat/completions"
BODY = {"model": "openai:gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}


def _config(tmp_path: Path, **overrides: object) -> GatewayConfig:
    return GatewayConfig(
        database_url=f"sqlite:///{tmp_path / 'cors.db'}",
        master_key=MASTER_KEY,
        require_pricing=False,
        **overrides,  # type: ignore[arg-type]
    )


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    with TestClient(create_app(_config(tmp_path, cors_inference_origins=["*"]))) as client:
        yield client


def _preflight(client: TestClient, path: str, origin: str, method: str = "POST") -> dict[str, str]:
    response = client.options(
        path,
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": method,
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )
    return {name.lower(): value for name, value in response.headers.items()} | {"status": str(response.status_code)}


def test_any_page_may_call_a_model_without_credentials(client: TestClient) -> None:
    preflight = _preflight(client, CHAT, PAGE)

    assert preflight["status"] == "200"
    assert preflight["access-control-allow-origin"] == "*"
    assert "authorization" in preflight["access-control-allow-headers"].lower()
    assert "access-control-allow-credentials" not in preflight


def test_a_header_the_client_adds_does_not_fail_the_preflight(client: TestClient) -> None:
    response = client.options(
        CHAT,
        headers={
            "Origin": PAGE,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type,x-app-version",
        },
    )

    assert response.status_code == 200
    assert "x-app-version" in response.headers["access-control-allow-headers"].lower()
    assert "access-control-allow-credentials" not in response.headers


def test_a_refusal_reaches_the_page_so_it_can_show_why(client: TestClient) -> None:
    refused = client.post(CHAT, json=BODY, headers={"Origin": PAGE})

    assert refused.status_code == 401
    assert refused.headers["access-control-allow-origin"] == "*"
    assert "access-control-allow-credentials" not in refused.headers


def test_a_page_may_read_who_paid_for_a_completion(client: TestClient) -> None:
    answered = client.post(CHAT, json=BODY, headers={"Origin": PAGE})

    assert DEPLOYMENT_PAID_HEADER.lower() in answered.headers["access-control-expose-headers"].lower()


def test_a_page_sending_a_cookie_is_still_not_allowed_credentials(client: TestClient) -> None:
    sent = client.post(CHAT, json=BODY, headers={"Origin": PAGE, "Cookie": "otari_session=anything"})

    assert "access-control-allow-credentials" not in sent.headers


@pytest.mark.parametrize(
    ("path", "method"),
    [
        (f"{API_ROOT}/keys", "POST"),
        (f"{API_ROOT}/settings", "GET"),
        (f"{API_ROOT}/users", "GET"),
        (f"{API_ROOT}/auth/session", "POST"),
        (f"{API_ROOT}/pricing", "GET"),
        (f"{API_ROOT}/models/discoverable", "GET"),
        (f"{API_ROOT}/models/metadata", "GET"),
        (f"{API_ROOT}/search-tools", "GET"),
        (f"{API_ROOT}/oauth/apps/workspaces", "GET"),
    ],
)
def test_no_other_route_is_opened(client: TestClient, path: str, method: str) -> None:
    assert "access-control-allow-origin" not in _preflight(client, path, PAGE, method)


@pytest.mark.parametrize("path", [f"{API_ROOT}/audio/speech", f"{API_ROOT}/images/generations", f"{API_ROOT}/messages"])
def test_the_other_model_endpoints_are_opened_too(client: TestClient, path: str) -> None:
    assert _preflight(client, path, PAGE)["access-control-allow-origin"] == "*"


@pytest.mark.parametrize("path", [f"{API_ROOT}/models", f"{API_ROOT}/models/openai:gpt-4o-mini", f"{API_ROOT}/tools"])
def test_a_page_may_read_the_models_and_tools_it_can_offer(client: TestClient, path: str) -> None:
    assert _preflight(client, path, PAGE, "GET")["access-control-allow-origin"] == "*"


@pytest.mark.parametrize(
    "path", [f"{API_ROOT}/search", f"{API_ROOT}/search/brave", f"{API_ROOT}/fetch", f"{API_ROOT}/fetch/jina"]
)
def test_a_page_may_search_and_fetch_with_its_key(client: TestClient, path: str) -> None:
    preflight = _preflight(client, path, PAGE)

    assert preflight["access-control-allow-origin"] == "*"
    assert "access-control-allow-credentials" not in preflight


def test_an_app_page_may_exchange_and_revoke_its_token_but_never_approve(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        cors_inference_origins=["*"],
        oauth_apps={"app": {"name": "App", "redirect_uris": [f"{PAGE}/callback"]}},
    )
    with TestClient(create_app(config)) as client:
        token = _preflight(client, f"{API_ROOT}/oauth/apps/token", PAGE)
        revoke = _preflight(client, f"{API_ROOT}/oauth/apps/revoke", PAGE)
        approve = _preflight(client, f"{API_ROOT}/oauth/apps/authorize", PAGE)

    assert token["access-control-allow-origin"] == "*"
    assert revoke["access-control-allow-origin"] == "*"
    assert "access-control-allow-credentials" not in token
    assert "access-control-allow-origin" not in approve


def test_a_signed_in_dashboard_session_cannot_be_used_from_the_page(client: TestClient) -> None:
    signed_in = client.post(f"{API_ROOT}/auth/session", json={"master_key": MASTER_KEY})
    assert signed_in.status_code == 200, signed_in.text

    sent = client.post(CHAT, json=BODY, headers={"Origin": PAGE, "Sec-Fetch-Site": "cross-site"})

    assert sent.status_code == 401


def test_the_dashboard_keeps_its_credentialed_policy_beside_it(tmp_path: Path) -> None:
    config = _config(tmp_path, cors_allow_origins=[DASHBOARD], cors_inference_origins=["*"])
    with TestClient(create_app(config)) as client:
        dashboard = _preflight(client, CHAT, DASHBOARD)
        page = _preflight(client, CHAT, PAGE)
        page_on_settings = _preflight(client, f"{API_ROOT}/settings", PAGE, "GET")

    assert dashboard["access-control-allow-origin"] == DASHBOARD
    assert dashboard["access-control-allow-credentials"] == "true"
    assert page["access-control-allow-origin"] == "*"
    assert "access-control-allow-credentials" not in page
    assert "access-control-allow-origin" not in page_on_settings


def test_the_dashboard_policy_keeps_its_own_header_list(tmp_path: Path) -> None:
    config = _config(tmp_path, cors_allow_origins=[DASHBOARD], cors_inference_origins=["*"])
    with TestClient(create_app(config)) as client:
        response = client.options(
            CHAT,
            headers={
                "Origin": DASHBOARD,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "x-app-version",
            },
        )

    assert response.status_code == 400


def test_a_named_origin_is_opened_and_no_other(tmp_path: Path) -> None:
    with TestClient(create_app(_config(tmp_path, cors_inference_origins=[PAGE]))) as client:
        listed = _preflight(client, CHAT, PAGE)
        other = _preflight(client, CHAT, "https://elsewhere.example.org")

    assert listed["access-control-allow-origin"] == PAGE
    assert "access-control-allow-credentials" not in listed
    assert "access-control-allow-origin" not in other


def test_a_wildcard_dashboard_list_keeps_every_origin_on_the_model_endpoints(tmp_path: Path) -> None:
    config = _config(tmp_path, cors_allow_origins=["*"], cors_inference_origins=[PAGE])
    with TestClient(create_app(config)) as client:
        preflight = _preflight(client, CHAT, "https://elsewhere.example.org")
        response = client.post(CHAT, json=BODY, headers={"Origin": "https://elsewhere.example.org"})

    assert preflight["access-control-allow-origin"] == "*"
    assert "PUT" in preflight["access-control-allow-methods"]
    assert response.headers["access-control-allow-origin"] == "*"
    assert "access-control-allow-credentials" not in response.headers


def test_with_neither_list_set_no_route_answers_cors(tmp_path: Path) -> None:
    with TestClient(create_app(_config(tmp_path))) as client:
        assert "access-control-allow-origin" not in _preflight(client, CHAT, PAGE)


def test_each_opened_path_is_a_route_the_gateway_serves(tmp_path: Path) -> None:
    oauth_apps = {"app": {"name": "App", "redirect_uris": [f"{PAGE}/callback"]}}
    app: FastAPI = create_app(_config(tmp_path, oauth_apps=oauth_apps))
    served = {
        context.path for context in iter_route_contexts(app.routes) if isinstance(context.original_route, APIRoute)
    }

    assert _INFERENCE_CORS_PATHS <= served, sorted(_INFERENCE_CORS_PATHS - served)


@pytest.mark.parametrize(
    "origin",
    [
        "https://app.example.com/",
        "https://app.example.com/path",
        "app.example.com",
        "https://user@app.example.com",
        "https://app.example.com:443",
        "http://localhost:80",
        "https://App.Example.com",
        "https://bücher.example",
        "https://*.example.com",
        "https://app.example.com:",
        "https://app.example.com:99999",
        "https://app.example.com?page=1",
        "ftp://app.example.com",
        "https://",
    ],
)
def test_an_entry_that_is_not_an_origin_is_refused_at_load(tmp_path: Path, origin: str) -> None:
    with pytest.raises(ValidationError, match="is not an origin"):
        _config(tmp_path, cors_inference_origins=[origin])


def test_a_wildcard_and_a_port_are_accepted(tmp_path: Path) -> None:
    origins = ["*", "http://localhost:5173", "https://app.example.com:8443", "http://[::1]:3000"]

    assert _config(tmp_path, cors_inference_origins=origins).cors_inference_origins == origins
