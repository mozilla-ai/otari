"""OAuth request limits and generated protocol contracts."""

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Annotated, Any, cast

import pytest
from fastapi import APIRouter, FastAPI, Form
from pydantic import ValidationError
from starlette.types import Message, Scope

from gateway.api.routes.identity_apps import OAuthAppRoute
from gateway.core.oauth_apps import parse_oauth_redirect
from gateway.core.settings.identity import OAuthAppConfig
from gateway.rate_limit import RateLimiter
from gateway.schemas.identity.oauth_apps import AppTokenRequest


def _script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generate_openapi = _script("generate_openapi")


@pytest.mark.parametrize("host", ["foo^bar.example", "foo|bar.example", "foo<bar.example", "foo>bar.example"])
def test_registration_rejects_hosts_a_browser_cannot_open(host: str) -> None:
    uri = f"https://{host}/callback"
    assert parse_oauth_redirect(uri) is None
    with pytest.raises(ValidationError):
        OAuthAppConfig(name="App", redirect_uris=[uri])


def _token_app() -> FastAPI:
    app = FastAPI()
    router = APIRouter(route_class=OAuthAppRoute)

    @router.post("/token")
    async def exchange(body: Annotated[AppTokenRequest, Form()]) -> AppTokenRequest:
        return body

    app.include_router(router)
    return app


async def _stream_request(
    app: FastAPI, chunks: list[bytes], content_length: str | None = None
) -> tuple[list[Message], int]:
    headers = [(b"content-type", b"application/x-www-form-urlencoded")]
    if content_length is not None:
        headers.append((b"content-length", content_length.encode()))
    scope: Scope = {
        "type": "http",
        "method": "POST",
        "path": "/token",
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 1234),
    }
    received = 0
    sent: list[Message] = []

    async def receive() -> Message:
        nonlocal received
        body = chunks[received]
        received += 1
        return {"type": "http.request", "body": body, "more_body": received < len(chunks)}

    async def send(message: Message) -> None:
        sent.append(message)

    await app(scope, receive, send)
    return sent, received


@pytest.mark.asyncio
@pytest.mark.parametrize("content_length", [None, "1"])
async def test_token_limit_stops_the_stream_without_trusting_content_length(content_length: str | None) -> None:
    sent, received = await _stream_request(
        _token_app(), [b"code=" + b"x" * 16000, b"x" * 17000, b"unread"], content_length
    )
    assert sent[0]["status"] == 413
    assert received == 2


@pytest.mark.asyncio
async def test_declared_oversize_token_body_is_rejected_before_reading() -> None:
    sent, received = await _stream_request(_token_app(), [b"unread"], "32769")
    assert sent[0]["status"] == 413
    assert received == 0


@pytest.mark.asyncio
async def test_throttled_token_requests_are_rejected_before_reading_or_parsing() -> None:
    app = _token_app()
    app.state.public_auth_rate_limiter = RateLimiter(1)
    app.state.public_auth_rate_limiter.check("127.0.0.1")
    sent, received = await _stream_request(app, [b"unread"], "999999999")
    assert sent[0]["status"] == 429
    assert received == 0


@pytest.fixture(scope="module")
def protocol_spec() -> dict[str, Any]:
    return cast(dict[str, Any], generate_openapi.generate_openapi_spec())


def test_openapi_distinguishes_public_protocol_from_protected_consent(protocol_spec: dict[str, Any]) -> None:
    paths = protocol_spec["paths"]
    for path, method in (("authorize", "get"), ("clients", "get"), ("token", "post"), ("revoke", "post")):
        assert paths[f"/api/v1/oauth/apps/{path}"][method]["security"] == []
    for path, method in (("authorize", "post"), ("workspaces", "get")):
        assert paths[f"/api/v1/oauth/apps/{path}"][method]["security"]
    redirect = paths["/api/v1/oauth/apps/authorize"]["get"]["responses"]
    assert "200" not in redirect
    assert "content" not in redirect["302"]
    assert redirect["302"]["headers"]["Location"]["schema"]["type"] == "string"
    assert redirect["400"]["content"]["application/json"]["schema"]["$ref"].endswith("/OAuthErrorResponse")
