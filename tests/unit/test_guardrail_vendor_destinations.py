"""An address an organization's guardrail definition names is dialed through the pinned client, where the vendor allows.

The definition service checks the address when it is saved. These tests re-point
the name between that check and the guardrail's own request, and stand in for the
socket to see what would have been dialed. watsonx and OpenAI moderation take an
HTTP client; the other vendors that name an address do not (see ``_create``).
"""

from __future__ import annotations

import base64
import ipaddress
import json
import time
from collections.abc import Callable, Sequence
from typing import Any

import httpx
import pytest
from any_guardrail import AnyGuardrail, GuardrailName

from gateway.services import url_safety, web_retrieval_network
from gateway.services.providers import OwnedEndpointSyncTransport
from gateway.services.tenancy.organization_guardrail_definition_service import _validate_argument_urls
from gateway.services.tenancy.organization_guardrail_runner import _create
from gateway.services.web_retrieval_network import PinnedAsyncHTTPTransport

PUBLIC = "93.184.216.34"


def _jwt() -> str:
    def part(value: dict[str, Any]) -> str:
        return base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")

    return f"{part({'alg': 'none'})}.{part({'exp': int(time.time()) + 3600})}.sig"


def _vendor(request: httpx.Request) -> httpx.Response:
    """Answers the way each vendor's service would, enough for its SDK to finish."""
    path = request.url.path
    if path.endswith("/identity/token"):
        return httpx.Response(200, json={"access_token": _jwt(), "expires_in": 3600, "refresh_token": "r"})
    if path.startswith("/v2/projects/"):
        entity = {
            "storage": {"type": "bmcos_object_storage", "guid": "s", "properties": {}},
            "compute": [{"type": "machine_learning", "guid": "g", "name": "n", "crn": "c"}],
        }
        return httpx.Response(200, json={"entity": entity, "metadata": {"guid": "p"}})
    if path.endswith("/text/detection"):
        return httpx.Response(200, json={"detections": []})
    if path.endswith("/moderations"):
        return httpx.Response(
            200,
            json={
                "id": "modr-1",
                "model": "omni-moderation-latest",
                "results": [
                    {"flagged": False, "categories": {"violence": False}, "category_scores": {"violence": 0.0}}
                ],
            },
        )
    return httpx.Response(404)


class _Socket:
    def __init__(self) -> None:
        self.dialed: list[httpx.Request] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.dialed.append(request)
        return _vendor(request)

    def hosts(self) -> list[str]:
        return [request.headers["host"] for request in self.dialed]


class _Resolver:
    """Connect-time answers per host; any host not named is public."""

    def __init__(self) -> None:
        self.answers: dict[str, str] = {}

    async def resolve(self, host: str, port: int) -> Sequence[Any]:
        return (ipaddress.ip_address(self.answers.get(host, PUBLIC)),)


@pytest.fixture
def socket(monkeypatch: pytest.MonkeyPatch) -> _Socket:
    fake = _Socket()
    monkeypatch.setattr(PinnedAsyncHTTPTransport, "_build_transport", lambda self: httpx.MockTransport(fake.handle))
    return fake


@pytest.fixture
def resolver(monkeypatch: pytest.MonkeyPatch) -> _Resolver:
    fake = _Resolver()

    async def resolve(self: object, host: str, port: int) -> Sequence[Any]:
        return await fake.resolve(host, port)

    monkeypatch.setattr(web_retrieval_network.SystemAddressResolver, "resolve", resolve)
    return fake


@pytest.fixture
def saved_as_public(monkeypatch: pytest.MonkeyPatch) -> None:
    async def public(_host: str) -> list[object]:
        return [ipaddress.ip_address(PUBLIC)]

    monkeypatch.setattr(url_safety, "_resolve_all_async", public)


# Each covered vendor: its name, the arguments a definition holds, and the host its address names.
VENDORS: dict[str, tuple[GuardrailName, dict[str, Any], str]] = {
    "watsonx_guardian": (
        GuardrailName.WATSONX_GUARDIAN,
        {"api_key": "wx-key", "url": "https://us-south.ml.cloud.ibm.com", "project_id": "p"},
        "us-south.ml.cloud.ibm.com",
    ),
    "openai_moderation": (
        GuardrailName.OPENAI_MODERATION,
        {"api_key": "sk-mod", "base_url": "https://moderation.example.com/v1"},
        "moderation.example.com",
    ),
}


def _evaluate(name: GuardrailName, arguments: dict[str, Any]) -> Any:
    return AnyGuardrail.evaluate(name, _create(name, dict(arguments)), "hello")


@pytest.mark.parametrize(("name", "arguments", "host"), VENDORS.values(), ids=VENDORS.keys())
def test_the_definitions_address_is_dialed_at_the_address_that_was_checked(
    name: GuardrailName, arguments: dict[str, Any], host: str, socket: _Socket, resolver: _Resolver
) -> None:
    verdict = _evaluate(name, arguments)

    assert verdict.valid is True
    sent = [request for request in socket.dialed if request.headers["host"] == host]
    assert sent, socket.hosts()
    assert {request.url.host for request in socket.dialed} == {PUBLIC}


@pytest.mark.parametrize("address", ["127.0.0.1", "::1", "10.0.0.5", "169.254.169.254"])
@pytest.mark.parametrize(("name", "arguments", "host"), VENDORS.values(), ids=VENDORS.keys())
@pytest.mark.asyncio
async def test_a_name_rebound_inside_after_the_definition_was_saved_is_refused(
    name: GuardrailName,
    arguments: dict[str, Any],
    host: str,
    address: str,
    socket: _Socket,
    resolver: _Resolver,
    saved_as_public: None,
) -> None:
    await _validate_argument_urls(arguments)
    resolver.answers[host] = address

    with pytest.raises(Exception) as refused:  # noqa: PT011 - each SDK wraps the transport error its own way
        await _run_in_a_thread(lambda: _evaluate(name, arguments))

    assert _refused_by_the_pinned_client(refused.value), repr(refused.value)
    assert host not in socket.hosts()


async def _run_in_a_thread(call: Callable[[], Any]) -> Any:
    """The runner calls a vendor from its own threads, never from the loop's."""
    import asyncio

    return await asyncio.get_running_loop().run_in_executor(None, call)


def _refused_by_the_pinned_client(exc: BaseException) -> bool:
    seen: BaseException | None = exc
    while seen is not None:
        if isinstance(seen, web_retrieval_network.PinnedTransportError) or "endpoint refused" in str(seen):
            return True
        seen = seen.__cause__ or seen.__context__
    return False


def test_an_address_from_the_operators_environment_is_left_to_the_vendor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only an address the definition names is the tenant's; one the environment supplies is the operator's."""
    guardrail = _create(GuardrailName.OPENAI_MODERATION, {"api_key": "sk-mod"})

    assert not isinstance(guardrail.client._client._transport, OwnedEndpointSyncTransport)  # type: ignore[attr-defined]  # noqa: SLF001


def test_the_sync_transport_reuses_one_loop_per_thread(socket: _Socket, resolver: _Resolver) -> None:
    with httpx.Client(transport=OwnedEndpointSyncTransport()) as client:
        first = client.get("https://moderation.example.com/v1/nothing-here")
        second = client.get("https://moderation.example.com/v1/nothing-here")

    assert (first.status_code, second.status_code) == (404, 404)
    assert len(socket.dialed) == 2
