"""Resolving a selector to an owned provider endpoint, and what a dispatch to one is given."""

import asyncio
import uuid
from collections.abc import AsyncIterator, Iterator

import httpx
import pytest
from any_llm import LLMProvider
from any_llm.exceptions import AnyLLMError

from gateway.core.config import GatewayConfig
from gateway.services.provider_kwargs import ResolvedProvider, apply_endpoint_defaults, resolve_provider_selector
from gateway.services.providers import OwnedEndpoint, _provider_endpoint_cache, owned_endpoint_http_client
from gateway.services.providers._owned_endpoint_network import (
    OwnedEndpointAddressError,
    OwnedEndpointTransport,
    check_owned_endpoint_api_base,
)
from gateway.services.web_retrieval_network import PinnedTransportError

WORKSPACE = uuid.uuid4()
OTHER_WORKSPACE = uuid.uuid4()


def _endpoint(api_base: str = "https://1.1.1.1/v1", defaults: dict[str, object] | None = None) -> OwnedEndpoint:
    return OwnedEndpoint(
        id=uuid.uuid4(), provider="openai", api_base=api_base, api_key="sk-own", default_params=dict(defaults or {})
    )


@pytest.fixture(autouse=True)
def _cache(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(_provider_endpoint_cache, "_shared", {WORKSPACE: {"box": _endpoint("https://1.1.1.1/ws")}})
    monkeypatch.setattr(
        _provider_endpoint_cache, "_per_user", {WORKSPACE: {"alice": {"box": _endpoint("https://1.1.1.1/alice")}}}
    )
    yield


def _config(*, enabled: bool = True) -> GatewayConfig:
    return GatewayConfig(providers={}, provider_endpoints_enabled=enabled)


def _resolve(selector: str, user_id: str | None = "alice", **kwargs: object) -> ResolvedProvider:
    options: dict[str, object] = {"workspace_id": WORKSPACE, "owned_endpoints": True, **kwargs}
    return resolve_provider_selector(_config(), selector, user_id, **options)  # type: ignore[arg-type]


def test_a_users_endpoint_shadows_the_workspaces() -> None:
    resolved = _resolve("box:qwen3")
    assert resolved.instance == "box"
    assert resolved.dispatch_model == "openai:qwen3"
    assert resolved.kwargs == {"api_base": "https://1.1.1.1/alice", "api_key": "sk-own"}
    assert resolved.owned_endpoint is not None


def test_other_callers_in_the_workspace_reach_the_workspaces_endpoint() -> None:
    assert _resolve("box:qwen3", user_id="bob").kwargs["api_base"] == "https://1.1.1.1/ws"
    assert _resolve("box:qwen3", user_id=None).kwargs["api_base"] == "https://1.1.1.1/ws"


def test_another_workspace_does_not_see_it() -> None:
    with pytest.raises((ValueError, AnyLLMError)):
        _resolve("box:qwen3", workspace_id=OTHER_WORKSPACE)


def test_a_caller_that_does_not_opt_in_never_reaches_one() -> None:
    """Batches, embeddings and routing plans do not exempt budgets, so they must not resolve an endpoint."""
    with pytest.raises((ValueError, AnyLLMError)):
        _resolve("box:qwen3", owned_endpoints=False)


def test_nothing_resolves_while_the_deployment_has_endpoints_off() -> None:
    with pytest.raises((ValueError, AnyLLMError)):
        resolve_provider_selector(
            _config(enabled=False), "box:qwen3", "alice", workspace_id=WORKSPACE, owned_endpoints=True
        )


def test_a_configured_instance_keeps_its_name_over_an_endpoint_saved_before_it() -> None:
    config = GatewayConfig(
        providers={"box": {"provider_type": "openai", "api_key": "sk-op"}}, provider_endpoints_enabled=True
    )
    resolved = resolve_provider_selector(config, "box:qwen3", "alice", workspace_id=WORKSPACE, owned_endpoints=True)
    assert resolved.owned_endpoint is None
    assert resolved.kwargs["api_key"] == "sk-op"


def test_a_provider_name_keeps_its_meaning_over_an_endpoint_of_that_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_provider_endpoint_cache, "_shared", {WORKSPACE: {"openai": _endpoint()}})
    resolved = _resolve("openai:gpt-4o", user_id=None)
    assert resolved.owned_endpoint is None
    assert resolved.instance == "openai"


def test_a_keyless_endpoint_gets_the_placeholder_even_with_the_operators_env_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-operator")
    monkeypatch.setattr(
        _provider_endpoint_cache,
        "_shared",
        {
            WORKSPACE: {
                "box": OwnedEndpoint(id=uuid.uuid4(), provider="openai", api_base="https://1.1.1.1", api_key=None)
            }
        },
    )
    assert _resolve("box:qwen3", user_id=None).kwargs["api_key"] not in {None, "", "sk-operator"}


def _resolved(**defaults: object) -> ResolvedProvider:
    return ResolvedProvider(
        instance="box", provider=LLMProvider.OPENAI, model="m", kwargs={}, owned_endpoint=_endpoint(defaults=defaults)
    )


def test_defaults_only_fill_fields_the_call_does_not_set() -> None:
    call = apply_endpoint_defaults({"model": "openai:m", "temperature": 0.9}, _resolved(temperature=0.1, top_k=20))
    assert call["temperature"] == 0.9
    assert call["extra_body"] == {"top_k": 20}


def test_defaults_join_an_existing_extra_body_without_overriding_it() -> None:
    call = apply_endpoint_defaults({"extra_body": {"input": "x", "top_k": 1}}, _resolved(top_k=20, seed_hint=3))
    assert call["extra_body"] == {"seed_hint": 3, "input": "x", "top_k": 1}


def test_a_forbidden_default_is_dropped_at_dispatch_too() -> None:
    """A field added to the denylist after an endpoint was saved still never reaches the wire."""
    call = apply_endpoint_defaults({"model": "openai:m"}, _resolved(api_base="http://10.0.0.1", stream=True))
    assert "extra_body" not in call


def test_a_selector_that_is_not_an_endpoint_is_left_alone() -> None:
    plain = ResolvedProvider(instance="openai", provider=LLMProvider.OPENAI, model="m", kwargs={})
    assert apply_endpoint_defaults({"model": "openai:m"}, plain) == {"model": "openai:m"}


@pytest.mark.parametrize(
    "api_base",
    ["http://127.0.0.1:8000/v1", "http://[::ffff:169.254.169.254]/", "http://100.100.100.200/", "ftp://1.1.1.1/"],
)
def test_a_non_public_base_url_is_refused_when_saved(api_base: str) -> None:
    with pytest.raises(OwnedEndpointAddressError):
        asyncio.run(check_owned_endpoint_api_base(api_base))


def test_a_public_ip_literal_is_accepted_without_dns() -> None:
    asyncio.run(check_owned_endpoint_api_base("https://1.1.1.1/v1"))


def test_the_transport_refuses_a_private_destination_before_connecting() -> None:
    async def go() -> None:
        transport = OwnedEndpointTransport()
        try:
            with pytest.raises(PinnedTransportError):
                await transport.handle_async_request(httpx.Request("POST", "http://10.0.0.5/v1/chat/completions"))
        finally:
            await transport.aclose()

    asyncio.run(go())


def test_the_client_follows_no_redirect_and_is_shared_within_a_loop() -> None:
    async def go() -> None:
        client = owned_endpoint_http_client()
        assert client.follow_redirects is False
        assert owned_endpoint_http_client() is client

    asyncio.run(go())


def _streaming_transport(chunks: list[bytes], seen: list[httpx.Request]) -> OwnedEndpointTransport:
    async def body() -> AsyncIterator[bytes]:
        for chunk in chunks:
            yield chunk

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body())

    return OwnedEndpointTransport(transport_factory=lambda: httpx.MockTransport(handler))


def _active_leases(transport: OwnedEndpointTransport) -> int:
    return sum(entry.active_responses for entry in transport._pools.values())  # noqa: SLF001


def test_a_streamed_post_arrives_whole_and_gives_its_connection_back() -> None:
    chunks = [b'data: {"n": 1}\n\n', b'data: {"n": 2}\n\n', b"data: [DONE]\n\n"]

    async def go() -> None:
        seen: list[httpx.Request] = []
        transport = _streaming_transport(chunks, seen)
        async with httpx.AsyncClient(transport=transport) as client:
            async with client.stream("POST", "https://1.1.1.1/v1/chat/completions", json={"stream": True}) as response:
                assert _active_leases(transport) == 1
                received = [chunk async for chunk in response.aiter_raw()]
        assert b"".join(received) == b"".join(chunks)
        assert [request.method for request in seen] == ["POST"]
        assert _active_leases(transport) == 0

    asyncio.run(go())


def test_a_stream_closed_early_gives_its_connection_back() -> None:
    async def go() -> None:
        transport = _streaming_transport([b"data: 1\n\n"] * 50, [])
        async with httpx.AsyncClient(transport=transport) as client:
            async with client.stream("POST", "https://1.1.1.1/v1/chat/completions") as response:
                async for _ in response.aiter_raw():
                    break
        assert _active_leases(transport) == 0

    asyncio.run(go())
