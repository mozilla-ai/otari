"""Unit tests for what a vLLM or SGLang server says about the models it serves, and how a listing uses it."""

import uuid
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from any_llm.types.model import Model

from gateway.core.config import GatewayConfig
from gateway.core.unit_of_work import create_unit_of_work
from gateway.services.catalog import _served
from gateway.services.catalog._limits import ModelLimits, model_capabilities, model_limits, served_context_window
from gateway.services.catalog._served import (
    EngineSettings,
    KeyOwner,
    ServedFacts,
    facts_from_listing,
    read_engine_settings,
    record_served_models,
    served_facts,
)
from gateway.services.model_discovery_service import ProviderDiscovery, discover_models_with_status, get_model_cache
from gateway.types.models_dev import parse_catalog

MODELS_DEV = parse_catalog(
    {
        "huggingface": {
            "models": {
                "Qwen/Qwen3-32B": {
                    "name": "Qwen3 32B",
                    "tool_call": True,
                    "reasoning": True,
                    "structured_output": True,
                    "modalities": {"input": ["text"], "output": ["text"]},
                    "limit": {"context": 524288, "output": 300000},
                },
            },
        },
    }
)


def _vllm(model_id: str = "cairn-qwen", **extra: Any) -> Model:
    return Model(id=model_id, object="model", created=0, owned_by="vllm", **extra)


def _sglang(model_id: str = "cairn-qwen", **extra: Any) -> Model:
    return Model(id=model_id, object="model", created=0, owned_by="sglang", **extra)


def _config() -> GatewayConfig:
    return GatewayConfig(master_key="test", providers={"cairn": {"provider_type": "vllm", "api_base": "http://x/v1"}})


@pytest.fixture(autouse=True)
def _empty_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_served, "_served", {})


def _serve(provider: str, model_id: str, facts: ServedFacts, organization_id: uuid.UUID | None = None) -> None:
    _served._served.setdefault((organization_id, provider), {})[model_id] = facts


# --------------------------------------------------------------------------- #
# Reading a listing
# --------------------------------------------------------------------------- #


def test_a_vllm_listing_gives_the_served_context_and_the_loaded_model() -> None:
    facts = facts_from_listing(_vllm(max_model_len=262144, root="Qwen/Qwen3-32B"), None)

    assert facts == ServedFacts(engine="vllm", context_window=262144, root="Qwen/Qwen3-32B")


def test_a_listing_from_anything_else_says_nothing() -> None:
    openai = Model.model_validate(
        {"id": "gpt-4o", "object": "model", "created": 0, "owned_by": "openai", "max_model_len": 128000}
    )
    # The declared ``models:`` fallback stamps the instance name, which may be ``vllm``, and adds no fields.
    declared = _vllm()

    assert facts_from_listing(openai, None) is None
    assert facts_from_listing(declared, None) is None


def test_a_context_length_that_is_not_a_positive_integer_is_left_unknown() -> None:
    for value in (0, -1, True, "262144", 2.5):
        facts = facts_from_listing(_vllm(max_model_len=value, root="r"), None)
        assert facts is not None and facts.context_window is None


def test_sglang_settings_make_a_missing_parser_mean_off() -> None:
    settings = EngineSettings(reasoning_parser="qwen3", tool_call_parser=None, input_modalities=("text", "image"))

    facts = facts_from_listing(_sglang(max_model_len=131072), settings)

    assert facts == ServedFacts(
        engine="sglang",
        context_window=131072,
        settings_reported=True,
        reasoning_parser="qwen3",
        input_modalities=("text", "image"),
    )


def test_settings_never_attach_to_a_vllm_listing() -> None:
    settings = EngineSettings(reasoning_parser="qwen3", tool_call_parser="hermes", input_modalities=None)

    facts = facts_from_listing(_vllm(max_model_len=1), settings)

    assert facts is not None and not facts.settings_reported


# --------------------------------------------------------------------------- #
# Asking SGLang how it was started
# --------------------------------------------------------------------------- #


def _route_http(monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]) -> None:
    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def factory(*_args: object, **kwargs: object) -> httpx.AsyncClient:
        assert kwargs.get("follow_redirects") is False
        return real_async_client(transport=transport)

    monkeypatch.setattr("gateway.services.catalog._served.httpx.AsyncClient", factory)


@pytest.mark.asyncio
async def test_sglang_model_info_is_read_beside_v1_with_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "reasoning_parser": "deepseek-r1",
                "tool_call_parser": "qwen25",
                "has_image_understanding": True,
                "has_audio_understanding": False,
            },
        )

    _route_http(monkeypatch, handler)

    settings = await read_engine_settings([_sglang(max_model_len=1)], api_base="http://box:30000/v1/", api_key="sk-x")

    assert settings == EngineSettings(
        reasoning_parser="deepseek-r1", tool_call_parser="qwen25", input_modalities=("text", "image")
    )
    assert [str(request.url) for request in seen] == ["http://box:30000/model_info"]
    assert seen[0].headers["Authorization"] == "Bearer sk-x"


@pytest.mark.asyncio
async def test_an_older_sglang_is_asked_on_its_deprecated_path(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/model_info":
            return httpx.Response(404)
        return httpx.Response(200, json={"reasoning_parser": None, "tool_call_parser": None})

    _route_http(monkeypatch, handler)

    settings = await read_engine_settings([_sglang(root="r")], api_base="http://box:30000/v1", api_key=None)

    assert settings == EngineSettings(reasoning_parser=None, tool_call_parser=None, input_modalities=None)


@pytest.mark.asyncio
async def test_a_server_that_will_not_say_reports_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    _route_http(monkeypatch, lambda _request: httpx.Response(500))

    assert await read_engine_settings([_sglang(root="r")], api_base="http://box/v1", api_key=None) is None


@pytest.mark.asyncio
async def test_an_oversized_model_info_is_not_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """An organization's key points at a server its tenant runs, so its answer is bounded."""
    padding = "x" * (_served._MAX_MODEL_INFO_BYTES + 1)
    _route_http(monkeypatch, lambda _request: httpx.Response(200, json={"tool_call_parser": "qwen25", "pad": padding}))

    assert await read_engine_settings([_sglang(root="r")], api_base="http://box/v1", api_key=None) is None


@pytest.mark.asyncio
async def test_only_an_sglang_server_is_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("asked a server that is not SGLang")

    _route_http(monkeypatch, handler)

    assert await read_engine_settings([_vllm(max_model_len=1)], api_base="http://box/v1", api_key=None) is None
    assert await read_engine_settings([_sglang(root="r")], api_base=None, api_key=None) is None


# --------------------------------------------------------------------------- #
# What a listing shows
# --------------------------------------------------------------------------- #


def test_the_served_context_beats_models_dev_and_caps_the_output() -> None:
    """Cairn's case: models.dev says 512k, the server runs the model at 256k."""
    _serve("cairn", "cairn-qwen", ServedFacts(engine="vllm", context_window=262144, root="Qwen/Qwen3-32B"))

    assert model_limits(_config(), MODELS_DEV, "cairn:cairn-qwen") == ModelLimits(262144, 262144)


def test_a_served_model_is_found_on_models_dev_by_what_the_server_loaded() -> None:
    _serve("cairn", "cairn-qwen", ServedFacts(engine="vllm", root="Qwen/Qwen3-32B"))

    capabilities = model_capabilities(_config(), MODELS_DEV, "cairn:cairn-qwen")

    assert capabilities is not None
    assert capabilities.name == "Qwen3 32B"
    assert capabilities.supported_parameters == ["tools", "tool_choice", "response_format", "reasoning"]
    assert model_limits(_config(), MODELS_DEV, "cairn:cairn-qwen") == ModelLimits(524288, 300000)


def test_an_unserved_model_is_not_looked_up_by_hugging_face_id() -> None:
    assert model_capabilities(_config(), MODELS_DEV, "cairn:Qwen/Qwen3-32B") is None


def test_sglang_without_a_tool_parser_offers_no_tools() -> None:
    _serve(
        "cairn",
        "cairn-qwen",
        ServedFacts(engine="sglang", root="Qwen/Qwen3-32B", settings_reported=True, input_modalities=("text", "image")),
    )

    capabilities = model_capabilities(_config(), MODELS_DEV, "cairn:cairn-qwen")

    assert capabilities is not None
    assert capabilities.supported_parameters == ["response_format", "reasoning"]
    assert capabilities.input_modalities == ["text", "image"]


def test_sglang_settings_describe_a_model_models_dev_does_not_know() -> None:
    _serve(
        "cairn",
        "custom",
        ServedFacts(engine="sglang", settings_reported=True, reasoning_parser="qwen3", tool_call_parser="qwen25"),
    )

    capabilities = model_capabilities(_config(), MODELS_DEV, "cairn:custom")

    assert capabilities is not None
    assert capabilities.name is None
    assert capabilities.supported_parameters == ["tools", "tool_choice", "reasoning"]
    assert capabilities.input_modalities == ["text"]


ACME = uuid.UUID(int=1)
GLOBEX = uuid.UUID(int=2)


def test_each_organization_reads_its_own_key_under_a_shared_prefix() -> None:
    _serve("byo", "qwen", ServedFacts(engine="vllm", context_window=262144), ACME)
    _serve("byo", "qwen", ServedFacts(engine="vllm", context_window=131072), GLOBEX)

    assert model_limits(_config(), None, "byo:qwen", ACME).context_window == 262144
    assert model_limits(_config(), None, "byo:qwen", GLOBEX).context_window == 131072
    # No organization, or one with no key there, reads only the deployment's.
    assert model_limits(_config(), None, "byo:qwen").context_window is None
    assert model_limits(_config(), None, "byo:qwen", uuid.UUID(int=3)).context_window is None


def test_an_organization_key_beats_a_hosted_provider_of_its_prefix() -> None:
    _serve("byo", "qwen", ServedFacts(engine="vllm", context_window=65536))
    _serve("byo", "qwen", ServedFacts(engine="vllm", context_window=262144), ACME)

    assert model_limits(_config(), None, "byo:qwen", ACME).context_window == 262144
    assert model_limits(_config(), None, "byo:qwen", GLOBEX).context_window == 65536


def test_a_configured_instance_is_never_read_from_an_organization_key() -> None:
    """A request to a configured instance's prefix never reaches an organization key, so neither does its listing."""
    _serve("cairn", "cairn-qwen", ServedFacts(engine="vllm", context_window=65536))
    _serve("cairn", "cairn-qwen", ServedFacts(engine="vllm", context_window=262144), ACME)

    assert model_limits(_config(), None, "cairn:cairn-qwen", ACME).context_window == 65536


def test_the_trace_view_reads_the_served_context() -> None:
    _serve("cairn", "cairn-qwen", ServedFacts(engine="vllm", context_window=262144))

    assert served_context_window("cairn", "cairn-qwen") == 262144
    assert served_context_window(None, "cairn-qwen") is None


def test_the_trace_view_reads_the_session_organization_s_own_key_first() -> None:
    _serve("byo", "qwen", ServedFacts(engine="vllm", context_window=65536))
    _serve("byo", "qwen", ServedFacts(engine="vllm", context_window=262144), ACME)

    assert served_context_window("byo", "qwen", ACME) == 262144
    assert served_context_window("byo", "qwen", GLOBEX) == 65536
    assert served_context_window("byo", "qwen") == 65536


# --------------------------------------------------------------------------- #
# Recording a listing
# --------------------------------------------------------------------------- #


@pytest.fixture
def writes(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Count each write attempt; there is no database, so each one ends in the logged RuntimeError."""
    attempts: list[str] = []

    def counting() -> Any:
        attempts.append("write")
        return create_unit_of_work()

    monkeypatch.setattr(_served, "create_unit_of_work", counting)
    return attempts


@pytest.mark.asyncio
async def test_a_listing_is_kept_and_written_only_when_it_changes(writes: list[str]) -> None:
    listing = [_vllm(max_model_len=262144, root="Qwen/Qwen3-32B")]

    await record_served_models("cairn", listing, None)
    await record_served_models("cairn", listing, None)

    assert served_facts("cairn", "cairn-qwen") == ServedFacts(
        engine="vllm", context_window=262144, root="Qwen/Qwen3-32B"
    )
    assert writes == ["write"]


@pytest.mark.asyncio
async def test_a_declared_fallback_keeps_what_the_server_said(writes: list[str]) -> None:
    await record_served_models("cairn", [_vllm(max_model_len=262144, root="r")], None)

    await record_served_models("cairn", [_vllm()], None)

    assert served_facts("cairn", "cairn-qwen") is not None
    assert writes == ["write"]


@pytest.mark.asyncio
async def test_a_model_the_server_stopped_serving_goes(writes: list[str]) -> None:
    await record_served_models("cairn", [_vllm("old", max_model_len=1, root="r")], None)

    await record_served_models("cairn", [_vllm("new", max_model_len=2, root="r")], None)

    assert served_facts("cairn", "old") is None
    assert served_facts("cairn", "new") == ServedFacts(engine="vllm", context_window=2, root="r")
    assert len(writes) == 2


@pytest.mark.asyncio
async def test_a_provider_that_is_no_engine_never_writes(writes: list[str]) -> None:
    await record_served_models("openai", [Model(id="gpt-4o", object="model", created=0, owned_by="openai")], None)

    assert writes == []


@pytest.mark.asyncio
async def test_a_hosted_provider_listing_is_kept_under_its_own_name(writes: list[str]) -> None:
    """A hosted provider is dialed through its own credential, and listed by its name, so it is kept under that."""
    get_model_cache().clear()

    async def dial() -> ProviderDiscovery:
        return ProviderDiscovery(provider="cairn-hosted", models=[_vllm(max_model_len=262144, root="r")])

    async def hosted() -> dict[str, Any]:
        return {"cairn-hosted": dial}

    await discover_models_with_status(GatewayConfig(master_key="test", providers={}), hosted=hosted, force=True)

    assert served_facts("cairn-hosted", "cairn-qwen") == ServedFacts(engine="vllm", context_window=262144, root="r")
    get_model_cache().clear()


@pytest.mark.asyncio
async def test_an_organization_key_listing_is_kept_apart_from_the_deployment(writes: list[str]) -> None:
    await record_served_models("byo", [_vllm(max_model_len=1, root="r")], None)
    await record_served_models(
        "byo", [_vllm(max_model_len=2, root="r")], None, owner=KeyOwner(organization_id=ACME, key_id=uuid.UUID(int=9))
    )

    assert served_facts("byo", "cairn-qwen") == ServedFacts(engine="vllm", context_window=1, root="r")
    assert served_facts("byo", "cairn-qwen", ACME) == ServedFacts(engine="vllm", context_window=2, root="r")
    assert len(writes) == 2


@pytest.mark.asyncio
async def test_a_listing_keeps_a_bounded_number_of_models(writes: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_served, "_MAX_MODELS_PER_LISTING", 2)

    await record_served_models("cairn", [_vllm(f"m{index}", max_model_len=1) for index in range(5)], None)

    assert [served_facts("cairn", f"m{index}") is not None for index in range(5)] == [True, True, False, False, False]


def test_values_too_long_to_store_are_dropped() -> None:
    assert facts_from_listing(_vllm("m" * 513, max_model_len=1), None) is None
    facts = facts_from_listing(_vllm(max_model_len=1, root="r" * 513), None)
    assert facts is not None and facts.root is None
