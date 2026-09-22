"""Unit tests for the hybrid smoke gate (scripts/hybrid_edition_smoke.py).

The gate's own proof is the CI job that runs it. Pinned here is what decides
*which edition* boots and what the fakes answer: a config that grew a provider
block would boot standalone with a platform token and fail at startup for a
reason unrelated to any change, and a fake that drifted from the protocol would
fail the gate while the gateway was right.
"""

import importlib.util
import json
import sys
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from gateway.core.config import API_ROOT, PLATFORM_TOKEN_ENV_VAR

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "hybrid_edition_smoke.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("hybrid_edition_smoke", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


smoke = _load()


def _call(method: str, url: str, *, headers: dict[str, str] | None = None, body: Any = None) -> tuple[int, Any]:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    for name, value in (headers or {}).items():
        request.add_header(name, value)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            raw = response.read()
            return response.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as error:
        raw = error.read()
        return error.code, json.loads(raw) if raw else None


# --------------------------------------------------------------------------- #
# What decides which edition boots
# --------------------------------------------------------------------------- #


def test_the_gate_walks_the_root_the_app_actually_serves() -> None:
    """The gate carries its own copy of the API root so it can stay standard-library only."""
    assert smoke.API_ROOT == API_ROOT


def test_the_gate_sets_the_token_the_app_reads() -> None:
    assert smoke.PLATFORM_TOKEN_ENV_VAR == PLATFORM_TOKEN_ENV_VAR


@pytest.mark.parametrize(
    "name",
    ["OTARI_MODE", "OTARI_BOOTSTRAP", "OTARI_PLATFORM_BASE_URL", "OTARI_DATABASE_URL", "DATABASE_URL"],
)
def test_developer_gateway_settings_are_dropped(name: str) -> None:
    env = smoke.hybrid_env({name: "set-by-the-developer-shell", "PATH": "/usr/bin"})
    assert name not in env
    assert env["PATH"] == "/usr/bin"


def test_the_platform_token_is_the_one_setting_put_back() -> None:
    """A developer's own token is replaced, not kept: the fake only knows this run's."""
    env = smoke.hybrid_env({PLATFORM_TOKEN_ENV_VAR: "gw_someone_elses"})
    assert env[PLATFORM_TOKEN_ENV_VAR] == smoke.GATEWAY_TOKEN


def test_config_is_a_hybrid_deployment_and_nothing_else() -> None:
    config = smoke.hybrid_config(port=8123, platform_base_url="http://127.0.0.1:9000/api/v1")
    assert config["platform"]["base_url"] == "http://127.0.0.1:9000/api/v1"
    assert "providers" not in config, "local providers are refused in hybrid mode"
    assert "database_url" not in config, "a hybrid gateway runs no database"
    assert "sandbox_url" not in config, "no sandbox is what makes native code execution pass through"
    # The gateway appends /search itself, and sends its token only under base_url.
    assert config["web_search_url"] == "http://127.0.0.1:9000/api/v1/gateway/web-search"


def test_config_file_is_loadable_as_yaml(tmp_path: Path) -> None:
    path = tmp_path / "hybrid.yml"
    config = smoke.hybrid_config(port=8123, platform_base_url="http://127.0.0.1:9000/api/v1")
    smoke.write_config(path, config)
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == config


# --------------------------------------------------------------------------- #
# The fake control plane
# --------------------------------------------------------------------------- #


@pytest.fixture
def control_plane() -> Iterator[Any]:
    state = smoke.ControlPlaneState(provider_base_url="http://provider.test", mcp_url="http://mcp.test/mcp")
    with smoke.serve(smoke.FakeControlPlane(state), "test-control-plane") as server:
        yield server


def _resolve_url(server: Any) -> str:
    return f"{server.base_url}{smoke.PLATFORM_PREFIX}/gateway/provider-keys/resolve"


def _tokens(user_token: str) -> dict[str, str]:
    return {"X-Gateway-Token": smoke.GATEWAY_TOKEN, "X-User-Token": user_token}


def test_resolve_answers_the_documented_multi_attempt_shape(control_plane: Any) -> None:
    status, body = _call(
        "POST",
        _resolve_url(control_plane),
        headers=_tokens(smoke.USER_TOKEN_OK),
        body={"model": "m", "provider": "openai"},
    )
    assert status == 200
    assert body["request_id"] and body["attempts"][0]["attempt_id"]
    attempt = body["attempts"][0]
    assert attempt["provider"] == "openai"
    assert attempt["model"] == "m"
    assert attempt["api_key"] == smoke.OPENAI_KEY
    assert attempt["api_base"] == "http://provider.test/openai/v1"


def test_resolve_routes_anthropic_to_the_messages_base(control_plane: Any) -> None:
    """The Anthropic SDK appends /v1/messages itself, so the base must not."""
    _, body = _call(
        "POST",
        _resolve_url(control_plane),
        headers=_tokens(smoke.USER_TOKEN_OK),
        body={"model": "m", "provider": "anthropic"},
    )
    assert body["attempts"][0]["api_base"] == "http://provider.test/anthropic"
    assert body["attempts"][0]["api_key"] == smoke.ANTHROPIC_KEY


# Parametrized by attribute name, not by value: the tokens are generated per
# import, so a value here would give each xdist worker a different test id and
# collection would disagree across workers.
@pytest.mark.parametrize(
    ("token_name", "status"),
    [
        ("USER_TOKEN_BROKE", 402),
        ("USER_TOKEN_THROTTLED", 429),
        ("USER_TOKEN_UNKNOWN", 401),
    ],
)
def test_resolve_refuses_by_user_token(control_plane: Any, token_name: str, status: int) -> None:
    token: str = getattr(smoke, token_name)
    got, _ = _call("POST", _resolve_url(control_plane), headers=_tokens(token), body={"model": "m"})
    assert got == status


def test_resolve_refuses_a_gateway_it_does_not_know(control_plane: Any) -> None:
    status, _ = _call(
        "POST",
        _resolve_url(control_plane),
        headers={"X-Gateway-Token": "gw_other", "X-User-Token": smoke.USER_TOKEN_OK},
        body={"model": "m"},
    )
    assert status == 401


def test_web_access_resolve_authorizes_search_only(control_plane: Any) -> None:
    """Fetch is never authorized here, so a Fetch declaration must fail before it."""
    status, body = _call(
        "POST",
        f"{control_plane.base_url}{smoke.PLATFORM_PREFIX}/gateway/web-search/resolve",
        headers=_tokens(smoke.USER_TOKEN_OK),
        body={"requested_tools": ["web_search"]},
    )
    assert status == 200
    assert body["enabled"] is True
    assert body["authorized_tools"] == ["web_search"]


def test_search_backend_requires_the_gateway_token(control_plane: Any) -> None:
    url = f"{control_plane.base_url}{smoke.PLATFORM_PREFIX}/gateway/web-search/search?q=x&format=json"
    assert _call("GET", url)[0] == 401
    status, body = _call("GET", url, headers={"X-Gateway-Token": smoke.GATEWAY_TOKEN})
    assert status == 200
    assert body["results"][0]["extracted_content"] == smoke.SEARCH_CONTENT, "supplied so nothing is retrieved"


def test_usage_is_accepted_and_recorded(control_plane: Any) -> None:
    status, _ = _call(
        "POST",
        f"{control_plane.base_url}{smoke.PLATFORM_PREFIX}/gateway/usage",
        headers={"X-Gateway-Token": smoke.GATEWAY_TOKEN},
        body={"correlation_id": "att_0001", "status": "success", "is_final_attempt": True},
    )
    assert status == 204
    [recorded] = control_plane.recorder.all("usage")
    assert recorded.body["correlation_id"] == "att_0001"


# --------------------------------------------------------------------------- #
# The mock provider
# --------------------------------------------------------------------------- #


@pytest.fixture
def provider() -> Iterator[Any]:
    with smoke.serve(smoke.MockProvider(), "test-provider") as server:
        yield server


def test_chat_calls_the_offered_tool_first_and_then_answers(provider: Any) -> None:
    url = f"{provider.base_url}/openai/v1/chat/completions"
    auth = {"Authorization": f"Bearer {smoke.OPENAI_KEY}"}
    tools = [{"type": "function", "function": {"name": "web_search", "parameters": {}}}]
    _, first = _call("POST", url, headers=auth, body={"messages": [{"role": "user", "content": "q"}], "tools": tools})
    call = first["choices"][0]["message"]["tool_calls"][0]["function"]
    assert call["name"] == "web_search"
    assert json.loads(call["arguments"]) == {"query": smoke.SEARCH_QUERY}

    _, second = _call(
        "POST",
        url,
        headers=auth,
        body={"messages": [{"role": "user", "content": "q"}, {"role": "tool", "content": "r"}], "tools": tools},
    )
    assert second["choices"][0]["message"]["content"] == smoke.REPLY


def test_provider_rejects_a_key_the_control_plane_did_not_issue(provider: Any) -> None:
    """This rejection is what makes a served completion prove the resolved key was used."""
    status, _ = _call(
        "POST",
        f"{provider.base_url}/openai/v1/chat/completions",
        headers={"Authorization": "Bearer sk-from-somewhere-else"},
        body={"messages": []},
    )
    assert status == 401
    status, _ = _call(
        "POST", f"{provider.base_url}/anthropic/v1/messages", headers={"x-api-key": "other"}, body={"messages": []}
    )
    assert status == 401


def test_messages_answers_a_native_web_search_declaration_in_kind(provider: Any) -> None:
    _, body = _call(
        "POST",
        f"{provider.base_url}/anthropic/v1/messages",
        headers={"x-api-key": smoke.ANTHROPIC_KEY},
        body={"tools": [{"type": "web_search_20250305", "name": "web_search"}]},
    )
    assert [block["type"] for block in body["content"]] == ["server_tool_use", "web_search_tool_result", "text"]
    assert body["content"][1]["content"][0]["url"] == smoke.SEARCH_URL


def test_responses_answers_a_native_web_search_declaration_in_kind(provider: Any) -> None:
    _, body = _call(
        "POST",
        f"{provider.base_url}/openai/v1/responses",
        headers={"Authorization": f"Bearer {smoke.OPENAI_KEY}"},
        body={"tools": [{"type": "web_search_preview"}]},
    )
    assert [item["type"] for item in body["output"]] == ["web_search_call", "message"]


def test_messages_answers_in_native_code_execution_blocks(provider: Any) -> None:
    _, body = _call(
        "POST", f"{provider.base_url}/anthropic/v1/messages", headers={"x-api-key": smoke.ANTHROPIC_KEY}, body={}
    )
    assert [block["type"] for block in body["content"]] == ["server_tool_use", "code_execution_tool_result", "text"]
    assert body["content"][1]["content"]["stdout"] == smoke.CODE_STDOUT


def test_responses_answers_with_a_code_interpreter_call(provider: Any) -> None:
    _, body = _call(
        "POST",
        f"{provider.base_url}/openai/v1/responses",
        headers={"Authorization": f"Bearer {smoke.OPENAI_KEY}"},
        body={},
    )
    assert [item["type"] for item in body["output"]] == ["code_interpreter_call", "message"]
    assert body["output"][0]["outputs"][0]["logs"] == smoke.CODE_STDOUT


# --------------------------------------------------------------------------- #
# The fake MCP server
# --------------------------------------------------------------------------- #


@pytest.fixture
def mcp() -> Iterator[Any]:
    with smoke.serve(smoke.FakeMcpServer(), "test-mcp") as server:
        yield server


def _rpc(server: Any, method: str, params: Any = None, *, request_id: int | None = 1) -> tuple[int, Any]:
    message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        message["params"] = params
    if request_id is not None:
        message["id"] = request_id
    return _call("POST", server.mcp_url, body=message)


def test_mcp_initialize_echoes_the_requested_protocol_version(mcp: Any) -> None:
    status, body = _rpc(mcp, "initialize", {"protocolVersion": "2025-03-26", "capabilities": {}})
    assert status == 200
    assert body["result"]["protocolVersion"] == "2025-03-26"
    assert "tools" in body["result"]["capabilities"]


def test_mcp_notification_is_accepted_without_a_body(mcp: Any) -> None:
    status, body = _rpc(mcp, "notifications/initialized", request_id=None)
    assert (status, body) == (202, None)


def test_mcp_lists_and_calls_the_one_tool(mcp: Any) -> None:
    _, listed = _rpc(mcp, "tools/list")
    assert [tool["name"] for tool in listed["result"]["tools"]] == [smoke.MCP_TOOL]
    _, called = _rpc(mcp, "tools/call", {"name": smoke.MCP_TOOL, "arguments": {"term": "smoke"}})
    assert called["result"]["content"] == [{"type": "text", "text": smoke.MCP_RESULT}]
    assert called["result"]["isError"] is False
    [recorded] = mcp.recorder.all("tools/call")
    assert recorded.body == {"name": smoke.MCP_TOOL, "arguments": {"term": "smoke"}}


def test_mcp_offers_no_server_stream(mcp: Any) -> None:
    """The streamable client reads 405 on GET as 'no stream offered' and carries on."""
    assert _call("GET", mcp.mcp_url)[0] == 405


# --------------------------------------------------------------------------- #
# The live mode
# --------------------------------------------------------------------------- #


_LIVE_ENV = {
    smoke.LIVE_OPENAI_KEY_ENV: "sk-live-openai",
    smoke.LIVE_ANTHROPIC_KEY_ENV: "sk-ant-live",
    smoke.LIVE_TAVILY_KEY_ENV: "tvly-live",
}


def test_live_needs_both_provider_keys_and_names_the_missing_one() -> None:
    """A live run with a provider key missing must fail up front, not skip a leg silently."""
    with pytest.raises(smoke.SmokeFailure, match=smoke.LIVE_ANTHROPIC_KEY_ENV):
        smoke.LiveProviders.from_env({k: v for k, v in _LIVE_ENV.items() if k != smoke.LIVE_ANTHROPIC_KEY_ENV})


def test_live_tavily_is_optional() -> None:
    """Without it, managed search runs a real model against the fake backend."""
    live = smoke.LiveProviders.from_env({k: v for k, v in _LIVE_ENV.items() if k != smoke.LIVE_TAVILY_KEY_ENV})
    assert live.tavily_key is None


def test_live_models_default_and_can_be_overridden() -> None:
    live = smoke.LiveProviders.from_env(dict(_LIVE_ENV))
    assert (live.openai_model, live.anthropic_model) == (
        smoke.DEFAULT_LIVE_OPENAI_MODEL,
        smoke.DEFAULT_LIVE_ANTHROPIC_MODEL,
    )
    live = smoke.LiveProviders.from_env({**_LIVE_ENV, smoke.LIVE_OPENAI_MODEL_ENV: "gpt-x"})
    assert live.openai_model == "gpt-x"


def test_live_keys_are_scrubbed_from_the_gateway_environment() -> None:
    """The gateway learns a key only through a resolve answer, as a deployment would."""
    env = smoke.hybrid_env({**_LIVE_ENV, "PATH": "/usr/bin"})
    assert not any(name in env for name in _LIVE_ENV)


def test_live_config_puts_tavily_ahead_of_the_search_url() -> None:
    config = smoke.hybrid_config(port=8123, platform_base_url="http://cp/api/v1", tavily_key="tvly-live")
    assert config["web_search_provider"] == "tavily"
    assert config["web_search_provider_api_key"] == "tvly-live"
    assert "web_search_url" in config, "the URL stays; the backend prefers the provider"
    assert "web_search_provider" not in smoke.hybrid_config(port=8123, platform_base_url="http://cp/api/v1")


def test_live_resolve_carries_the_real_key_and_no_api_base() -> None:
    live = smoke.LiveProviders.from_env(dict(_LIVE_ENV))
    state = smoke.ControlPlaneState(provider_base_url="http://provider.test", mcp_url="http://mcp.test/mcp", live=live)
    with smoke.serve(smoke.FakeControlPlane(state), "test-live-control-plane") as server:
        for provider, key in (("openai", "sk-live-openai"), ("anthropic", "sk-ant-live")):
            _, body = _call(
                "POST",
                _resolve_url(server),
                headers=_tokens(smoke.USER_TOKEN_OK),
                body={"model": "m", "provider": provider},
            )
            [attempt] = body["attempts"]
            assert attempt["api_key"] == key
            assert attempt["api_base"] is None, "a live attempt dials the provider's own endpoint"
            assert attempt["provider"] == provider


# --------------------------------------------------------------------------- #
# The container mode
# --------------------------------------------------------------------------- #


def test_container_config_leaves_the_listen_address_to_the_image() -> None:
    """The image pins OTARI_HOST/OTARI_PORT, and env beats a config file.

    Writing them anyway would state something the run does not honor, and was
    how the first container run failed: the gateway listened on the image's 8000
    while the smoke polled a port of its own.
    """
    config = smoke.hybrid_config(port=8123, platform_base_url="http://cp/api/v1", in_container=True)
    assert "host" not in config
    assert "port" not in config
    source = smoke.hybrid_config(port=8123, platform_base_url="http://cp/api/v1")
    assert (source["host"], source["port"]) == (smoke.LOOPBACK, 8123)


def test_fakes_bind_loopback_by_default_and_are_dialled_there() -> None:
    with smoke.serve(smoke.MockProvider(), "test-bind-default") as server:
        assert server.peer_host == smoke.LOOPBACK
        assert server.base_url == f"http://{smoke.LOOPBACK}:{server.port}"


def test_fakes_are_dialled_through_the_host_alias_when_bound_for_a_container() -> None:
    """A container has its own loopback, so the fakes must be reachable by name."""
    with smoke.serve(smoke.MockProvider(smoke.ALL_INTERFACES), "test-bind-container") as server:
        assert server.peer_host == smoke.CONTAINER_HOST_ALIAS
        assert server.base_url == f"http://{smoke.CONTAINER_HOST_ALIAS}:{server.port}"
