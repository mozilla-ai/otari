"""Unit tests for the OSS-edition smoke gate (scripts/oss_edition_smoke.py).

The gate's own end-to-end proof is the CI job that runs it. What is worth pinning
here is the part that decides *which edition* gets booted, plus the mock provider
the run depends on: a scrub that stopped dropping the platform token would leave
the job green while checking the hybrid build, which is the one failure this gate
cannot detect by running.
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

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "oss_edition_smoke.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("oss_edition_smoke", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


smoke = _load()


def _post(url: str, *, headers: dict[str, str] | None = None) -> tuple[int, Any]:
    request = urllib.request.Request(url, data=b"{}", method="POST")
    request.add_header("Content-Type", "application/json")
    for name, value in (headers or {}).items():
        request.add_header(name, value)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


# --------------------------------------------------------------------------- #
# The environment the gateway is booted in
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name",
    [
        "OTARI_MODE",  # selects hybrid mode outright
        "OTARI_AI_TOKEN",  # a platform token selects hybrid mode when mode is unset
        "OTARI_BOOTSTRAP",  # the planned overlay selector
        "OTARI_PLATFORM_BASE_URL",  # the platform block, which only hybrid reads
        "OTARI_DATABASE_URL",  # would smoke a database nobody chose
        "DATABASE_URL",  # the CLI reads this one for --database-url
    ],
)
def test_gateway_env_settings_are_dropped(name: str) -> None:
    env = smoke.oss_edition_env({name: "set-by-the-developer-shell", "PATH": "/usr/bin"}, "secret")
    assert name not in env
    assert env["PATH"] == "/usr/bin", "only the gateway's own settings are scrubbed"


def test_secret_key_is_set_for_credential_storage() -> None:
    env = smoke.oss_edition_env({}, "a-generated-fernet-key")
    assert env["OTARI_SECRET_KEY"] == "a-generated-fernet-key"


# --------------------------------------------------------------------------- #
# The config the gateway is booted with
# --------------------------------------------------------------------------- #


def test_config_selects_no_other_edition() -> None:
    names = smoke.Names.for_run("abcd1234")
    config = smoke.oss_edition_config(
        database_url="sqlite:///smoke.db",
        port=8123,
        mock_base_url="http://127.0.0.1:9000",
        names=names,
    )
    assert "mode" not in config
    assert "platform" not in config
    assert list(config["providers"]) == [names.failing_instance]


def test_config_file_is_loadable_as_yaml(tmp_path: Path) -> None:
    """The loader parses config.yml with yaml.safe_load, which JSON satisfies."""
    path = tmp_path / "oss-edition.yml"
    config = smoke.oss_edition_config(
        database_url="sqlite:///smoke.db",
        port=8123,
        mock_base_url="http://127.0.0.1:9000",
        names=smoke.Names.for_run("abcd1234"),
    )
    smoke.write_config(path, config)
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == config


def test_run_names_are_routable() -> None:
    """A provider instance or policy name carrying ':' or '/' is refused by the API."""
    names = smoke.Names.for_run("abcd1234")
    for name in (names.policy, names.failing_instance, names.byo_instance):
        assert ":" not in name
        assert "/" not in name


def test_run_names_differ_between_runs() -> None:
    """Two runs must be able to share one database without colliding."""
    assert smoke.Names.for_run("aaaa1111").user_id != smoke.Names.for_run("bbbb2222").user_id


# --------------------------------------------------------------------------- #
# The mock provider
# --------------------------------------------------------------------------- #


@pytest.fixture
def provider() -> Iterator[Any]:
    """The real mock provider from the script, already serving on a free port."""
    with smoke.mock_provider("the-stored-byo-key") as server:
        yield server


def _base_url(server: Any) -> str:
    return f"http://127.0.0.1:{server.server_address[1]}"


def test_working_endpoint_answers_a_completion(provider: Any) -> None:
    status, body = _post(
        f"{_base_url(provider)}{smoke.WORKING_PREFIX}/v1/chat/completions",
        headers={"Authorization": "Bearer the-stored-byo-key"},
    )
    assert status == 200
    assert body["choices"][0]["message"]["content"] == smoke.REPLY
    assert body["usage"]["total_tokens"] > 0
    assert provider.working_calls == 1


def test_working_endpoint_rejects_any_other_key(provider: Any) -> None:
    """This rejection is what makes the completion prove the BYO key was resolved."""
    status, _ = _post(
        f"{_base_url(provider)}{smoke.WORKING_PREFIX}/v1/chat/completions",
        headers={"Authorization": "Bearer the-key-from-config-yml"},
    )
    assert status == 401
    assert provider.unauthorized_calls == 1
    assert provider.working_calls == 0


def test_failing_endpoint_fails_with_a_retryable_status(provider: Any) -> None:
    """5xx, because a caller-fault status is one the gateway is right not to retry."""
    status, _ = _post(f"{_base_url(provider)}{smoke.FAILING_PREFIX}/v1/chat/completions")
    assert status == 500
    assert provider.failing_calls == 1


def test_unknown_route_is_not_mistaken_for_a_completion(provider: Any) -> None:
    status, _ = _post(f"{_base_url(provider)}/somewhere/else")
    assert status == 404
    assert provider.working_calls == 0
    assert provider.failing_calls == 0
