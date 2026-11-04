"""Unit tests for the OSS-edition smoke gate (scripts/oss_edition_smoke.py).

The gate's own end-to-end proof is the CI job that runs it. What is worth pinning
here is the part that decides *which build* gets booted, plus the mock provider
the run depends on: a scrub that stopped dropping ``OTARI_BOOTSTRAP`` would leave
the job green while checking an overlay build, which is the one failure this gate
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

from gateway.core.config import API_ROOT

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


def test_the_gate_walks_the_root_the_app_actually_serves() -> None:
    """The gate carries its own copy of the API root, and this is what keeps it honest.

    It has to: importing the app would give up the standard-library-only
    property that lets this gate catch a dev-only import reaching an OSS code
    path. So the copy is deliberate, and a drift between the two would send
    every request in the gate to a path the app does not serve.
    """
    assert smoke.API_ROOT == API_ROOT


@pytest.mark.parametrize(
    "name",
    [
        "OTARI_BOOTSTRAP",  # the overlay selector, which would boot a build nobody chose
        "OTARI_DATABASE_URL",  # would smoke a database nobody chose
        # A name the scrub has never heard of, so a regression from the prefix
        # sweep to a fixed allowlist of the names above fails here.
        "OTARI_SETTING_ADDED_NEXT_YEAR",
        "DATABASE_URL",  # the CLI reads this one for --database-url
    ],
)
def test_gateway_env_settings_are_dropped(name: str) -> None:
    env = smoke.oss_edition_env({name: "set-by-the-developer-shell", "PATH": "/usr/bin"}, "secret", "pepper")
    assert name not in env
    assert env["PATH"] == "/usr/bin", "only the gateway's own settings are scrubbed"


def test_secret_key_is_set_for_credential_storage() -> None:
    env = smoke.oss_edition_env({}, "a-generated-fernet-key", "pepper")
    assert env["OTARI_SECRET_KEY"] == "a-generated-fernet-key"


def test_the_provider_account_pepper_is_set_for_provider_copies() -> None:
    env = smoke.oss_edition_env({"OTARI_PROVIDER_ACCOUNT_PEPPER": "from-the-shell"}, "key", "a-generated-pepper")
    assert env["OTARI_PROVIDER_ACCOUNT_PEPPER"] == "a-generated-pepper"


# --------------------------------------------------------------------------- #
# The config the gateway is booted with
# --------------------------------------------------------------------------- #


def test_config_declares_only_the_failing_instance() -> None:
    names = smoke.Names.for_run("abcd1234")
    config = smoke.oss_edition_config(
        database_url="sqlite:///smoke.db",
        port=8123,
        mock_base_url="http://127.0.0.1:9000",
        names=names,
    )
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


# --------------------------------------------------------------------------- #
# The container mode
# --------------------------------------------------------------------------- #


def test_image_is_off_by_default_and_taken_as_given() -> None:
    assert smoke.parse_args([]).image is None
    assert smoke.parse_args(["--image", "otari:pr-sha"]).image == "otari:pr-sha"


def test_container_config_leaves_the_listen_address_to_the_image() -> None:
    """The image pins OTARI_HOST/OTARI_PORT, and env beats a config file."""
    names = smoke.Names.for_run("abcd1234")
    config = smoke.oss_edition_config(
        database_url=smoke.CONTAINER_DATABASE_URL,
        port=None,
        mock_base_url="http://host.docker.internal:9000",
        names=names,
    )
    assert "host" not in config
    assert "port" not in config
    source = smoke.oss_edition_config(
        database_url="sqlite:///smoke.db", port=8123, mock_base_url="http://127.0.0.1:9000", names=names
    )
    assert (source["host"], source["port"]) == (smoke.LOOPBACK, 8123)


def test_container_database_is_an_absolute_sqlite_path() -> None:
    """Four slashes: a relative path would land wherever the image's workdir is."""
    assert smoke.CONTAINER_DATABASE_URL.startswith("sqlite:////")


def test_mock_binds_loopback_by_default_and_is_dialed_there() -> None:
    with smoke.mock_provider("key") as server:
        assert server.server_address[0] == smoke.LOOPBACK
        assert smoke.peer_base_url(server, smoke.LOOPBACK) == f"http://{smoke.LOOPBACK}:{server.server_address[1]}"


def test_mock_is_dialed_through_the_host_alias_when_bound_for_a_container() -> None:
    """A container has its own loopback, so the mock must be reachable by name."""
    with smoke.mock_provider("key", smoke.ALL_INTERFACES) as server:
        url = smoke.peer_base_url(server, smoke.ALL_INTERFACES)
        assert url == f"http://{smoke.CONTAINER_HOST_ALIAS}:{server.server_address[1]}"
