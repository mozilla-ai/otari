"""The ``hosted_providers:`` block: what it accepts, what it refuses, and that a key never leaks.

Validated at load (``load_config``), so a misconfigured entry fails the start
rather than a request. Every refusal names the entry and the field and never
the value, because a value here may be a key.
"""

import json
import os
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from any_llm import LLMProvider

from gateway.api.routes.settings import _current_settings
from gateway.core.config import GatewayConfig, load_config
from gateway.core.settings.providers import parse_hosted_providers
from gateway.services.providers import refuse_unsafe_configured_endpoints

KEY = "sk-live-config-9876"


@pytest.fixture(autouse=True)
def _isolated_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Hide OTARI_ variables and .env files, and restore os.environ, which loading writes to."""
    monkeypatch.chdir(tmp_path)
    with mock.patch.dict(os.environ):
        for key in [key for key in os.environ if key.startswith("OTARI_")]:
            del os.environ[key]
        yield


def _load(tmp_path: Path, text: str) -> GatewayConfig:
    config_file = tmp_path / "config.yml"
    config_file.write_text(text, encoding="utf-8")
    return load_config(str(config_file))


def _validated(block: dict[str, Any]) -> GatewayConfig:
    config = GatewayConfig(hosted_providers=block)
    config.validate_hosted_providers()
    return config


def test_an_entry_takes_the_fields_the_api_body_takes() -> None:
    organization_id = uuid.uuid4()
    config = _validated(
        {
            "openai": {
                "api_key": KEY,
                "api_base": " https://hosted.test/v1 ",
                "client_args": {"timeout": 9},
                "enabled": False,
                "restricted_to_organization_id": str(organization_id),
                "models": ["gpt-4o"],
            }
        }
    )

    entry = config.hosted_provider_entries()["openai"]
    assert entry.api_key == KEY
    assert entry.api_base == "https://hosted.test/v1"
    assert entry.client_args == {"timeout": 9}
    assert entry.enabled is False
    assert entry.restricted_to_organization_id == organization_id
    assert entry.models == ("gpt-4o",)


def test_an_entry_is_enabled_by_default() -> None:
    assert _validated({"openai": {"api_key": KEY}}).hosted_provider_entries()["openai"].enabled is True


def test_an_empty_block_is_no_providers(tmp_path: Path) -> None:
    config = _load(tmp_path, "hosted_providers:\n")

    assert config.hosted_provider_entries() == {}


def test_an_unknown_implementation_is_refused() -> None:
    with pytest.raises(ValueError, match=r"hosted_providers\.not-a-provider is not an any-llm implementation"):
        _validated({"not-a-provider": {"api_key": KEY}})


def test_a_named_entry_names_its_implementation_in_provider() -> None:
    entry = _validated({"openai-eu": {"provider": "openai", "api_key": KEY}}).hosted_provider_entries()["openai-eu"]

    assert (entry.name, entry.provider) == ("openai-eu", "openai")


def test_a_named_entry_naming_no_implementation_is_refused() -> None:
    with pytest.raises(ValueError, match=r"hosted_providers\.openai-eu\.provider is not an any-llm implementation"):
        _validated({"openai-eu": {"provider": "not-a-provider", "api_key": KEY}})


@pytest.mark.parametrize(
    ("name", "problem"),
    [("anthropic", "names another provider"), ("hosted", "is reserved"), ("eu/openai", "must not contain")],
)
def test_a_name_a_selector_could_not_address_is_refused(name: str, problem: str) -> None:
    with pytest.raises(ValueError, match=rf"hosted_providers\.{name}: the name {problem}"):
        _validated({name: {"provider": "openai", "api_key": KEY}})


def test_a_provider_type_is_refused_because_the_implementation_goes_in_provider() -> None:
    with pytest.raises(ValueError, match=r"hosted_providers\.openai\.provider_type is not accepted"):
        _validated({"openai": {"api_key": KEY, "provider_type": "openai"}})


def test_a_keyed_provider_without_a_key_is_refused() -> None:
    with pytest.raises(ValueError, match=r"hosted_providers\.openai\.api_key is required"):
        _validated({"openai": {"enabled": True}})


def test_a_keyed_provider_on_an_endpoint_of_its_own_may_leave_the_key_out() -> None:
    entry = _validated({"openai": {"api_base": "https://hosted.test/v1"}}).hosted_provider_entries()["openai"]

    assert entry.api_key is None


def test_a_bare_keyed_entry_is_refused() -> None:
    with pytest.raises(ValueError, match=r"hosted_providers\.anthropic\.api_key is required"):
        _validated({"anthropic": None})


# The providers docs/configuration.md lists as taking no ``api_key``; keep the two in step.
DOCUMENTED_KEYLESS = {
    "cascadia",
    "llamacpp",
    "llamafile",
    "lmstudio",
    "ollama",
    "otari",
    "vertexai",
    "vertexaianthropic",
    "vllm",
}


@pytest.mark.parametrize("provider", sorted(DOCUMENTED_KEYLESS))
def test_a_keyless_backend_may_leave_the_key_out(provider: str) -> None:
    block = {"own": {"provider": provider, "api_base": "http://localhost:8000/v1"}}

    assert _validated(block).hosted_provider_entries()["own"].api_key is None


def test_an_otari_entry_must_take_a_name_of_its_own() -> None:
    """``otari`` prefixes the gateway's own tool prices, so a provider under it would lose its model prices."""
    with pytest.raises(ValueError, match=r"hosted_providers\.otari: the name is reserved; give the provider a name"):
        _validated({"otari": {"api_base": "http://localhost:8000/v1"}})


def test_the_documented_keyless_providers_are_exactly_those_accepted_without_a_key() -> None:
    accepted = set()
    for provider in LLMProvider:
        try:
            _validated({"own": {"provider": provider.value}})
        except ValueError:
            continue
        accepted.add(provider.value)

    assert accepted == DOCUMENTED_KEYLESS


def test_a_restriction_must_name_an_organization_by_id() -> None:
    with pytest.raises(ValueError, match=r"hosted_providers\.openai\.restricted_to_organization_id"):
        _validated({"openai": {"api_key": KEY, "restricted_to_organization_id": "acme"}})


def test_an_unknown_field_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match=r"hosted_providers\.openai\.max_retries"):
        _validated({"openai": {"api_key": KEY, "max_retries": 3}})


def test_a_bedrock_botocore_config_is_checked() -> None:
    with pytest.raises(ValueError, match=r"hosted_providers\.bedrock\.client_args\.config"):
        _validated({"bedrock": {"api_key": KEY, "client_args": {"config": {"not_an_option": 1}}}})


@pytest.mark.parametrize(
    "entry",
    [
        {"api_key": KEY, "enabled": "maybe"},
        {"api_key": KEY, "client_args": KEY},
        {"api_key": KEY, "surprise": KEY},
        {"api_key": KEY, "provider_type": KEY},
    ],
)
def test_a_refusal_never_quotes_a_value(entry: dict[str, Any]) -> None:
    with pytest.raises(ValueError) as refused:
        parse_hosted_providers({"openai": entry})

    assert KEY not in str(refused.value)


def test_the_key_appears_in_no_rendering_of_the_config_or_the_settings_view() -> None:
    config = _validated({"openai": {"api_key": KEY}})

    assert KEY not in repr(config)
    assert KEY not in repr(config.hosted_provider_entries())
    assert KEY not in _current_settings(config).model_dump_json()
    assert "hosted_providers" not in {field.key for field in _current_settings(config).config}


def test_the_key_is_interpolated_from_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", KEY)

    config = _load(tmp_path, "hosted_providers:\n  openai:\n    api_key: ${OPENAI_API_KEY}\n")

    assert config.hosted_provider_entries()["openai"].api_key == KEY


def test_an_unset_key_variable_fails_the_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        _load(tmp_path, "hosted_providers:\n  openai:\n    api_key: ${OPENAI_API_KEY}\n")


def test_the_whole_block_is_settable_as_json_in_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "OTARI_HOSTED_PROVIDERS", json.dumps({"openai": {"api_key": KEY, "api_base": "https://hosted.test/v1"}})
    )

    config = _load(tmp_path, "")

    entry = config.hosted_provider_entries()["openai"]
    assert entry.api_key == KEY
    assert entry.api_base == "https://hosted.test/v1"


def test_the_environment_block_is_validated_at_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_HOSTED_PROVIDERS", json.dumps({"openai": {"enabled": True}}))

    with pytest.raises(ValueError, match=r"hosted_providers\.openai\.api_key is required"):
        _load(tmp_path, "")


def test_the_environment_block_replaces_the_file_block_whole(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "OTARI_HOSTED_PROVIDERS", json.dumps({"openrouter": {"api_key": KEY, "api_base": "https://env.test/v1"}})
    )

    config = _load(
        tmp_path,
        "hosted_providers:\n"
        "  openrouter:\n    api_key: sk-file-1111\n    api_base: https://file.test/v1\n"
        "  anthropic:\n    api_key: sk-file-2222\n",
    )

    entries = config.hosted_provider_entries()
    assert set(entries) == {"openrouter"}
    assert entries["openrouter"].api_key == KEY
    assert entries["openrouter"].api_base == "https://env.test/v1"


def test_the_environment_block_beats_the_structured_environment_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OTARI_CONFIG_YAML", "hosted_providers:\n  anthropic:\n    api_key: sk-yaml-3333\n")
    monkeypatch.setenv("OTARI_HOSTED_PROVIDERS", json.dumps({"openai": {"api_key": KEY}}))

    config = _load(tmp_path, "")

    assert set(config.hosted_provider_entries()) == {"openai"}


def test_the_environment_block_interpolates_variables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    monkeypatch.setenv("OTARI_HOSTED_PROVIDERS", json.dumps({"openai": {"api_key": "${OPENAI_API_KEY}"}}))

    config = _load(tmp_path, "")

    assert config.hosted_provider_entries()["openai"].api_key == KEY


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ('{"openai": {"api_key": "' + KEY + '"', "is not valid JSON"),
        (json.dumps([{"openai": {"api_key": KEY}}]), "must be a JSON object"),
    ],
)
def test_a_malformed_environment_block_fails_without_quoting_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw: str, match: str
) -> None:
    monkeypatch.setenv("OTARI_HOSTED_PROVIDERS", raw)

    with pytest.raises(ValueError, match=match) as refused:
        _load(tmp_path, "")

    assert KEY not in str(refused.value)


def test_an_empty_environment_block_leaves_the_file_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_HOSTED_PROVIDERS", "")

    config = _load(tmp_path, f"hosted_providers:\n  openai:\n    api_key: {KEY}\n")

    assert set(config.hosted_provider_entries()) == {"openai"}


@pytest.mark.asyncio
async def test_with_private_hosts_refused_a_private_endpoint_fails_the_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """The gate a stored provider's endpoint passes at write, applied to the entry at boot."""
    monkeypatch.setenv("OTARI_PROVIDER_ALLOW_PRIVATE_HOSTS", "false")
    config = _validated({"vllm": {"api_base": "http://127.0.0.1:8000/v1"}, "openai": {"api_key": KEY}})

    with pytest.raises(ValueError, match=r"hosted_providers\.vllm\.api_base") as refused:
        await refuse_unsafe_configured_endpoints(config)

    assert "openai" not in str(refused.value)


@pytest.mark.asyncio
async def test_by_default_a_private_endpoint_is_accepted() -> None:
    await refuse_unsafe_configured_endpoints(_validated({"vllm": {"api_base": "http://127.0.0.1:8000/v1"}}))


def test_an_entry_serves_every_country_by_default() -> None:
    assert _validated({"openai": {"api_key": KEY}}).hosted_provider_entries()["openai"].supported_countries is None


def test_an_entry_normalizes_its_supported_countries() -> None:
    entry = _validated({"openai": {"api_key": KEY, "supported_countries": ["us", " de ", "US"]}})

    assert entry.hosted_provider_entries()["openai"].supported_countries == ("US", "DE")


def test_an_entry_with_a_value_that_is_no_country_code_is_refused() -> None:
    with pytest.raises(ValueError, match=r"hosted_providers\.openai\.supported_countries"):
        _validated({"openai": {"api_key": KEY, "supported_countries": ["USA"]}})
