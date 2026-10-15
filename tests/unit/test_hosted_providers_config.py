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

from gateway.api.routes.settings import _current_settings
from gateway.core.config import GatewayConfig, load_config
from gateway.core.settings.providers import parse_hosted_providers

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


def test_a_provider_type_is_refused_because_named_instances_stay_in_providers() -> None:
    with pytest.raises(ValueError, match=r"hosted_providers\.openai\.provider_type is not accepted"):
        _validated({"openai": {"api_key": KEY, "provider_type": "openai"}})


def test_a_keyed_provider_without_a_key_is_refused() -> None:
    with pytest.raises(ValueError, match=r"hosted_providers\.openai\.api_key is required"):
        _validated({"openai": {"api_base": "https://hosted.test/v1"}})


def test_a_bare_keyed_entry_is_refused() -> None:
    with pytest.raises(ValueError, match=r"hosted_providers\.anthropic\.api_key is required"):
        _validated({"anthropic": None})


@pytest.mark.parametrize("provider", ["ollama", "vllm"])
def test_a_keyless_backend_may_leave_the_key_out(provider: str) -> None:
    entry = _validated({provider: {"api_base": "http://localhost:8000/v1"}}).hosted_provider_entries()[provider]

    assert entry.api_key is None


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
    monkeypatch.setenv("OTARI_HOSTED_PROVIDERS", json.dumps({"openai": {"api_base": "https://hosted.test/v1"}}))

    with pytest.raises(ValueError, match=r"hosted_providers\.openai\.api_key is required"):
        _load(tmp_path, "")
