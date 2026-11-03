"""Which providers a request's country may reach: ``provider_supported_countries``."""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from gateway.core.config import GatewayConfig, load_config
from gateway.models.routing import PolicySpec
from gateway.services.providers import CountryGate
from gateway.services.providers._country_gate import provider_vendor
from gateway.services.routing import NoEligibleCandidatesError, compile_catalog_plan, compile_policy

_HEADER = "CF-IPCountry"


def _config(**overrides: Any) -> GatewayConfig:
    settings: dict[str, Any] = {
        "master_key": "test-master-key",
        "model_discovery": False,
        "client_country_header": _HEADER,
        "providers": {
            "openai": {"api_key": "sk-openai"},
            "anthropic": {"api_key": "sk-anthropic"},
            "openai-eu": {"provider_type": "openai", "api_key": "sk-openai-eu"},
            "local": {"provider_type": "openai-compatible", "api_base": "http://localhost:8000/v1"},
        },
        "provider_supported_countries": {"openai": ["US", "DE"]},
    }
    settings.update(overrides)
    return GatewayConfig(**settings)


def test_the_gate_is_off_by_default() -> None:
    assert GatewayConfig().provider_supported_countries == {}


def test_provider_types_and_country_codes_are_normalized() -> None:
    config = _config(provider_supported_countries={" OpenAI ": ["us", " de ", "US"]})

    assert config.provider_supported_countries == {"openai": ["US", "DE"]}


def test_an_unknown_provider_type_is_refused() -> None:
    with pytest.raises(ValidationError, match="provider_supported_countries"):
        _config(provider_supported_countries={"openai-eu": ["US"]})


@pytest.mark.parametrize("code", ["USA", "U", "1A", "", "XX", "ZZ"])
def test_a_value_that_is_no_country_code_is_refused(code: str) -> None:
    with pytest.raises(ValidationError, match="provider_supported_countries"):
        _config(provider_supported_countries={"openai": [code]})


def test_kosovo_is_a_country_code() -> None:
    # ISO 3166-1 does not assign XK, but MaxMind and DB-IP files give it to Kosovo.
    assert _config(provider_supported_countries={"openai": ["xk"]}).provider_supported_countries == {"openai": ["XK"]}


def test_a_list_with_no_configured_country_source_needs_a_stored_file() -> None:
    # The config loads: whether a file is stored is checked at startup, where the database is reachable.
    assert _config(client_country_header=None).country_gate_needs_stored_database()


@pytest.mark.parametrize(
    "source", [{"client_country_header": _HEADER}, {"client_country_header": None, "client_country_database": "c.mmdb"}]
)
def test_a_configured_country_source_needs_no_stored_file(source: dict[str, str | None]) -> None:
    assert not _config(**source).country_gate_needs_stored_database()


def test_no_list_needs_no_stored_file() -> None:
    assert not _config(client_country_header=None, provider_supported_countries={}).country_gate_needs_stored_database()


@pytest.mark.parametrize(
    ("instance", "vendor"),
    [
        ("openai", "openai"),
        ("openai-eu", "openai"),
        ("anthropic", "anthropic"),
        ("local", None),
        ("mistral", "mistral"),
    ],
)
def test_the_vendor_is_who_serves_the_instance(instance: str, vendor: str | None) -> None:
    assert provider_vendor(_config(), instance) == vendor


@pytest.mark.parametrize(
    ("instance", "country", "refused"),
    [
        ("openai", "US", False),
        ("openai", "FR", True),
        ("openai", None, True),
        ("openai-eu", "FR", True),
        ("anthropic", "FR", False),
        ("anthropic", None, False),
        ("local", "FR", False),
    ],
)
def test_the_gate_refuses_a_listed_vendor_outside_its_countries(
    instance: str, country: str | None, refused: bool
) -> None:
    assert CountryGate(country).refuses(_config(), instance) is refused


def _spec(default: str, *on_failure: str) -> PolicySpec:
    return PolicySpec.model_validate({"select": [{"default": default}], "on_failure": list(on_failure)})


def test_a_policy_falls_over_past_a_provider_the_country_may_not_use() -> None:
    plan = compile_policy(
        _config(), "smart", _spec("openai:gpt-5-mini", "anthropic:claude-haiku-4-5"), country_gate=CountryGate("FR")
    )

    assert [attempt.instance for attempt in plan.attempts] == ["anthropic"]
    assert [(item.selector, item.reason) for item in plan.dropped] == [("openai:gpt-5-mini", "country_not_supported")]
    assert "FR" in plan.dropped[0].detail


def test_a_policy_with_no_candidate_left_is_refused_as_access() -> None:
    with pytest.raises(NoEligibleCandidatesError) as exc_info:
        compile_policy(_config(), "smart", _spec("openai:gpt-5-mini"), country_gate=CountryGate(None))

    assert exc_info.value.status_code == 403


def test_explain_compiles_without_a_country() -> None:
    plan = compile_policy(_config(), "smart", _spec("openai:gpt-5-mini"))

    assert [attempt.instance for attempt in plan.attempts] == ["openai"]


def test_a_catalog_plan_drops_a_provider_the_country_may_not_use(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "gateway.services.routing.compiler.resolve_catalog_fallback",
        lambda *_args, **_kwargs: ["openai:gpt-5-mini", "openai-eu:gpt-5-mini", "anthropic:claude-haiku-4-5"],
    )

    plan = compile_catalog_plan(_config(), "gpt-5-mini", country_gate=CountryGate("FR"))

    assert plan is not None
    assert [attempt.instance for attempt in plan.attempts] == ["anthropic"]


def test_the_lists_are_settable_as_json_in_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_PROVIDER_SUPPORTED_COUNTRIES", json.dumps({"openai": ["us"]}))
    monkeypatch.setenv("OTARI_CLIENT_COUNTRY_HEADER", _HEADER)
    config_file = tmp_path / "config.yml"
    config_file.write_text("", encoding="utf-8")

    config = load_config(str(config_file))

    assert config.provider_supported_countries == {"openai": ["US"]}
    assert config.client_country_header == _HEADER


@pytest.mark.parametrize(
    ("country", "countries", "excluded"),
    [
        ("US", ("US", "DE"), False),
        ("FR", ("US", "DE"), True),
        (None, ("US",), True),
        ("FR", (), True),
        ("FR", None, False),
        (None, None, False),
    ],
)
def test_a_hosted_providers_own_list_excludes_the_countries_it_leaves_out(
    country: str | None, countries: tuple[str, ...] | None, excluded: bool
) -> None:
    assert CountryGate(country).excludes(countries) is excluded
