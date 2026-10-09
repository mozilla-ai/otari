"""Parsing and validating the ``budgets`` and ``api_keys`` sections of config.yml."""

import os
import secrets
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from pydantic import SecretStr, ValidationError

from gateway.core.config import GatewayConfig, load_config
from gateway.core.settings.api_keys import ApiKeyConfig, declared_secret_problem

SECRET = "tk-" + secrets.token_urlsafe(48)
OTHER_SECRET = "tk-" + secrets.token_urlsafe(48)

MLPA_CONFIG = """
budgets:
  end-user-budget-ai: {max_budget: 0.1, reset_alignment: calendar_day, rpm_limit: 40, tpm_limit: 2000}
  mlpa-global: {max_budget: 100, reset_alignment: calendar_day}
api_keys:
  mlpa:
    secret: ${MLPA_SERVICE_KEY}
    user_id: mlpa
    is_service_key: true
    end_user_budget_ids: [end-user-budget-ai]
    end_user_budget_id: end-user-budget-ai
    ceiling: mlpa-global
"""


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


def _key(secret: str = SECRET, **fields: Any) -> dict[str, Any]:
    return {"secret": secret, **fields}


def test_the_mlpa_example_loads_with_its_secret_from_the_environment(tmp_path: Path) -> None:
    os.environ["MLPA_SERVICE_KEY"] = SECRET

    config = _load(tmp_path, MLPA_CONFIG)

    budget = config.budgets["end-user-budget-ai"]
    assert (budget.max_budget, budget.reset_alignment, budget.rpm_limit, budget.tpm_limit) == (
        0.1,
        "calendar_day",
        40,
        2000,
    )
    key = config.api_keys["mlpa"]
    assert key.secret.get_secret_value() == SECRET
    assert key.is_service_key is True
    assert key.end_user_budget_ids == ["end-user-budget-ai"]
    assert key.ceiling == "mlpa-global"


def test_the_secret_appears_in_no_rendering_of_the_config() -> None:
    config = GatewayConfig(api_keys={"mlpa": _key()})

    assert SECRET not in repr(config)
    assert SECRET not in str(config.model_dump())


def test_an_unset_secret_variable_fails_the_load(tmp_path: Path) -> None:
    os.environ.pop("MLPA_SERVICE_KEY", None)

    with pytest.raises(ValueError, match="MLPA_SERVICE_KEY"):
        _load(tmp_path, MLPA_CONFIG)


@pytest.mark.parametrize(
    ("secret", "problem"),
    [
        ("", "is empty"),
        (f" {SECRET}", "whitespace"),
        ("sk-" + secrets.token_urlsafe(48), "must start with 'tk-'"),
        ("tk-" + secrets.token_urlsafe(16), "at least 50 characters"),
        ("tk-" + secrets.token_urlsafe(48)[:-1] + ".", "may hold only"),
        ("tk-" + "changeme" * 7, "placeholder"),
        ("tk-" + "a" * 60, "placeholder"),
    ],
)
def test_a_weak_or_malformed_secret_is_refused_without_being_quoted(secret: str, problem: str) -> None:
    assert problem in (declared_secret_problem(secret) or "")

    config = GatewayConfig(api_keys={"mlpa": _key(secret)})
    with pytest.raises(ValueError, match=r"api_keys\.mlpa\.secret") as refused:
        config.validate_declared_access()
    if secret.strip():
        assert secret.strip() not in str(refused.value)


def test_a_minted_shape_secret_is_accepted() -> None:
    assert declared_secret_problem(SECRET) is None
    GatewayConfig(api_keys={"mlpa": _key()}).validate_declared_access()


def test_two_keys_may_not_share_a_secret() -> None:
    config = GatewayConfig(api_keys={"one": _key(), "two": _key()})

    with pytest.raises(ValueError, match=r"api_keys\.two\.secret is the same as api_keys\.one\.secret") as refused:
        config.validate_declared_access()
    assert SECRET not in str(refused.value)


def test_a_secret_may_not_be_the_master_key() -> None:
    config = GatewayConfig(master_key=SECRET, api_keys={"mlpa": _key()})

    with pytest.raises(ValueError, match="must differ from the master key"):
        config.validate_declared_access()


def test_a_hybrid_gateway_refuses_declarations() -> None:
    with pytest.raises(ValueError, match="hybrid gateway holds neither"):
        GatewayConfig(mode="hybrid", budgets={"b": {"max_budget": 1}}).validate_declared_access()
    with pytest.raises(ValueError, match="hybrid gateway holds neither"):
        GatewayConfig(mode="hybrid", api_keys={"mlpa": _key()}).validate_declared_access()


@pytest.mark.parametrize(
    "api_keys",
    [
        {"mlpa": {"secrt": SECRET}},
        {"mlpa": {"secret": SECRET, "is_service_key": "maybe"}},
        {"mlpa": SECRET},
        [{"secret": SECRET}],
        {"mlpa": {"secret": SECRET, "user_id": "mlpa", "allowed_modles": ["x"], "token": SECRET}},
    ],
)
def test_a_malformed_key_is_refused_without_quoting_its_secret(api_keys: Any) -> None:
    with pytest.raises(ValidationError) as refused:
        GatewayConfig(api_keys=api_keys)

    assert SECRET not in str(refused.value)
    assert SECRET not in repr(refused.value.errors())


def test_the_default_end_user_budget_must_be_on_the_list() -> None:
    with pytest.raises(ValidationError, match="end_user_budget_id must be one of end_user_budget_ids"):
        ApiKeyConfig(secret=SecretStr(SECRET), end_user_budget_ids=["a"], end_user_budget_id="b")


def test_a_key_name_must_be_an_identifier() -> None:
    with pytest.raises(ValidationError, match="api key name 'my key'"):
        GatewayConfig(api_keys={"my key": _key()})


@pytest.mark.parametrize(
    ("budget", "message"),
    [
        ({"budget_duration_sec": 86400, "reset_alignment": "calendar_day"}, "not both"),
        ({"max_budget": -1}, "greater than or equal to 0"),
        ({"rpm_limit": 0}, "greater than or equal to 1"),
        ({"reset_alignment": "fortnightly"}, "calendar_day"),
        ({"max_budgt": 1}, "Extra inputs are not permitted"),
    ],
)
def test_an_invalid_budget_is_refused(budget: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        GatewayConfig(budgets={"b": budget})


def test_a_budget_id_must_match_the_api_rule() -> None:
    with pytest.raises(ValidationError, match="budget id '-b'"):
        GatewayConfig(budgets={"-b": {"max_budget": 1}})


def test_nothing_declared_is_valid() -> None:
    config = GatewayConfig()

    assert config.budgets == {}
    assert config.api_keys == {}
    config.validate_declared_access()


def test_a_different_secret_per_key_is_valid() -> None:
    GatewayConfig(api_keys={"one": _key(), "two": _key(OTHER_SECRET)}).validate_declared_access()
