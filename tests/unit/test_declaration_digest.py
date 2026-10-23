"""The digest a start compares to tell whether a config.yml entry changed."""

import hashlib
import secrets
from typing import Any

from pydantic import SecretStr

from gateway.core.settings.api_keys import ApiKeyConfig
from gateway.core.settings.budgets import BudgetConfig
from gateway.core.settings.declarations import declaration_digest

SECRET = "tk-" + secrets.token_urlsafe(48)


def _key(**overrides: Any) -> ApiKeyConfig:
    fields: dict[str, Any] = {"secret": SecretStr(SECRET), "user_id": "mlpa", "is_service_key": True}
    fields.update(overrides)
    return ApiKeyConfig(**fields)


def test_the_digest_ignores_key_order() -> None:
    assert declaration_digest({"a": 1, "b": [1, 2]}) == declaration_digest({"b": [1, 2], "a": 1})


def test_a_budget_digest_changes_with_any_field() -> None:
    budget = BudgetConfig(max_budget=1.0, reset_alignment="calendar_day")

    assert (
        budget.declaration_digest() == BudgetConfig(max_budget=1.0, reset_alignment="calendar_day").declaration_digest()
    )
    assert (
        budget.declaration_digest() != BudgetConfig(max_budget=2.0, reset_alignment="calendar_day").declaration_digest()
    )


def test_a_key_digest_changes_with_its_settings_and_its_secret() -> None:
    digest = _key().declaration_digest()

    assert digest == _key().declaration_digest()
    assert digest != _key(is_service_key=False).declaration_digest()
    assert digest != _key(secret=SecretStr("tk-" + secrets.token_urlsafe(48))).declaration_digest()


def test_a_key_digest_is_not_a_digest_of_the_secret_alone() -> None:
    assert SECRET not in _key().declaration_digest()
    assert declaration_digest({"secret": SECRET}) != _key().declaration_digest()


def test_a_key_that_leaves_a_later_field_unset_keeps_the_digest_it_had_before_the_field() -> None:
    key = _key()
    before = key.model_dump(mode="json", exclude={"secret", "allow_mock_response"})
    before["secret_sha256"] = hashlib.sha256(SECRET.encode()).hexdigest()

    assert key.declaration_digest() == declaration_digest(before)
    assert _key(allow_mock_response=True).declaration_digest() != declaration_digest(before)
