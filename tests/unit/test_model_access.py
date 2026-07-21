"""Unit tests for the per-key model access-control matcher and validation."""

import pytest

from gateway.core.config import GatewayConfig
from gateway.services.model_access import (
    effective_allowlist,
    is_allowlist_subset,
    is_model_allowed,
    validate_allowed_models,
)


class _Key:
    """Minimal stand-in for an APIKey row carrying only ``allowed_models``."""

    def __init__(self, allowed_models: list[str] | None) -> None:
        self.allowed_models = allowed_models


class _User:
    """Minimal stand-in for a User row carrying only ``allowed_models``."""

    def __init__(self, allowed_models: list[str] | None) -> None:
        self.allowed_models = allowed_models


def test_none_allowlist_is_unrestricted() -> None:
    assert is_model_allowed(None, "openai:gpt-4o") is True


def test_empty_allowlist_is_deny_all() -> None:
    # The load-bearing distinction: [] must NOT collapse into None.
    assert is_model_allowed([], "openai:gpt-4o") is False
    assert is_model_allowed([], "anthropic:claude-3") is False


def test_exact_match() -> None:
    assert is_model_allowed(["openai:gpt-4o"], "openai:gpt-4o") is True
    assert is_model_allowed(["openai:gpt-4o"], "openai:gpt-4o-mini") is False


def test_instance_wildcard() -> None:
    assert is_model_allowed(["openai:*"], "openai:gpt-4o") is True
    assert is_model_allowed(["openai:*"], "anthropic:claude-3") is False


def test_prefix_glob() -> None:
    assert is_model_allowed(["openai:gpt-4*"], "openai:gpt-4o") is True
    assert is_model_allowed(["openai:gpt-4*"], "openai:gpt-3.5-turbo") is False


def test_effective_allowlist_key_wins() -> None:
    assert effective_allowlist(_Key(["openai:*"])) == ["openai:*"]  # type: ignore[arg-type]
    assert effective_allowlist(_Key(None)) is None  # type: ignore[arg-type]
    assert effective_allowlist(None) is None
    # A deny-all key stays deny-all (not conflated with unrestricted).
    assert effective_allowlist(_Key([])) == []  # type: ignore[arg-type]


def test_effective_allowlist_inherits_user_default() -> None:
    # A key with no list of its own inherits the user's default.
    assert effective_allowlist(_Key(None), _User(["openai:*"])) == ["openai:*"]  # type: ignore[arg-type]
    # The key's own list wins over the user default (it may only narrow it).
    assert effective_allowlist(_Key(["openai:gpt-4o"]), _User(["openai:*"])) == ["openai:gpt-4o"]  # type: ignore[arg-type]
    # A deny-all user default is inherited, not conflated with unrestricted.
    assert effective_allowlist(_Key(None), _User([])) == []  # type: ignore[arg-type]
    # No user, no key list -> unrestricted.
    assert effective_allowlist(_Key(None), None) is None  # type: ignore[arg-type]


def test_is_allowlist_subset_inherit_and_unrestricted() -> None:
    # A child that inherits (None) never broadens -> always a subset.
    assert is_allowlist_subset(None, ["openai:gpt-4o"]) is True
    # An unrestricted parent (None) covers any child.
    assert is_allowlist_subset(["openai:*"], None) is True
    # Deny-all child fits any parent; deny-all parent rejects a granting child.
    assert is_allowlist_subset([], ["openai:*"]) is True
    assert is_allowlist_subset(["openai:gpt-4o"], []) is False


def test_is_allowlist_subset_concrete_and_wildcards() -> None:
    # Concrete child within a parent wildcard.
    assert is_allowlist_subset(["openai:gpt-4o"], ["openai:*"]) is True
    # Different instance is never covered.
    assert is_allowlist_subset(["anthropic:claude-3"], ["openai:*"]) is False
    # A wildcard child needs a parent at least as broad.
    assert is_allowlist_subset(["openai:*"], ["openai:*"]) is True
    assert is_allowlist_subset(["openai:*"], ["openai:gpt-4*"]) is False
    # A child glob that extends the parent glob is covered; a shorter one is not.
    assert is_allowlist_subset(["openai:gpt-4*"], ["openai:gpt-*"]) is True
    assert is_allowlist_subset(["openai:gpt-*"], ["openai:gpt-4*"]) is False
    # A concrete parent cannot cover a wildcard child.
    assert is_allowlist_subset(["openai:gpt-4*"], ["openai:gpt-4o"]) is False
    # Every child entry must be covered.
    assert is_allowlist_subset(["openai:gpt-4o", "anthropic:claude-3"], ["openai:*"]) is False


def test_validate_passthrough_and_dedup() -> None:
    config = GatewayConfig()
    assert validate_allowed_models(config, None) is None
    assert validate_allowed_models(config, []) == []
    assert validate_allowed_models(config, ["openai:gpt-4o", "openai:gpt-4o"]) == ["openai:gpt-4o"]
    assert validate_allowed_models(config, ["openai:*", "anthropic:claude-3*"]) == ["openai:*", "anthropic:claude-3*"]


@pytest.mark.parametrize(
    "bad",
    [
        "gpt-4o",  # no instance prefix
        "openai:gpt-*-turbo",  # mid-string glob
        "openai:*extra",  # glob not trailing
        "openai:a*b",  # glob not trailing
        "openai:*:*",  # multiple globs / bad shape
        "bogusprovider:x",  # unknown provider/instance
        "openai:",  # empty model
    ],
)
def test_validate_rejects_bad_entries(bad: str) -> None:
    with pytest.raises(ValueError):
        validate_allowed_models(GatewayConfig(), [bad])
