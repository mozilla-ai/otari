"""Unit tests for the per-key model access-control matcher and validation."""

import pytest

from gateway.core.config import GatewayConfig
from gateway.services.model_access import (
    effective_allowlist,
    is_model_allowed,
    validate_allowed_models,
)


class _Key:
    """Minimal stand-in for an APIKey row carrying only ``allowed_models``."""

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
