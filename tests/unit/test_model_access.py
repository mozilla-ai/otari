"""Unit tests for the per-key model access-control matcher and validation."""

import uuid
from collections.abc import Iterator

import pytest

from gateway.core.config import GatewayConfig
from gateway.core.error_codes import MODEL_NOT_ALLOWED, MODEL_NOT_FOUND, MODEL_NOT_SERVING
from gateway.services.model_access import (
    effective_allowlist,
    is_allowlist_subset,
    is_model_allowed,
    org_model_refusal,
    validate_allowed_models,
)
from gateway.services.tenancy import org_provider_key_service as org_store
from gateway.services.tenancy.org_provider_key_service import KeyOffer


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


# --- organization-key refusals ---------------------------------------------


@pytest.fixture
def workspace_id() -> Iterator[uuid.UUID]:
    """A workspace whose overlay entries are cleared after the test."""
    workspace_id = uuid.uuid4()
    org_store._org_model_restrictions.clear()
    org_store._org_key_offers.clear()
    yield workspace_id
    org_store._org_model_restrictions.clear()
    org_store._org_key_offers.clear()


def _narrow(workspace_id: uuid.UUID, *, served: list[str], offered: dict[str, bool] | None) -> None:
    org_store._org_model_restrictions[(workspace_id, "openai")] = served
    org_store._org_key_offers[(workspace_id, "openai")] = KeyOffer(key_name="primary", offered=offered)


def test_an_unnarrowed_provider_refuses_nothing(workspace_id: uuid.UUID) -> None:
    assert org_model_refusal(workspace_id, "openai", "gpt-4o", selector="openai:gpt-4o") is None


def test_a_served_model_is_not_refused(workspace_id: uuid.UUID) -> None:
    _narrow(workspace_id, served=["gpt-4o"], offered={"gpt-4o": True, "gpt-4o-mini": False})
    assert org_model_refusal(workspace_id, "openai", "gpt-4o", selector="openai:gpt-4o") is None


def test_a_model_switched_off_is_refused_as_not_serving(workspace_id: uuid.UUID) -> None:
    _narrow(workspace_id, served=["gpt-4o"], offered={"gpt-4o": True, "gpt-4o-mini": False})
    refusal = org_model_refusal(workspace_id, "openai", "gpt-4o-mini", selector="openai:gpt-4o-mini")
    assert refusal is not None
    assert (refusal.code, refusal.status_code) == (MODEL_NOT_SERVING, 403)
    assert "offered on provider key 'primary' but not serving" in refusal.detail
    assert "openai:gpt-4o-mini" in refusal.detail


def test_a_model_no_key_offers_is_not_found(workspace_id: uuid.UUID) -> None:
    _narrow(workspace_id, served=["gpt-4o"], offered={"gpt-4o": True})
    refusal = org_model_refusal(workspace_id, "openai", "gpt-does-not-exist", selector="openai:gpt-does-not-exist")
    assert refusal is not None
    assert (refusal.code, refusal.status_code) == (MODEL_NOT_FOUND, 404)
    assert "Provider key 'primary' does not offer 'openai:gpt-does-not-exist'" in refusal.detail


def test_a_workspace_restriction_on_a_served_model_stays_not_allowed(workspace_id: uuid.UUID) -> None:
    """The key serves it; this workspace chose not to reach it, which is a permission, not a switch."""
    _narrow(workspace_id, served=["gpt-4o"], offered={"gpt-4o": True, "gpt-4o-mini": True})
    refusal = org_model_refusal(workspace_id, "openai", "gpt-4o-mini", selector="fast")
    assert refusal is not None
    assert (refusal.code, refusal.status_code) == (MODEL_NOT_ALLOWED, 403)
    assert refusal.detail == "Model 'fast' is not permitted for this API key."


def test_a_restriction_on_a_key_offering_no_rows_stays_not_allowed(workspace_id: uuid.UUID) -> None:
    _narrow(workspace_id, served=["gpt-4o"], offered=None)
    refusal = org_model_refusal(workspace_id, "openai", "gpt-4o-mini", selector="openai:gpt-4o-mini")
    assert refusal is not None
    assert refusal.code == MODEL_NOT_ALLOWED


def test_the_detail_names_the_selector_not_the_resolved_model(workspace_id: uuid.UUID) -> None:
    """An alias exists partly to keep its target off the wire."""
    _narrow(workspace_id, served=[], offered={"gpt-4o": False})
    refusal = org_model_refusal(workspace_id, "openai", "gpt-4o", selector="fast")
    assert refusal is not None
    assert "fast" in refusal.detail
    assert "gpt-4o" not in refusal.detail
