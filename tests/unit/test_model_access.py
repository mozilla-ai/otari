"""Unit tests for the per-key model access-control matcher and validation."""

import uuid
from collections.abc import Iterator

import pytest

from gateway.core.config import GatewayConfig
from gateway.core.error_codes import INVALID_MODEL, MODEL_NOT_ALLOWED, MODEL_NOT_FOUND, MODEL_NOT_SERVING
from gateway.core.settings.tools import ALL_SEARCH_TOOLS
from gateway.services.model_access import (
    effective_allowlist,
    effective_tool_allowlist,
    is_allowlist_subset,
    is_model_allowed,
    is_tool_allowed,
    is_tool_allowlist_subset,
    key_model_refusal,
    legacy_tool_grants,
    org_model_refusal,
    tool_not_allowed_detail,
    tools_within_user,
    validate_allowed_models,
    validate_allowed_tools,
)
from gateway.services.tenancy import org_provider_key_service as org_store
from gateway.services.tenancy.org_provider_key_service import KeyOffer

NO_TOOLS: frozenset[str] = frozenset()


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
    assert is_allowlist_subset(None, ["openai:gpt-4o"], tool_keys=NO_TOOLS) is True
    # An unrestricted parent (None) covers any child.
    assert is_allowlist_subset(["openai:*"], None, tool_keys=NO_TOOLS) is True
    # Deny-all child fits any parent; deny-all parent rejects a granting child.
    assert is_allowlist_subset([], ["openai:*"], tool_keys=NO_TOOLS) is True
    assert is_allowlist_subset(["openai:gpt-4o"], [], tool_keys=NO_TOOLS) is False


def test_is_allowlist_subset_concrete_and_wildcards() -> None:
    # Concrete child within a parent wildcard.
    assert is_allowlist_subset(["openai:gpt-4o"], ["openai:*"], tool_keys=NO_TOOLS) is True
    # Different instance is never covered.
    assert is_allowlist_subset(["anthropic:claude-3"], ["openai:*"], tool_keys=NO_TOOLS) is False
    # A wildcard child needs a parent at least as broad.
    assert is_allowlist_subset(["openai:*"], ["openai:*"], tool_keys=NO_TOOLS) is True
    assert is_allowlist_subset(["openai:*"], ["openai:gpt-4*"], tool_keys=NO_TOOLS) is False
    # A child glob that extends the parent glob is covered; a shorter one is not.
    assert is_allowlist_subset(["openai:gpt-4*"], ["openai:gpt-*"], tool_keys=NO_TOOLS) is True
    assert is_allowlist_subset(["openai:gpt-*"], ["openai:gpt-4*"], tool_keys=NO_TOOLS) is False
    # A concrete parent cannot cover a wildcard child.
    assert is_allowlist_subset(["openai:gpt-4*"], ["openai:gpt-4o"], tool_keys=NO_TOOLS) is False
    # Every child entry must be covered.
    assert is_allowlist_subset(["openai:gpt-4o", "anthropic:claude-3"], ["openai:*"], tool_keys=NO_TOOLS) is False


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


# --- search tool access ------------------------------------------------------

# DeepSeek searches and serves models under one prefix: the shape that put tools beside models.
_TOOLS = GatewayConfig(
    search_tools={"deepseek": {"provider": "deepseek", "api_key": "k"}, "exa-search": {"provider": "exa"}}
)


def test_a_model_list_grants_a_tool_only_by_naming_its_key_whole() -> None:
    assert legacy_tool_grants(_TOOLS, None) is None
    assert legacy_tool_grants(_TOOLS, []) == []
    assert legacy_tool_grants(_TOOLS, ["deepseek:deepseek", "openai:gpt-4o"]) == ["deepseek"]
    assert legacy_tool_grants(_TOOLS, ["deepseek:*", "exa:exa-*"]) == []


def test_the_key_list_wins_then_the_users_then_the_model_lists() -> None:
    def effective(**lists: list[str] | None) -> list[str] | None:
        given = {"key_tools": None, "key_models": None, "user_tools": None, "user_models": None, **lists}
        return effective_tool_allowlist(_TOOLS, **given)

    assert effective() is None
    assert effective(key_tools=["exa-search"], user_tools=["*"]) == ["exa-search"]
    # [] is a real answer, never collapsed into inheriting.
    assert effective(key_tools=[], user_tools=["*"]) == []
    assert effective(user_tools=["*"], key_models=["openai:gpt-4o"]) == ["*"]
    # Neither list set: the key's model list, else its user's, read the deprecated way.
    assert effective(key_models=["deepseek:deepseek"]) == ["deepseek"]
    assert effective(user_models=["exa:exa-search"]) == ["exa-search"]
    # A key's grant is bounded by its user's: a user's wildcard admits a key's exact entry and grants no tool.
    assert effective(key_models=["deepseek:deepseek"], user_models=["exa:exa-search"]) == []
    assert effective(key_models=["deepseek:deepseek"], user_models=["deepseek:*"]) == []
    assert effective(key_models=["deepseek:deepseek", "exa:exa-search"], user_models=["exa:exa-search"]) == [
        "exa-search"
    ]
    # A key restricted to models stays refused every tool until one is granted.
    assert effective(key_models=["deepseek:*"]) == []


def test_a_tool_is_allowed_by_its_name_or_the_star() -> None:
    assert is_tool_allowed(None, "deepseek") is True
    assert is_tool_allowed([], "deepseek") is False
    assert is_tool_allowed([ALL_SEARCH_TOOLS], "deepseek") is True
    assert is_tool_allowed(["exa-search"], "deepseek") is False


def test_a_key_narrows_its_users_tools_and_the_star_covers_only_the_star() -> None:
    assert is_tool_allowlist_subset(None, []) is True
    assert is_tool_allowlist_subset(["*"], None) is True
    assert is_tool_allowlist_subset(["deepseek"], ["*"]) is True
    assert is_tool_allowlist_subset(["*"], ["deepseek", "exa-search"]) is False
    assert is_tool_allowlist_subset(["deepseek"], ["exa-search"]) is False
    assert is_tool_allowlist_subset([], []) is True


def test_a_users_tools_bound_a_key_by_its_tool_list_else_its_model_list() -> None:
    assert tools_within_user(_TOOLS, ["deepseek"], user_tools=["*"], user_models=[]) is True
    assert tools_within_user(_TOOLS, ["deepseek"], user_tools=["exa-search"], user_models=None) is False
    # With no tool list, the user's model list grants only the tools it names exactly.
    assert tools_within_user(_TOOLS, ["deepseek"], user_tools=None, user_models=["deepseek:deepseek"]) is True
    assert tools_within_user(_TOOLS, ["deepseek"], user_tools=None, user_models=["deepseek:*"]) is False
    assert tools_within_user(_TOOLS, ["*"], user_tools=None, user_models=None) is True


def test_a_model_wildcard_does_not_cover_a_key_naming_a_tool_the_deprecated_way() -> None:
    legacy = _TOOLS.legacy_search_tool_keys()
    assert is_allowlist_subset(["deepseek:deepseek"], ["deepseek:*"], tool_keys=legacy) is False
    assert is_allowlist_subset(["deepseek:deepseek"], ["deepseek:deepseek"], tool_keys=legacy) is True
    assert is_allowlist_subset(["deepseek:deepseek-chat"], ["deepseek:*"], tool_keys=legacy) is True


def test_a_tool_list_names_configured_tools_or_the_star() -> None:
    assert validate_allowed_tools(_TOOLS, None) is None
    assert validate_allowed_tools(_TOOLS, []) == []
    assert validate_allowed_tools(_TOOLS, [" deepseek ", "deepseek", "*"]) == ["deepseek", "*"]
    for bad in (["deepseek:deepseek"], ["nope"], [""]):
        with pytest.raises(ValueError):
            validate_allowed_tools(_TOOLS, bad)
    with pytest.raises(ValueError):
        validate_allowed_tools(_TOOLS, "deepseek")  # type: ignore[arg-type]


def test_the_refusal_says_a_model_wildcard_grants_models_only_when_one_matched() -> None:
    plain = tool_not_allowed_detail("deepseek", "deepseek:deepseek")
    assert "Name it in allowed_tools" in plain and "wildcard" not in plain
    legacy = tool_not_allowed_detail("deepseek", "deepseek:deepseek", [["deepseek:*"]])
    assert "A wildcard in allowed_models allows models only" in legacy
    # The user's wildcard is the one that matched when the key names the tool exactly.
    bounded = [["deepseek:deepseek"], ["deepseek:*"]]
    assert "allows models only" in tool_not_allowed_detail("deepseek", "deepseek:deepseek", bounded)
    # An exact entry on the user is no wildcard: the key's own list is what leaves the tool out.
    narrowed = [["openai:gpt-4o"], ["openai:*", "deepseek:deepseek"]]
    assert "wildcard" not in tool_not_allowed_detail("deepseek", "deepseek:deepseek", narrowed)


def test_a_model_list_names_a_search_tool_only_by_its_exact_key() -> None:
    """``exa`` names search tools and no model, so a wildcard or another name under it grants nothing."""
    assert validate_allowed_models(_TOOLS, ["exa:exa-search", "deepseek:*"]) == ["exa:exa-search", "deepseek:*"]
    for bad in ("exa:*", "exa:exa-*", "exa:nope"):
        with pytest.raises(ValueError, match="name tools in allowed_tools"):
            validate_allowed_models(_TOOLS, [bad])


def test_a_tool_named_against_the_rules_has_no_deprecated_grant() -> None:
    """A search entry named ``*`` still loads, with a warning; ``exa:*`` must not read as every tool."""
    config = GatewayConfig(search_tools={"*": {"provider": "exa"}, "exa-search": {"provider": "exa"}})
    assert config.legacy_search_tool_keys() == {"exa:exa-search"}
    assert legacy_tool_grants(config, ["exa:*", "exa:exa-search"]) == ["exa-search"]
    with pytest.raises(ValueError, match="name tools in allowed_tools"):
        validate_allowed_models(config, ["exa:*"])


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
    assert org_model_refusal(workspace_id, "openai", "gpt-4o", selector="openai:gpt-4o", through_alias=False) is None


def test_a_served_model_is_not_refused(workspace_id: uuid.UUID) -> None:
    _narrow(workspace_id, served=["gpt-4o"], offered={"gpt-4o": True, "gpt-4o-mini": False})
    assert org_model_refusal(workspace_id, "openai", "gpt-4o", selector="openai:gpt-4o", through_alias=False) is None


def test_a_model_switched_off_is_refused_as_not_serving(workspace_id: uuid.UUID) -> None:
    _narrow(workspace_id, served=["gpt-4o"], offered={"gpt-4o": True, "gpt-4o-mini": False})
    refusal = org_model_refusal(
        workspace_id, "openai", "gpt-4o-mini", selector="openai:gpt-4o-mini", through_alias=False
    )
    assert refusal is not None
    assert (refusal.code, refusal.status_code) == (MODEL_NOT_SERVING, 403)
    assert "offered on provider key 'primary' but not serving" in refusal.detail
    assert "openai:gpt-4o-mini" in refusal.detail


def test_a_model_no_key_offers_is_not_found(workspace_id: uuid.UUID) -> None:
    _narrow(workspace_id, served=["gpt-4o"], offered={"gpt-4o": True})
    refusal = org_model_refusal(
        workspace_id, "openai", "gpt-does-not-exist", selector="openai:gpt-does-not-exist", through_alias=False
    )
    assert refusal is not None
    assert (refusal.code, refusal.status_code) == (MODEL_NOT_FOUND, 404)
    assert "Provider key 'primary' does not offer 'openai:gpt-does-not-exist'" in refusal.detail


def test_a_workspace_restriction_on_a_served_model_stays_not_allowed(workspace_id: uuid.UUID) -> None:
    """The key serves it; this workspace chose not to reach it, which is a permission, not a switch."""
    _narrow(workspace_id, served=["gpt-4o"], offered={"gpt-4o": True, "gpt-4o-mini": True})
    refusal = org_model_refusal(workspace_id, "openai", "gpt-4o-mini", selector="fast", through_alias=True)
    assert refusal is not None
    assert (refusal.code, refusal.status_code) == (MODEL_NOT_ALLOWED, 403)
    assert refusal.detail == "Model 'fast' is not permitted for this API key."


def test_a_restriction_on_a_key_offering_no_rows_stays_not_allowed(workspace_id: uuid.UUID) -> None:
    _narrow(workspace_id, served=["gpt-4o"], offered=None)
    refusal = org_model_refusal(
        workspace_id, "openai", "gpt-4o-mini", selector="openai:gpt-4o-mini", through_alias=False
    )
    assert refusal is not None
    assert refusal.code == MODEL_NOT_ALLOWED


def test_the_detail_names_the_selector_not_the_resolved_model(workspace_id: uuid.UUID) -> None:
    """An alias exists partly to keep its target off the wire."""
    _narrow(workspace_id, served=[], offered={"gpt-4o": False})
    refusal = org_model_refusal(workspace_id, "openai", "gpt-4o", selector="fast", through_alias=True)
    assert refusal is not None
    assert "fast" in refusal.detail
    assert "gpt-4o" not in refusal.detail


def test_an_unrestricted_key_is_never_refused_by_its_allow_list() -> None:
    assert key_model_refusal(None, None, selector="no-such-model") is None
    assert key_model_refusal(None, "openai:gpt-4o", selector="openai:gpt-4o") is None


def test_a_restricted_key_is_told_an_unresolvable_selector_is_invalid_not_forbidden() -> None:
    refusal = key_model_refusal(["openai:gpt-4o"], None, selector="no-such-model")
    assert refusal is not None
    assert (refusal.code, refusal.status_code) == (INVALID_MODEL, 400)
    assert "no-such-model" in refusal.detail
    assert "'provider:model' selector, an alias, or a model id from the catalog" in refusal.detail


def test_a_restricted_key_is_refused_a_model_its_list_excludes() -> None:
    refusal = key_model_refusal(["openai:gpt-4o"], "gemini:gemini-2.5-flash", selector="gemini:gemini-2.5-flash")
    assert refusal is not None
    assert (refusal.code, refusal.status_code) == (MODEL_NOT_ALLOWED, 403)
    assert key_model_refusal(["openai:*"], "openai:gpt-4o", selector="fast") is None


def _narrow_anthropic(workspace_id: uuid.UUID, *, offered: dict[str, bool]) -> None:
    org_store._org_model_restrictions[(workspace_id, "anthropic")] = [m for m, on in offered.items() if on]
    org_store._org_key_offers[(workspace_id, "anthropic")] = KeyOffer(key_name="primary", offered=offered)


def test_an_undated_name_whose_dated_builds_are_all_off_is_not_serving(workspace_id: uuid.UUID) -> None:
    _narrow_anthropic(workspace_id, offered={"claude-haiku-4-5-20251001": False, "claude-sonnet-4-6": True})
    refusal = org_model_refusal(
        workspace_id, "anthropic", "claude-haiku-4-5", selector="anthropic:claude-haiku-4-5", through_alias=False
    )
    assert refusal is not None
    assert (refusal.code, refusal.status_code) == (MODEL_NOT_SERVING, 403)
    assert refusal.detail == (
        "Model 'anthropic:claude-haiku-4-5' is offered on provider key 'primary' "
        "as 'anthropic:claude-haiku-4-5-20251001' but not serving. Turn it on under Organization > Providers."
    )


def test_an_undated_name_with_two_dated_builds_is_not_found_and_names_them(workspace_id: uuid.UUID) -> None:
    _narrow_anthropic(workspace_id, offered={"claude-x-20250101": True, "claude-x-20250601": True})
    refusal = org_model_refusal(
        workspace_id, "anthropic", "claude-x", selector="anthropic:claude-x", through_alias=False
    )
    assert refusal is not None
    assert (refusal.code, refusal.status_code) == (MODEL_NOT_FOUND, 404)
    assert refusal.detail == (
        "Provider key 'primary' does not offer 'anthropic:claude-x'. "
        "It offers 'anthropic:claude-x-20250101', 'anthropic:claude-x-20250601'; send one of those."
    )


def test_an_undated_name_with_every_dated_build_off_is_not_serving_and_names_them(workspace_id: uuid.UUID) -> None:
    """Two builds is ambiguous only while one serves; with both off, the switch is what refuses."""
    _narrow_anthropic(workspace_id, offered={"claude-x-20250101": False, "claude-x-20250601": False})
    refusal = org_model_refusal(
        workspace_id, "anthropic", "claude-x", selector="anthropic:claude-x", through_alias=False
    )
    assert refusal is not None
    assert (refusal.code, refusal.status_code) == (MODEL_NOT_SERVING, 403)
    assert "as 'anthropic:claude-x-20250101', 'anthropic:claude-x-20250601' but not serving" in refusal.detail


@pytest.mark.parametrize("alias", ["fast", "claude-x"])
def test_an_undated_name_behind_an_alias_does_not_name_the_dated_builds(workspace_id: uuid.UUID, alias: str) -> None:
    """A bare alias may share its target's model name, so provenance decides, not the spelling."""
    _narrow_anthropic(workspace_id, offered={"claude-x-20250101": True, "claude-x-20250601": True})
    refusal = org_model_refusal(workspace_id, "anthropic", "claude-x", selector=alias, through_alias=True)
    assert refusal is not None
    assert refusal.code == MODEL_NOT_FOUND
    assert "20250101" not in refusal.detail
    assert "20250601" not in refusal.detail
