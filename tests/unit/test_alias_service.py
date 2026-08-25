"""Unit tests for the alias cache that keeps resolution synchronous."""

import time
import uuid
from collections.abc import Iterator

import pytest

from gateway.core.config import GatewayConfig
from gateway.services import alias_service
from gateway.services.alias_service import (
    all_alias_names,
    cache_is_stale,
    cached_aliases,
    effective_aliases,
    reset_alias_cache,
    resolve_effective_alias,
)

CONFIG = GatewayConfig(
    providers={"anthropic": {"api_key": "sk-ant"}},
    aliases={"configalias": "anthropic:claude-opus-4"},
)

# The workspace an omitted ``workspace_id`` falls back to, which is what every
# pre-existing row was backfilled onto and where a deployment-wide write lands.
DEFAULT = uuid.UUID("00000000-0000-4000-8000-00000000d3fa")
OTHER = uuid.UUID("00000000-0000-4000-8000-0000000007e5")


@pytest.fixture(autouse=True)
def _clean_cache() -> Iterator[None]:
    reset_alias_cache()
    yield
    reset_alias_cache()


def _prime(
    aliases: dict[str, str],
    per_user: dict[str, dict[str, str]] | None = None,
    *,
    workspace_id: uuid.UUID = DEFAULT,
    default_workspace: uuid.UUID | None = DEFAULT,
) -> None:
    """Stand in for a database load without needing a session.

    Adds one workspace's layers, so calling it twice builds a two-workspace
    cache. Stamped with monotonic(), which is process uptime rather than an
    epoch: a literal 0.0 would read as loaded-at-boot and so already ancient.
    """
    alias_service._cache.setdefault(workspace_id, {}).update(aliases)
    for user_id, names in (per_user or {}).items():
        alias_service._user_cache.setdefault(workspace_id, {}).setdefault(user_id, {}).update(names)
    alias_service._default_workspace = default_workspace
    alias_service._cached_at = time.monotonic()


def test_config_aliases_resolve_without_any_stored_ones() -> None:
    assert resolve_effective_alias(CONFIG, "configalias") == "anthropic:claude-opus-4"


def test_stored_aliases_resolve() -> None:
    _prime({"fast": "anthropic:claude-haiku-4"})
    assert resolve_effective_alias(CONFIG, "fast", workspace_id=DEFAULT) == "anthropic:claude-haiku-4"


def test_unknown_name_is_not_an_alias() -> None:
    _prime({"fast": "anthropic:claude-haiku-4"})
    assert resolve_effective_alias(CONFIG, "openai:gpt-4o", workspace_id=DEFAULT) is None


def test_both_kinds_are_merged() -> None:
    _prime({"fast": "anthropic:claude-haiku-4"})
    assert effective_aliases(CONFIG, workspace_id=DEFAULT) == {
        "fast": "anthropic:claude-haiku-4",
        "configalias": "anthropic:claude-opus-4",
    }


def test_config_wins_over_a_stored_alias_of_the_same_name() -> None:
    # The API refuses to create this, so it is a safety net: if a row somehow
    # exists, the config alias is what resolves, matching what the listing says.
    _prime({"configalias": "home_lab:qwen3"})
    assert resolve_effective_alias(CONFIG, "configalias", workspace_id=DEFAULT) == "anthropic:claude-opus-4"


def test_empty_target_is_not_an_alias() -> None:
    _prime({"broken": ""})
    assert resolve_effective_alias(CONFIG, "broken", workspace_id=DEFAULT) is None


def test_cache_starts_stale_and_is_fresh_after_priming() -> None:
    # Staleness is what drives the refresher; an unloaded cache must not look
    # like an empty-but-current one, or a worker would never load aliases.
    assert cache_is_stale()
    _prime({"fast": "anthropic:claude-haiku-4"})
    assert not cache_is_stale(ttl=3600)
    assert cache_is_stale(ttl=0)


def test_reset_clears_the_cache() -> None:
    _prime({"fast": "anthropic:claude-haiku-4"})
    reset_alias_cache()
    assert cached_aliases(workspace_id=DEFAULT) == {}
    assert cache_is_stale()


def test_cached_aliases_is_a_copy() -> None:
    _prime({"fast": "anthropic:claude-haiku-4"})
    cached_aliases(workspace_id=DEFAULT)["fast"] = "tampered"
    assert resolve_effective_alias(CONFIG, "fast", workspace_id=DEFAULT) == "anthropic:claude-haiku-4"


# ---------------------------------------------------------------------------
# Workspace scoping
# ---------------------------------------------------------------------------


def test_two_workspaces_each_resolve_their_own_alias() -> None:
    """The whole point of widening the uniqueness constraint.

    While the cache was keyed on name alone the second row loaded would have
    shadowed the first for every caller, which is why the constraint could not
    allow both until this cache could tell them apart.
    """
    _prime({"fast": "anthropic:claude-haiku-4"}, workspace_id=DEFAULT)
    _prime({"fast": "home_lab:qwen3"}, workspace_id=OTHER)

    assert resolve_effective_alias(CONFIG, "fast", workspace_id=DEFAULT) == "anthropic:claude-haiku-4"
    assert resolve_effective_alias(CONFIG, "fast", workspace_id=OTHER) == "home_lab:qwen3"


def test_an_alias_does_not_leak_into_another_workspace() -> None:
    _prime({"mine": "home_lab:qwen3"}, workspace_id=OTHER)
    assert resolve_effective_alias(CONFIG, "mine", workspace_id=DEFAULT) is None


def test_a_user_scoped_alias_does_not_cross_workspaces() -> None:
    """Same user, two workspaces: the alias belongs to one of them, not to them."""
    _prime({}, {"alice": {"fast": "home_lab:qwen3"}}, workspace_id=OTHER)
    assert resolve_effective_alias(CONFIG, "fast", "alice", workspace_id=OTHER) == "home_lab:qwen3"
    assert resolve_effective_alias(CONFIG, "fast", "alice", workspace_id=DEFAULT) is None


def test_omitting_the_workspace_reads_the_default_one() -> None:
    """What a master-key request and an operator-configured selector get.

    They act deployment-wide, and the default workspace is where their own
    writes land, so it is what they resolve against.
    """
    _prime({"fast": "anthropic:claude-haiku-4"}, workspace_id=DEFAULT)
    _prime({"fast": "home_lab:qwen3"}, workspace_id=OTHER)
    assert resolve_effective_alias(CONFIG, "fast") == "anthropic:claude-haiku-4"


def test_a_config_alias_is_in_force_in_every_workspace() -> None:
    """It comes from a file the deployment owns, so it has no workspace."""
    _prime({}, workspace_id=OTHER)
    assert resolve_effective_alias(CONFIG, "configalias", workspace_id=OTHER) == "anthropic:claude-opus-4"


def test_no_default_workspace_leaves_only_the_config_layer() -> None:
    """A deployment with no workspace at all has no stored aliases to resolve."""
    _prime({"fast": "anthropic:claude-haiku-4"}, default_workspace=None)
    assert resolve_effective_alias(CONFIG, "fast") is None
    assert resolve_effective_alias(CONFIG, "configalias") == "anthropic:claude-opus-4"
    assert cached_aliases() == {}


# ---------------------------------------------------------------------------
# User scoping
# ---------------------------------------------------------------------------


def test_a_user_scoped_alias_resolves_only_for_that_user() -> None:
    _prime({}, {"alice": {"fast": "home_lab:qwen3"}})
    assert resolve_effective_alias(CONFIG, "fast", "alice", workspace_id=DEFAULT) == "home_lab:qwen3"
    assert resolve_effective_alias(CONFIG, "fast", "bob", workspace_id=DEFAULT) is None
    # A caller with no user (the master key) is not an implicit member of anyone.
    assert resolve_effective_alias(CONFIG, "fast", workspace_id=DEFAULT) is None


def test_a_user_scoped_alias_beats_the_workspace_wide_one() -> None:
    _prime({"fast": "anthropic:claude-haiku-4"}, {"alice": {"fast": "home_lab:qwen3"}})
    assert resolve_effective_alias(CONFIG, "fast", "alice", workspace_id=DEFAULT) == "home_lab:qwen3"
    assert resolve_effective_alias(CONFIG, "fast", "bob", workspace_id=DEFAULT) == "anthropic:claude-haiku-4"


def test_a_user_scoped_alias_beats_a_config_alias() -> None:
    # Most-specific-wins. The reverse holds for a *workspace-wide* stored alias
    # (see above), which is why the API refuses to create one shadowing config.
    _prime({}, {"alice": {"configalias": "home_lab:qwen3"}})
    assert resolve_effective_alias(CONFIG, "configalias", "alice", workspace_id=DEFAULT) == "home_lab:qwen3"
    assert (
        resolve_effective_alias(CONFIG, "configalias", "bob", workspace_id=DEFAULT)
        == "anthropic:claude-opus-4"
    )


def test_effective_aliases_layers_all_three_scopes() -> None:
    _prime({"fast": "anthropic:claude-haiku-4"}, {"alice": {"mine": "home_lab:qwen3"}})
    assert effective_aliases(CONFIG, "alice", workspace_id=DEFAULT) == {
        "fast": "anthropic:claude-haiku-4",
        "configalias": "anthropic:claude-opus-4",
        "mine": "home_lab:qwen3",
    }
    assert "mine" not in effective_aliases(CONFIG, "bob", workspace_id=DEFAULT)


def test_cached_aliases_returns_one_scope_at_a_time() -> None:
    _prime({"fast": "anthropic:claude-haiku-4"}, {"alice": {"mine": "home_lab:qwen3"}})
    assert cached_aliases(workspace_id=DEFAULT) == {"fast": "anthropic:claude-haiku-4"}
    assert cached_aliases("alice", workspace_id=DEFAULT) == {"mine": "home_lab:qwen3"}
    assert cached_aliases("bob", workspace_id=DEFAULT) == {}


def test_overriding_a_config_alias_drops_its_target_from_that_users_map() -> None:
    """The catalog withholds alias *targets*, so an override un-hides one.

    /v1/models hides every value in this map. When a user's alias replaces a
    config name, the configured target is no longer a value here, so it stops
    being withheld and reappears in that user's listing. Subtle enough to be
    worth pinning, and it inverts the usual "an alias hides its target" rule.
    """
    _prime({}, {"alice": {"configalias": "home_lab:qwen3"}})
    assert "anthropic:claude-opus-4" in set(effective_aliases(CONFIG, "bob", workspace_id=DEFAULT).values())
    assert "anthropic:claude-opus-4" not in set(
        effective_aliases(CONFIG, "alice", workspace_id=DEFAULT).values()
    )


def test_all_alias_names_spans_every_scope() -> None:
    # Scope-blind, for the writes that must never store an alias name as a model
    # key (pricing rows, allow-list entries). Blind to the workspace too: the
    # name means "alias" to somebody wherever the row lives.
    _prime({"fast": "anthropic:claude-haiku-4"}, {"alice": {"mine": "home_lab:qwen3"}})
    _prime({"elsewhere": "home_lab:qwen3"}, {"bob": {"theirs": "home_lab:qwen3"}}, workspace_id=OTHER)
    assert all_alias_names(CONFIG) == {"fast", "configalias", "mine", "elsewhere", "theirs"}


def test_reset_clears_the_user_layer_too() -> None:
    _prime({}, {"alice": {"mine": "home_lab:qwen3"}})
    reset_alias_cache()
    assert cached_aliases("alice", workspace_id=DEFAULT) == {}
    assert resolve_effective_alias(CONFIG, "mine", "alice", workspace_id=DEFAULT) is None
