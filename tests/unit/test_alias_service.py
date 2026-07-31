"""Unit tests for the alias cache that keeps resolution synchronous."""

import time
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


@pytest.fixture(autouse=True)
def _clean_cache() -> Iterator[None]:
    reset_alias_cache()
    yield
    reset_alias_cache()


def _prime(aliases: dict[str, str], per_user: dict[str, dict[str, str]] | None = None) -> None:
    """Stand in for a database load without needing a session.

    Stamped with monotonic(), which is process uptime rather than an epoch: a
    literal 0.0 would read as loaded-at-boot and so already ancient.
    """
    alias_service._cache.clear()
    alias_service._cache.update(aliases)
    alias_service._user_cache.clear()
    alias_service._user_cache.update(per_user or {})
    alias_service._cached_at = time.monotonic()


def test_config_aliases_resolve_without_any_stored_ones() -> None:
    assert resolve_effective_alias(CONFIG, "configalias") == "anthropic:claude-opus-4"


def test_stored_aliases_resolve() -> None:
    _prime({"fast": "anthropic:claude-haiku-4"})
    assert resolve_effective_alias(CONFIG, "fast") == "anthropic:claude-haiku-4"


def test_unknown_name_is_not_an_alias() -> None:
    _prime({"fast": "anthropic:claude-haiku-4"})
    assert resolve_effective_alias(CONFIG, "openai:gpt-4o") is None


def test_both_kinds_are_merged() -> None:
    _prime({"fast": "anthropic:claude-haiku-4"})
    assert effective_aliases(CONFIG) == {
        "fast": "anthropic:claude-haiku-4",
        "configalias": "anthropic:claude-opus-4",
    }


def test_config_wins_over_a_stored_alias_of_the_same_name() -> None:
    # The API refuses to create this, so it is a safety net: if a row somehow
    # exists, the config alias is what resolves, matching what the listing says.
    _prime({"configalias": "home_lab:qwen3"})
    assert resolve_effective_alias(CONFIG, "configalias") == "anthropic:claude-opus-4"


def test_empty_target_is_not_an_alias() -> None:
    _prime({"broken": ""})
    assert resolve_effective_alias(CONFIG, "broken") is None


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
    assert cached_aliases() == {}
    assert cache_is_stale()


def test_cached_aliases_is_a_copy() -> None:
    _prime({"fast": "anthropic:claude-haiku-4"})
    cached_aliases()["fast"] = "tampered"
    assert resolve_effective_alias(CONFIG, "fast") == "anthropic:claude-haiku-4"


# ---------------------------------------------------------------------------
# Scoping
# ---------------------------------------------------------------------------


def test_a_user_scoped_alias_resolves_only_for_that_user() -> None:
    _prime({}, {"alice": {"fast": "home_lab:qwen3"}})
    assert resolve_effective_alias(CONFIG, "fast", "alice") == "home_lab:qwen3"
    assert resolve_effective_alias(CONFIG, "fast", "bob") is None
    # A caller with no user (the master key) is not an implicit member of anyone.
    assert resolve_effective_alias(CONFIG, "fast") is None


def test_a_user_scoped_alias_beats_the_global_one() -> None:
    _prime({"fast": "anthropic:claude-haiku-4"}, {"alice": {"fast": "home_lab:qwen3"}})
    assert resolve_effective_alias(CONFIG, "fast", "alice") == "home_lab:qwen3"
    assert resolve_effective_alias(CONFIG, "fast", "bob") == "anthropic:claude-haiku-4"


def test_a_user_scoped_alias_beats_a_config_alias() -> None:
    # Most-specific-wins. The reverse holds for a *global* stored alias (see
    # above), which is why the API refuses to create one shadowing config.
    _prime({}, {"alice": {"configalias": "home_lab:qwen3"}})
    assert resolve_effective_alias(CONFIG, "configalias", "alice") == "home_lab:qwen3"
    assert resolve_effective_alias(CONFIG, "configalias", "bob") == "anthropic:claude-opus-4"


def test_effective_aliases_layers_all_three_scopes() -> None:
    _prime({"fast": "anthropic:claude-haiku-4"}, {"alice": {"mine": "home_lab:qwen3"}})
    assert effective_aliases(CONFIG, "alice") == {
        "fast": "anthropic:claude-haiku-4",
        "configalias": "anthropic:claude-opus-4",
        "mine": "home_lab:qwen3",
    }
    assert "mine" not in effective_aliases(CONFIG, "bob")


def test_cached_aliases_returns_one_scope_at_a_time() -> None:
    _prime({"fast": "anthropic:claude-haiku-4"}, {"alice": {"mine": "home_lab:qwen3"}})
    assert cached_aliases() == {"fast": "anthropic:claude-haiku-4"}
    assert cached_aliases("alice") == {"mine": "home_lab:qwen3"}
    assert cached_aliases("bob") == {}


def test_overriding_a_config_alias_drops_its_target_from_that_users_map() -> None:
    """The catalogue withholds alias *targets*, so an override un-hides one.

    /v1/models hides every value in this map. When a user's alias replaces a
    config name, the configured target is no longer a value here, so it stops
    being withheld and reappears in that user's listing. Subtle enough to be
    worth pinning, and it inverts the usual "an alias hides its target" rule.
    """
    _prime({}, {"alice": {"configalias": "home_lab:qwen3"}})
    assert "anthropic:claude-opus-4" in set(effective_aliases(CONFIG, "bob").values())
    assert "anthropic:claude-opus-4" not in set(effective_aliases(CONFIG, "alice").values())


def test_all_alias_names_spans_every_scope() -> None:
    # Scope-blind, for the writes that must never store an alias name as a model
    # key (pricing rows, allow-list entries).
    _prime({"fast": "anthropic:claude-haiku-4"}, {"alice": {"mine": "home_lab:qwen3"}})
    assert all_alias_names(CONFIG) == {"fast", "configalias", "mine"}


def test_reset_clears_the_user_layer_too() -> None:
    _prime({}, {"alice": {"mine": "home_lab:qwen3"}})
    reset_alias_cache()
    assert cached_aliases("alice") == {}
    assert resolve_effective_alias(CONFIG, "mine", "alice") is None
