"""Unit tests for the alias cache that keeps resolution synchronous."""

import time
from collections.abc import Iterator

import pytest

from gateway.core.config import GatewayConfig
from gateway.services import alias_service
from gateway.services.alias_service import (
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


def _prime(aliases: dict[str, str]) -> None:
    """Stand in for a database load without needing a session.

    Stamped with monotonic(), which is process uptime rather than an epoch: a
    literal 0.0 would read as loaded-at-boot and so already ancient.
    """
    alias_service._cache.clear()
    alias_service._cache.update(aliases)
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
