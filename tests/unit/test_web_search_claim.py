"""Who runs a provider-named web-search declaration: the provider or the gateway."""

from __future__ import annotations

import pytest

from gateway.api.routes._tools import (
    claims_provider_web_search,
    first_provider_web_search_tool,
    provider_runs_web_search_natively,
)
from gateway.services.tools import Dialect

ANTHROPIC_DATED = {"type": "web_search_20250305", "name": "web_search"}
OPENAI_BARE = {"type": "web_search"}


@pytest.mark.parametrize(
    ("entry", "provider", "dialect", "native"),
    [
        (ANTHROPIC_DATED, "anthropic", Dialect.MESSAGES, True),
        (OPENAI_BARE, "openai", Dialect.RESPONSES, True),
        ({"type": "web_search_preview"}, "openai", Dialect.RESPONSES, True),
        (ANTHROPIC_DATED, "bedrock", Dialect.MESSAGES, False),
        (ANTHROPIC_DATED, "openai", Dialect.MESSAGES, False),
        (ANTHROPIC_DATED, "anthropic", Dialect.CHAT, False),
        (ANTHROPIC_DATED, None, Dialect.MESSAGES, False),
        ({"type": "otari_web_search"}, "anthropic", Dialect.MESSAGES, False),
    ],
)
def test_native_only_in_the_providers_own_wire_format(
    entry: dict[str, str], provider: str | None, dialect: Dialect, native: bool
) -> None:
    assert provider_runs_web_search_natively(entry, provider=provider, dialect=dialect) is native


def test_a_provider_that_can_search_keeps_its_search() -> None:
    assert not claims_provider_web_search(
        ANTHROPIC_DATED, intercept=False, backend_configured=True, provider="anthropic", dialect=Dialect.MESSAGES
    )


def test_a_provider_that_cannot_search_has_it_claimed() -> None:
    assert claims_provider_web_search(
        ANTHROPIC_DATED, intercept=False, backend_configured=True, provider="bedrock", dialect=Dialect.MESSAGES
    )


def test_interception_claims_even_a_native_search() -> None:
    assert claims_provider_web_search(
        ANTHROPIC_DATED, intercept=True, backend_configured=True, provider="anthropic", dialect=Dialect.MESSAGES
    )


@pytest.mark.parametrize("intercept", [True, False])
def test_nothing_is_claimed_without_a_backend(intercept: bool) -> None:
    assert not claims_provider_web_search(
        ANTHROPIC_DATED, intercept=intercept, backend_configured=False, provider="bedrock", dialect=Dialect.MESSAGES
    )


def test_nothing_is_claimed_without_a_provider_keyword() -> None:
    assert not claims_provider_web_search(
        None, intercept=True, backend_configured=True, provider="bedrock", dialect=Dialect.MESSAGES
    )


def test_a_function_named_web_search_is_not_a_keyword() -> None:
    tools = [{"type": "function", "function": {"name": "web_search"}}, ANTHROPIC_DATED]
    assert first_provider_web_search_tool(tools) is ANTHROPIC_DATED
