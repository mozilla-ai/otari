"""Who runs a provider-named web-search declaration: the provider or the gateway."""

from __future__ import annotations

from typing import Any

import pytest

from gateway.api.routes._tools import (
    claims_provider_web_search,
    first_provider_web_search_tool,
    parse_web_search_header,
    provider_runs_web_search_natively,
)
from gateway.models.tools import CodeExecutor
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


def _claims(
    *,
    requested: CodeExecutor | None,
    intercept: bool = False,
    backend_configured: bool = True,
    provider: str | None = "bedrock",
    entry: dict[str, Any] | None = ANTHROPIC_DATED,
) -> bool:
    return claims_provider_web_search(
        entry,
        requested=requested,
        intercept=intercept,
        backend_configured=backend_configured,
        provider=provider,
        dialect=Dialect.MESSAGES,
    )


@pytest.mark.parametrize("provider", ["anthropic", "bedrock"])
def test_without_the_header_or_interception_nothing_is_claimed(provider: str) -> None:
    assert not _claims(requested=None, provider=provider)


@pytest.mark.parametrize("provider", ["anthropic", "bedrock"])
def test_without_the_header_interception_claims_everything(provider: str) -> None:
    assert _claims(requested=None, intercept=True, provider=provider)


def test_auto_keeps_a_providers_own_search() -> None:
    assert not _claims(requested=CodeExecutor.AUTO, provider="anthropic")


def test_auto_claims_a_search_the_provider_cannot_run() -> None:
    assert _claims(requested=CodeExecutor.AUTO, provider="bedrock")


def test_auto_wins_over_interception() -> None:
    assert not _claims(requested=CodeExecutor.AUTO, intercept=True, provider="anthropic")


def test_otari_claims_even_a_native_search() -> None:
    assert _claims(requested=CodeExecutor.OTARI, provider="anthropic")


def test_provider_wins_over_interception() -> None:
    assert not _claims(requested=CodeExecutor.PROVIDER, intercept=True, provider="bedrock")


@pytest.mark.parametrize("requested", [None, *CodeExecutor])
def test_nothing_is_claimed_without_a_backend(requested: CodeExecutor | None) -> None:
    assert not _claims(requested=requested, intercept=True, backend_configured=False)


def test_nothing_is_claimed_without_a_provider_keyword() -> None:
    assert not _claims(requested=CodeExecutor.OTARI, intercept=True, entry=None)


def test_a_function_named_web_search_is_not_a_keyword() -> None:
    tools: list[dict[str, Any]] = [{"type": "function", "function": {"name": "web_search"}}, ANTHROPIC_DATED]
    assert first_provider_web_search_tool(tools) is ANTHROPIC_DATED


@pytest.mark.parametrize(("value", "expected"), [(None, None), ("  ", None), (" AUTO ", CodeExecutor.AUTO)])
def test_header_parses_case_insensitively(value: str | None, expected: CodeExecutor | None) -> None:
    assert parse_web_search_header(value) is expected


def test_an_unknown_header_value_is_an_error() -> None:
    with pytest.raises(ValueError, match="Otari-Web-Search"):
        parse_web_search_header("gateway")
