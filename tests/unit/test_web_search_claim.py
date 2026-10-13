"""Who runs a provider-named web-search declaration: the provider or the gateway."""

from __future__ import annotations

from typing import Any

import pytest

from gateway.core.config import GatewayConfig
from gateway.exceptions.tools_exceptions import WebSearchInterceptedError
from gateway.models.tools import CodeExecutor
from gateway.services.tools import WEB_SEARCH_TOOL_NAME, Dialect, decide_candidate_tools, read_web_declarations
from gateway.services.tools._web_declarations import (
    first_provider_web_search_tool,
    parse_web_search_header,
    provider_runs_web_search_natively,
    web_search_header_conflicts,
)

ANTHROPIC_DATED = {"type": "web_search_20250305", "name": "web_search"}
OPENAI_BARE = {"type": "web_search"}


@pytest.mark.parametrize(
    ("entry", "provider", "dialect", "native"),
    [
        (ANTHROPIC_DATED, "anthropic", Dialect.MESSAGES, True),
        (OPENAI_BARE, "openai", Dialect.RESPONSES, True),
        ({"type": "web_search_preview"}, "openai", Dialect.RESPONSES, True),
        ({"type": "web_search_preview_2025_03_11"}, "openai", Dialect.RESPONSES, True),
        (ANTHROPIC_DATED, "openai", Dialect.RESPONSES, False),
        (OPENAI_BARE, "anthropic", Dialect.MESSAGES, False),
        ({"type": "web_search_preview"}, "anthropic", Dialect.MESSAGES, False),
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


@pytest.fixture(autouse=True)
def _no_search_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OTARI_WEB_SEARCH_URL", "OTARI_WEB_SEARCH_INTERCEPT", "OTARI_WEB_SEARCH_PROVIDER"):
        monkeypatch.delenv(name, raising=False)


def _claimed(
    *,
    requested: CodeExecutor | None,
    intercept: bool = False,
    backend_configured: bool = True,
    providers: list[str | None],
    entry: dict[str, Any] | None = ANTHROPIC_DATED,
    uniform: bool = False,
) -> list[bool]:
    """Whether Otari runs the search on each candidate in ``providers``, head first."""
    config = GatewayConfig(
        web_search_url="http://search.invalid" if backend_configured else None, web_search_intercept=intercept
    )
    declarations = read_web_declarations(
        [entry] if entry is not None else [],
        web_search_header=requested.value if requested is not None else None,
        config=config,
        uniform=uniform,
    )
    decided = decide_candidate_tools(declarations, providers=providers, dialect=Dialect.MESSAGES)
    return [WEB_SEARCH_TOOL_NAME in candidate.claimed for candidate in decided]


def _claims(*, provider: str | None = "bedrock", **kwargs: Any) -> bool:
    return _claimed(providers=[provider], **kwargs)[0]


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


@pytest.mark.parametrize(
    ("providers", "claimed"),
    [
        (["anthropic", "anthropic"], [False, False]),
        (["anthropic", "bedrock"], [False, True]),
        (["bedrock", "anthropic"], [True, False]),
        (["anthropic", None], [False, True]),
    ],
    ids=["all-native", "non-native-fallback", "non-native-head", "unknown-fallback"],
)
def test_auto_decides_each_candidate_on_its_own(providers: list[str | None], claimed: list[bool]) -> None:
    assert _claimed(requested=CodeExecutor.AUTO, providers=providers) == claimed


def test_a_uniform_auto_claims_for_every_candidate_unless_all_run_the_search() -> None:
    """How hybrid mode decides, since the platform's attempts share one set of tools."""
    assert _claimed(requested=CodeExecutor.AUTO, providers=["anthropic", "bedrock"], uniform=True) == [True, True]
    assert _claimed(requested=CodeExecutor.AUTO, providers=["anthropic", "anthropic"], uniform=True) == [False, False]


@pytest.mark.parametrize("requested", [None, CodeExecutor.AUTO, CodeExecutor.OTARI])
def test_interception_claims_whatever_else_the_header_asks(requested: CodeExecutor | None) -> None:
    assert _claims(requested=requested, intercept=True, provider="anthropic")


def test_the_header_cannot_undo_interception() -> None:
    with pytest.raises(WebSearchInterceptedError):
        _claims(requested=CodeExecutor.PROVIDER, intercept=True, provider="anthropic")


def test_otari_claims_even_a_native_search() -> None:
    assert _claims(requested=CodeExecutor.OTARI, provider="anthropic")


@pytest.mark.parametrize(
    ("requested", "intercept", "conflicts"),
    [
        (CodeExecutor.PROVIDER, True, True),
        (CodeExecutor.AUTO, True, False),
        (CodeExecutor.OTARI, True, False),
        (None, True, False),
        (CodeExecutor.PROVIDER, False, False),
    ],
)
def test_only_provider_under_interception_conflicts(
    requested: CodeExecutor | None, intercept: bool, conflicts: bool
) -> None:
    assert web_search_header_conflicts(requested, intercept=intercept) is conflicts


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
