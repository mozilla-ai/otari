"""Who runs Anthropic's web-fetch keyword: the provider or the gateway."""

from __future__ import annotations

from typing import Any

import pytest

from gateway.core.config import GatewayConfig
from gateway.exceptions.tools_exceptions import WebToolDeclarationError
from gateway.models.tools import CodeExecutor
from gateway.services.tools import Dialect, WebClaims, claim_web_declarations, extract_web_tools
from gateway.services.tools._web_declarations import (
    claims_provider_web_fetch,
    parse_web_fetch_header,
    provider_runs_web_fetch_natively,
)

ANTHROPIC_FETCH = {"type": "web_fetch_20250910", "name": "web_fetch"}


@pytest.mark.parametrize(
    ("entry", "provider", "dialect", "native"),
    [
        (ANTHROPIC_FETCH, "anthropic", Dialect.MESSAGES, True),
        ({"type": "web_fetch_20260318", "name": "web_fetch"}, "Anthropic", Dialect.MESSAGES, True),
        (ANTHROPIC_FETCH, "anthropic", Dialect.RESPONSES, False),
        (ANTHROPIC_FETCH, "anthropic", Dialect.CHAT, False),
        (ANTHROPIC_FETCH, "openai", Dialect.MESSAGES, False),
        (ANTHROPIC_FETCH, "bedrock", Dialect.MESSAGES, False),
        (ANTHROPIC_FETCH, None, Dialect.MESSAGES, False),
        ({"type": "web_fetch_preview"}, "anthropic", Dialect.MESSAGES, False),
        ({"type": "otari_web_fetch"}, "anthropic", Dialect.MESSAGES, False),
    ],
)
def test_only_anthropics_dated_keyword_on_messages_is_native(
    entry: dict[str, str], provider: str | None, dialect: Dialect, native: bool
) -> None:
    assert provider_runs_web_fetch_natively(entry, provider=provider, dialect=dialect) is native


def _claims(
    *,
    requested: CodeExecutor | None,
    providers: list[str | None],
    fetch_enabled: bool = True,
    entry: dict[str, Any] | None = ANTHROPIC_FETCH,
) -> bool:
    return claims_provider_web_fetch(
        entry, requested=requested, fetch_enabled=fetch_enabled, providers=providers, dialect=Dialect.MESSAGES
    )


@pytest.mark.parametrize("provider", ["anthropic", "gemini"])
def test_without_the_header_nothing_is_claimed(provider: str) -> None:
    assert not _claims(requested=None, providers=[provider])


@pytest.mark.parametrize(
    ("providers", "claimed"),
    [
        (["anthropic"], False),
        (["anthropic", "anthropic"], False),
        (["gemini"], True),
        (["anthropic", "gemini"], True),
        ([None], True),
        ([], True),
    ],
    ids=["anthropic", "all-anthropic", "non-anthropic", "non-anthropic-fallback", "unknown-provider", "unknown-chain"],
)
def test_auto_claims_unless_every_candidate_runs_the_fetch(providers: list[str | None], claimed: bool) -> None:
    assert _claims(requested=CodeExecutor.AUTO, providers=providers) is claimed


def test_otari_claims_even_anthropics_own_fetch() -> None:
    assert _claims(requested=CodeExecutor.OTARI, providers=["anthropic"])


def test_provider_never_claims() -> None:
    assert not _claims(requested=CodeExecutor.PROVIDER, providers=["gemini"])


@pytest.mark.parametrize("requested", [None, *CodeExecutor])
def test_nothing_is_claimed_with_fetch_disabled(requested: CodeExecutor | None) -> None:
    assert not _claims(requested=requested, providers=["gemini"], fetch_enabled=False)


def test_nothing_is_claimed_without_a_fetch_keyword() -> None:
    assert not _claims(requested=CodeExecutor.OTARI, providers=["gemini"], entry=None)


@pytest.mark.parametrize(("value", "expected"), [(None, None), ("  ", None), (" Otari ", CodeExecutor.OTARI)])
def test_header_parses_case_insensitively(value: str | None, expected: CodeExecutor | None) -> None:
    assert parse_web_fetch_header(value) is expected


def test_an_unknown_header_value_is_an_error() -> None:
    with pytest.raises(ValueError, match="Otari-Web-Fetch"):
        parse_web_fetch_header("gateway")


# --- admission -----------------------------------------------------------------


def _config() -> GatewayConfig:
    return GatewayConfig(web_fetch_enabled=True)


def _claim(tools: list[dict[str, Any]], *, fetch_header: str | None, provider: str = "gemini") -> WebClaims:
    return claim_web_declarations(
        tools,
        web_search_header=None,
        web_fetch_header=fetch_header,
        config=_config(),
        providers=[provider],
        dialect=Dialect.MESSAGES,
    )


def test_the_fetch_header_does_not_claim_a_search_keyword() -> None:
    claims = _claim([dict(ANTHROPIC_FETCH), {"type": "web_search_20250305"}], fetch_header="auto")
    assert claims == WebClaims(search=False, fetch=True)


def test_a_claimed_keyword_beside_otari_web_fetch_is_a_duplicate() -> None:
    with pytest.raises(WebToolDeclarationError, match="at most once"):
        _claim([dict(ANTHROPIC_FETCH), {"type": "otari_web_fetch"}], fetch_header="auto")


def test_an_unclaimed_keyword_beside_otari_web_fetch_is_not_a_duplicate() -> None:
    assert _claim([dict(ANTHROPIC_FETCH), {"type": "otari_web_fetch"}], fetch_header=None) == WebClaims()


def test_a_claimed_keyword_reserves_the_function_name() -> None:
    caller_function = {"type": "function", "function": {"name": "web_fetch"}}
    with pytest.raises(WebToolDeclarationError, match="reserved"):
        _claim([dict(ANTHROPIC_FETCH), caller_function], fetch_header="auto")


def test_a_claimed_keyword_is_taken_out_of_the_forwarded_tools() -> None:
    entry = {**ANTHROPIC_FETCH, "allowed_domains": ["Docs.Example.com."]}
    other = {"name": "lookup", "input_schema": {"type": "object"}}

    web = extract_web_tools([entry, other], config=_config(), claims=WebClaims(fetch=True))

    assert web.fetch_tool_entry is entry
    assert entry["allowed_domains"] == ["docs.example.com"]
    assert web.remaining_user_tools == [other]


def test_an_unclaimed_keyword_is_left_for_the_provider() -> None:
    web = extract_web_tools([dict(ANTHROPIC_FETCH)], config=_config(), claims=WebClaims())

    assert web.fetch_tool_entry is None
    assert web.remaining_user_tools == [ANTHROPIC_FETCH]


@pytest.mark.parametrize("max_uses", [-1, True, "2", 1.5])
def test_an_invalid_max_uses_is_refused_rather_than_uncapped(max_uses: Any) -> None:
    with pytest.raises(WebToolDeclarationError, match="web_fetch max_uses"):
        extract_web_tools([{**ANTHROPIC_FETCH, "max_uses": max_uses}], config=_config(), claims=WebClaims(fetch=True))


def test_an_invalid_domain_list_is_refused() -> None:
    with pytest.raises(WebToolDeclarationError, match="Web fetch allowed_domains"):
        extract_web_tools(
            [{**ANTHROPIC_FETCH, "allowed_domains": ["https://example.com/path"]}],
            config=_config(),
            claims=WebClaims(fetch=True),
        )
