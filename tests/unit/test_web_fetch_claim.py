"""Who runs Anthropic's web-fetch keyword: the provider or the gateway."""

from __future__ import annotations

from typing import Any

import pytest

from gateway.core.config import GatewayConfig
from gateway.exceptions.tools_exceptions import WebToolDeclarationError
from gateway.models.tools import CodeExecutor
from gateway.services.tools import (
    WEB_FETCH_TOOL_NAME,
    WEB_SEARCH_TOOL_NAME,
    Dialect,
    WebClaims,
    decide_candidate_tools,
    extract_web_tools,
    read_web_declarations,
    validate_web_declarations,
)
from gateway.services.tools._web_declarations import (
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


def _claimed(
    *,
    requested: CodeExecutor | None,
    providers: list[str | None],
    fetch_enabled: bool = True,
    entry: dict[str, Any] | None = ANTHROPIC_FETCH,
) -> list[bool]:
    """Whether Otari runs the fetch on each candidate in ``providers``, head first."""
    declarations = read_web_declarations(
        [entry] if entry is not None else [],
        web_search_header=None,
        web_fetch_header=requested.value if requested is not None else None,
        config=GatewayConfig(web_fetch_enabled=fetch_enabled),
        uniform=False,
    )
    decided = decide_candidate_tools(declarations, providers=providers, dialect=Dialect.MESSAGES)
    return [WEB_FETCH_TOOL_NAME in candidate.claimed for candidate in decided]


def _claims(*, providers: list[str | None], **kwargs: Any) -> bool:
    """Whether Otari runs the fetch on any of ``providers``."""
    return any(_claimed(providers=providers, **kwargs))


@pytest.mark.parametrize("provider", ["anthropic", "gemini"])
def test_without_the_header_nothing_is_claimed(provider: str) -> None:
    assert not _claims(requested=None, providers=[provider])


@pytest.mark.parametrize(
    ("providers", "claimed"),
    [
        (["anthropic"], [False]),
        (["anthropic", "anthropic"], [False, False]),
        (["gemini"], [True]),
        (["anthropic", "gemini"], [False, True]),
        ([None], [True]),
    ],
    ids=["anthropic", "all-anthropic", "non-anthropic", "non-anthropic-fallback", "unknown-provider"],
)
def test_auto_decides_each_candidate_on_its_own(providers: list[str | None], claimed: list[bool]) -> None:
    assert _claimed(requested=CodeExecutor.AUTO, providers=providers) == claimed


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
    declarations = read_web_declarations(
        tools, web_search_header=None, web_fetch_header=fetch_header, config=_config(), uniform=False
    )
    claimed = decide_candidate_tools(declarations, providers=[provider], dialect=Dialect.MESSAGES)[0].claimed
    return WebClaims(search=WEB_SEARCH_TOOL_NAME in claimed, fetch=WEB_FETCH_TOOL_NAME in claimed)


def test_the_fetch_header_does_not_claim_a_search_keyword() -> None:
    claims = _claim([dict(ANTHROPIC_FETCH), {"type": "web_search_20250305"}], fetch_header="auto")
    assert claims == WebClaims(search=False, fetch=True)


def test_a_claimed_keyword_beside_otari_web_fetch_is_a_duplicate() -> None:
    with pytest.raises(WebToolDeclarationError, match="at most once"):
        validate_web_declarations([dict(ANTHROPIC_FETCH), {"type": "otari_web_fetch"}], claims=WebClaims(fetch=True))


def test_an_unclaimed_keyword_beside_otari_web_fetch_is_not_a_duplicate() -> None:
    validate_web_declarations([dict(ANTHROPIC_FETCH), {"type": "otari_web_fetch"}], claims=WebClaims())


def test_a_claimed_keyword_reserves_the_function_name() -> None:
    caller_function = {"type": "function", "function": {"name": "web_fetch"}}
    with pytest.raises(WebToolDeclarationError, match="reserved"):
        validate_web_declarations([dict(ANTHROPIC_FETCH), caller_function], claims=WebClaims(fetch=True))


def test_a_claimed_keyword_is_taken_out_of_the_forwarded_tools() -> None:
    entry = {**ANTHROPIC_FETCH, "allowed_domains": ["Docs.Example.com."]}
    other = {"name": "lookup", "input_schema": {"type": "object"}}

    web = extract_web_tools([entry, other], config=_config(), claims=WebClaims(fetch=True))

    assert web.fetch_tool_entry == {**ANTHROPIC_FETCH, "allowed_domains": ["docs.example.com"]}
    assert web.remaining_user_tools == [other]
    # A copy: another candidate of the same request may forward the caller's own entry.
    assert entry["allowed_domains"] == ["Docs.Example.com."]


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
