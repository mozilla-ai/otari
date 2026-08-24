"""How a workspace's web-search row composes with the request's own tool entry.

The composition rule is the whole of #656's answer to #655, and it differs from
the hybrid path's on purpose: there the platform's policy supplies *defaults* a
request overrides, here the workspace's row *narrows* what the request asked
for. Everything a request could shed under default-only precedence is pinned
here, because shedding it is how a guardrail fails open.

The two ceilings the dashboard card repeats as literals are pinned at the
bottom, the same drift `test_code_execution_policy_limits.py` catches next door.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from gateway.services.tenancy.errors import WorkspaceWebSearchDomainsExcludedError
from gateway.services.tenancy.workspace_web_search_service import (
    _MAX_DOMAINS,
    _MAX_RESULTS,
    ResolvedWebSearchConfig,
    narrow_web_search_tool_entry,
)


def _config(**overrides: object) -> ResolvedWebSearchConfig:
    values: dict[str, object] = {
        "enabled": True,
        "max_results": None,
        "purpose_hint": None,
        "allowed_domains": None,
        "blocked_domains": None,
        "provider_options": None,
    }
    values.update(overrides)
    return ResolvedWebSearchConfig(**values)  # type: ignore[arg-type]


# What a request would get with no workspace row: the deployment's own setting,
# or the backend's built-in. `routes/_tools.web_search_max_results_baseline`
# answers it for real; the cases below vary it to say which one is in play.
_BASELINE = 5


def _narrow(
    entry: dict[str, Any],
    config: ResolvedWebSearchConfig,
    baseline: int = _BASELINE,
) -> dict[str, Any]:
    # `dict[str, Any]`, not `dict[str, object]`: `dict` is invariant, so a
    # literal entry whose values are all lists infers as `dict[str, Sequence[str]]`
    # and would not be assignable to the narrower annotation.
    return narrow_web_search_tool_entry(entry, config, baseline_max_results=baseline)


def test_a_row_that_narrows_nothing_leaves_the_entry_alone() -> None:
    entry = {"type": "otari_web_search", "max_results": 8}

    assert _narrow(entry, _config()) == entry


def test_the_caller_s_entry_is_never_mutated() -> None:
    """It is the dict extracted from the request body, and the caller still holds it."""
    entry = {"type": "otari_web_search", "max_results": 8}

    narrowed = _narrow(entry, _config(max_results=2, purpose_hint="Cite sources"))

    assert entry == {"type": "otari_web_search", "max_results": 8}
    assert narrowed["max_results"] == 2


@pytest.mark.parametrize(
    ("requested", "ceiling", "expected"),
    [
        (8, 2, 2),  # the workspace lowers what the request asked for
        (2, 8, 2),  # and never raises it
        (None, 3, 3),  # a request that named none is lowered from the baseline
        # The widening case: a ceiling above the deployment's own number must
        # leave that number alone rather than replacing it.
        (None, 9, _BASELINE),
        (True, 4, 4),  # a JSON `true` is not a one-result ceiling
        (0, 4, 4),  # nor is a nonsense value the request sent
    ],
)
def test_max_results_is_floored_never_raised(requested: object, ceiling: int, expected: int) -> None:
    entry: dict[str, object] = {"type": "otari_web_search"}
    if requested is not None:
        entry["max_results"] = requested

    narrowed = _narrow(entry, _config(max_results=ceiling))

    assert narrowed["max_results"] == expected


def test_a_workspace_that_named_no_ceiling_leaves_the_entry_untouched() -> None:
    """No ceiling means no narrowing, so nothing is written where nothing was."""
    assert "max_results" not in _narrow({"type": "otari_web_search"}, _config())


def test_a_block_list_is_added_to_rather_than_replaced() -> None:
    """The fail-open case: a request must not shed its workspace's blocks by sending its own."""
    entry: dict[str, object] = {"type": "otari_web_search", "blocked_domains": ["noise.example"]}

    narrowed = _narrow(entry, _config(blocked_domains=("banned.example",)))

    assert narrowed["blocked_domains"] == ["noise.example", "banned.example"]


def test_a_block_list_is_applied_to_a_request_that_named_none() -> None:
    narrowed = _narrow({"type": "otari_web_search"}, _config(blocked_domains=("banned.example",)))

    assert narrowed["blocked_domains"] == ["banned.example"]


def test_an_allow_list_is_intersected_rather_than_replaced() -> None:
    entry: dict[str, object] = {"type": "otari_web_search", "allowed_domains": ["arxiv.org", "elsewhere.example"]}

    narrowed = _narrow(entry, _config(allowed_domains=("arxiv.org", "wikipedia.org")))

    assert narrowed["allowed_domains"] == ["arxiv.org"]


def test_an_allow_list_applies_whole_to_a_request_that_named_none() -> None:
    narrowed = _narrow({"type": "otari_web_search"}, _config(allowed_domains=("arxiv.org",)))

    assert narrowed["allowed_domains"] == ["arxiv.org"]


def test_a_disjoint_allow_list_is_refused_rather_than_emptied() -> None:
    """An empty allow-list reads as *no* allow-list downstream, so it cannot be the answer.

    `_build_web_search_backend` applies the field only when it is truthy, so
    narrowing to `[]` would turn the narrowest possible policy into none at all.
    """
    entry: dict[str, object] = {"type": "otari_web_search", "allowed_domains": ["elsewhere.example"]}

    with pytest.raises(WorkspaceWebSearchDomainsExcludedError):
        _narrow(entry, _config(allowed_domains=("arxiv.org",)))


def test_a_request_naming_a_subdomain_of_a_permitted_domain_keeps_it() -> None:
    """A list entry is a suffix, not a host, so the narrower side of a pair survives.

    `WebSearchBackend._apply_domain_filters` keeps a result whose hostname equals an
    entry *or ends in* `"." + entry`, so a workspace allowing `example.com`
    already permits `docs.example.com`. A request naming the subdomain is asking
    for strictly less, and comparing the two as opaque strings would refuse it.
    """
    entry: dict[str, object] = {"type": "otari_web_search", "allowed_domains": ["docs.example.com"]}

    narrowed = _narrow(entry, _config(allowed_domains=("example.com",)))

    assert narrowed["allowed_domains"] == ["docs.example.com"]


def test_a_workspace_subdomain_narrows_a_request_that_named_the_parent() -> None:
    """The other direction: the workspace is the narrower side, so its entry is the answer."""
    entry: dict[str, object] = {"type": "otari_web_search", "allowed_domains": ["example.com"]}

    narrowed = _narrow(entry, _config(allowed_domains=("docs.example.com",)))

    assert narrowed["allowed_domains"] == ["docs.example.com"]


def test_a_permitted_subdomain_is_not_dropped_alongside_an_exact_match() -> None:
    """The silent case: an overlapping entry must not vanish because another matched exactly."""
    entry: dict[str, object] = {"type": "otari_web_search", "allowed_domains": ["docs.example.com", "arxiv.org"]}

    narrowed = _narrow(entry, _config(allowed_domains=("example.com", "arxiv.org")))

    assert narrowed["allowed_domains"] == ["docs.example.com", "arxiv.org"]


def test_a_domain_that_merely_ends_in_another_is_not_a_subdomain_of_it() -> None:
    """`notexample.com` is not under `example.com`; only a dot-separated suffix is."""
    entry: dict[str, object] = {"type": "otari_web_search", "allowed_domains": ["notexample.com"]}

    with pytest.raises(WorkspaceWebSearchDomainsExcludedError):
        _narrow(entry, _config(allowed_domains=("example.com",)))


def test_domains_are_compared_case_insensitively() -> None:
    entry: dict[str, object] = {"type": "otari_web_search", "allowed_domains": ["ArXiv.ORG"]}

    narrowed = _narrow(entry, _config(allowed_domains=("arxiv.org",)))

    assert narrowed["allowed_domains"] == ["arxiv.org"]


def test_a_hint_fills_a_gap_and_never_overrides_the_request_s_own() -> None:
    """A hint informs the model; it does not permit anything, so the request wins."""
    assert (
        _narrow({"type": "otari_web_search"}, _config(purpose_hint="Workspace hint"))["purpose_hint"]
        == "Workspace hint"
    )
    assert (
        _narrow(
            {"type": "otari_web_search", "purpose_hint": "Request hint"},
            _config(purpose_hint="Workspace hint"),
        )["purpose_hint"]
        == "Request hint"
    )


def test_provider_options_merge_per_key_with_the_request_winning() -> None:
    """The one field that keeps the hybrid precedence: an opaque bag has no narrowing relation."""
    entry: dict[str, object] = {"type": "otari_web_search", "provider_options": {"topic": "news"}}

    narrowed = _narrow(entry, _config(provider_options={"topic": "general", "search_depth": "basic"}))

    assert narrowed["provider_options"] == {"topic": "news", "search_depth": "basic"}


_CARD = Path(__file__).resolve().parents[2] / "web" / "src" / "features" / "tools" / "WorkspaceWebSearchCard.tsx"


def _card_constant(name: str) -> int:
    match = re.search(rf"^export const {name} = (\d+)$", _CARD.read_text(), re.MULTILINE)
    assert match is not None, f"{name} is no longer declared in {_CARD.name}; update this test with it"
    return int(match.group(1))


@pytest.mark.parametrize(
    ("name", "server_value"),
    [("MAX_RESULTS", _MAX_RESULTS), ("MAX_DOMAINS", _MAX_DOMAINS)],
)
def test_the_card_repeats_the_ceiling_the_server_enforces(name: str, server_value: int) -> None:
    """The card cannot import a Python constant, so it repeats both and this pins the pair.

    Raising `web_search_backend.MAX_RESULTS_CAP` without the card following
    leaves the form refusing a value the API would accept, with the failure
    showing up nowhere.
    """
    assert _card_constant(name) == server_value, (
        f"{name} in {_CARD.name} is {_card_constant(name)} but the server enforces {server_value}, "
        "so the form and the API disagree about which values are acceptable."
    )
