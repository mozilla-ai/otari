"""Apply a workspace's web search policy to the web access one request declared.

A policy may veto and may narrow, and it never grants.
The rule is the same whichever plane the policy came from.
:func:`narrow_web_search_tool_entry` narrows one Search declaration:

* ``max_results`` is floored against what the request would otherwise get,
  which is the request's own value or the deployment's default.
* ``blocked_domains`` is added to the request's own block-list.
* ``allowed_domains`` is intersected with the request's by domain suffix,
  and a request whose list overlaps the workspace's nowhere is refused.
* ``purpose_hint`` fills in only when the request named none.

``provider_options`` is merged per key with the request winning.
It is an opaque mapping of backend options, so no narrowing relation holds between two values of it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from gateway.exceptions.tools_exceptions import (
    WebAccessNotEnabledError,
    WebAccessToolNotAuthorizedError,
    WebSearchNotEnabledError,
    WorkspaceWebSearchDomainsExcludedError,
)
from gateway.models.tools import WebTool
from gateway.services.tools._web_search_results import web_search_max_results_baseline
from gateway.services.web_retrieval_policy import (
    CanonicalHost,
    DisjointDomainAllowListsError,
    DomainPolicy,
    DomainRuleValidationError,
    canonicalize_domain_rule,
    canonicalize_domain_rules,
    domain_rule_matches,
    intersect_domain_allow_lists,
    union_domain_block_lists,
)

if TYPE_CHECKING:
    from gateway.core.config import GatewayConfig
    from gateway.models.tools import ResolvedWebSearchConfig, WebSearchCredential


@dataclass(frozen=True)
class WebAccessGrant:
    """The web access a request may use once its workspace's policy applies."""

    search_tool_entry: dict[str, Any] | None
    fetch_policy: DomainPolicy
    # The workspace's own search key, set only where it declared a search.
    search_credential: WebSearchCredential | None = None


def apply_web_access_policy(
    policy: ResolvedWebSearchConfig | None,
    *,
    requested_tools: Sequence[WebTool],
    search_tool_entry: dict[str, Any] | None,
    config: GatewayConfig,
) -> WebAccessGrant:
    """Narrow the declared web access to what ``policy`` permits.

    ``None`` is no policy, which narrows nothing.
    ``search_tool_entry`` is not mutated.

    Raises:
        WebAccessRefusedError: the policy refuses the request, and the subclass says why.
        WorkspaceWebSearchDomainsExcludedError: the request's domains share nothing with the workspace's.
    """
    fetch_requested = WebTool.FETCH in requested_tools
    workspace_domains = DomainPolicy()
    if policy is not None:
        if not policy.enabled:
            raise WebAccessNotEnabledError() if fetch_requested else WebSearchNotEnabledError()
        workspace_domains = DomainPolicy(
            allowed=canonicalize_domain_rules(policy.allowed_domains or ()),
            blocked=canonicalize_domain_rules(policy.blocked_domains or ()),
        )
        if (
            policy.authorized_tools is not None
            and not {tool.value for tool in requested_tools} <= policy.authorized_tools
        ):
            raise WebAccessToolNotAuthorizedError()
    try:
        fetch_policy = _fetch_policy(workspace_domains, search_tool_entry if fetch_requested else None)
    except DisjointDomainAllowListsError as exc:
        raise WorkspaceWebSearchDomainsExcludedError() from exc
    if policy is not None and search_tool_entry is not None:
        search_tool_entry = narrow_web_search_tool_entry(
            search_tool_entry,
            policy,
            baseline_max_results=web_search_max_results_baseline(config),
        )
    return WebAccessGrant(
        search_tool_entry=search_tool_entry,
        fetch_policy=fetch_policy,
        search_credential=policy.credential if policy is not None and search_tool_entry is not None else None,
    )


def _fetch_policy(workspace_domains: DomainPolicy, search_tool_entry: dict[str, Any] | None) -> DomainPolicy:
    """Let a Search declaration's domains narrow, but never replace, the workspace's Fetch domains."""
    request_allowed = None
    request_blocked = None
    if search_tool_entry is not None:
        if search_tool_entry.get("allowed_domains"):
            request_allowed = canonicalize_domain_rules(search_tool_entry["allowed_domains"])
        if search_tool_entry.get("blocked_domains"):
            request_blocked = canonicalize_domain_rules(search_tool_entry["blocked_domains"])
    allowed = intersect_domain_allow_lists(workspace_domains.allowed or None, request_allowed) or ()
    blocked = union_domain_block_lists(workspace_domains.blocked or None, request_blocked)
    return DomainPolicy(allowed=allowed, blocked=blocked)


def narrow_web_search_tool_entry(
    tool_entry: dict[str, Any],
    config: ResolvedWebSearchConfig,
    *,
    baseline_max_results: int,
) -> dict[str, Any]:
    """Compose a workspace's configuration onto the request's tool entry.

    Returns a new entry rather than mutating the caller's, which is the one
    reachable from the request body it was extracted from.

    Assumes ``config.enabled``; the veto is the caller's to raise, because only
    the caller knows which error shape the request format wants.

    ``baseline_max_results`` is how many results this request would get without
    a workspace row at all (``web_search_max_results_baseline``:
    the deployment's own setting, or the backend's built-in). The workspace
    ceiling is floored against it and not merely written in, because writing it
    in would let a workspace whose ceiling sits above the operator's *raise* the
    operator's number, which is the one thing the narrowing rule forbids.

    Raises :class:`WorkspaceWebSearchDomainsExcludedError` when the request
    names an allow-list that overlaps the workspace's nowhere (see
    :func:`_intersect` for what overlapping means when the entries are domain
    suffixes rather than hosts). The alternative is an empty effective allow-list,
    which the route's web retrieval backend reads as *no* allow-list because an empty list
    is falsy, and that turns the narrowest possible policy into no policy at
    all. Refusing also tells the caller something a silent zero-result search
    would not.
    """
    narrowed = dict(tool_entry)

    if config.max_results is not None:
        requested_max = narrowed.get("max_results")
        # ``bool`` is an ``int`` subclass, so exclude it: a JSON ``true`` must
        # not be read as a one-result ceiling.
        if not isinstance(requested_max, int) or isinstance(requested_max, bool) or requested_max <= 0:
            requested_max = baseline_max_results
        narrowed["max_results"] = min(requested_max, config.max_results)

    if config.blocked_domains:
        # Union: a workspace block a request could drop by sending a block-list
        # of its own would be a guardrail that fails open.
        narrowed["blocked_domains"] = _union(_entry_domains(narrowed.get("blocked_domains")), config.blocked_domains)

    if config.allowed_domains:
        requested_allowed = _entry_domains(narrowed.get("allowed_domains"))
        if requested_allowed is None:
            narrowed["allowed_domains"] = list(config.allowed_domains)
        else:
            both = _intersect(requested_allowed, config.allowed_domains)
            if not both:
                raise WorkspaceWebSearchDomainsExcludedError()
            narrowed["allowed_domains"] = both

    # A hint informs the model, it does not permit anything, so the request's
    # own wins and the workspace's fills a gap.
    if not narrowed.get("purpose_hint") and config.purpose_hint:
        narrowed["purpose_hint"] = config.purpose_hint

    if config.provider_options:
        request_options = narrowed.get("provider_options")
        narrowed["provider_options"] = (
            {**config.provider_options, **request_options}
            if isinstance(request_options, dict)
            else dict(config.provider_options)
        )

    return narrowed


def _entry_domains(value: Any) -> list[str] | None:
    """Read a domain list off a request's tool entry, normalized like a stored one.

    ``None`` for anything that is not a non-empty list, so a malformed or absent
    field reads as "the request named no list" rather than as an empty one.
    """
    if not isinstance(value, list):
        return None
    hosts = [str(host).strip().lower() for host in value if str(host).strip()]
    return hosts or None


def _union(requested: list[str] | None, workspace: tuple[str, ...]) -> list[str]:
    """Every domain either side named, in request-then-workspace order, de-duplicated."""
    merged: dict[str, None] = {}
    for host in (*(requested or ()), *workspace):
        merged.setdefault(host, None)
    return list(merged)


def _canonical_rules(values: list[str] | tuple[str, ...]) -> list[tuple[str, CanonicalHost]]:
    rules: list[tuple[str, CanonicalHost]] = []
    for value in dict.fromkeys(values):
        try:
            rules.append((value, canonicalize_domain_rule(value)))
        except DomainRuleValidationError:
            continue
    return rules


def _intersect(requested: list[str], workspace: tuple[str, ...]) -> list[str]:
    """The domains both sides permit, in the request's order.

    Not a set intersection, because an entry in either list is a *suffix* and not
    a host: ``WebRetrievalBackend._apply_domain_filters`` keeps a result when its hostname
    equals an entry or ends in ``"." + entry``. So ``example.com`` on the
    workspace's list already covers ``docs.example.com``, and a request naming
    the subdomain is asking for strictly less than the workspace permits rather
    than for something outside it. Whichever side is the narrower of an
    overlapping pair is the one that survives; genuinely disjoint lists still
    intersect to nothing, which is what the caller refuses.
    """
    requested_rules = _canonical_rules(requested)
    workspace_rules = _canonical_rules(workspace)
    kept: dict[str, None] = {}
    for host, candidate in requested_rules:
        for allowed, rule in workspace_rules:
            if domain_rule_matches(rule, candidate):
                kept.setdefault(host, None)
            elif domain_rule_matches(candidate, rule):
                kept.setdefault(allowed, None)
    return list(kept)
