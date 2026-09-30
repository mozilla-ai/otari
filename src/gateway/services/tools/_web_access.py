"""Apply a workspace's web search policy to the web access one request declared.

A policy may veto and may narrow, and it never grants.
The rule is the same whichever plane the policy came from.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from gateway.exceptions.tools_exceptions import (
    WebAccessDomainsExcludedError,
    WebAccessNotEnabledError,
    WebAccessToolNotAuthorizedError,
    WebSearchNotEnabledError,
)
from gateway.services.tenancy.workspace_web_search_service import narrow_web_search_tool_entry
from gateway.services.tools._web_search_results import web_search_max_results_baseline
from gateway.services.web_retrieval_backend import WEB_FETCH_TOOL_NAME
from gateway.services.web_retrieval_policy import (
    DisjointDomainAllowListsError,
    DomainPolicy,
    canonicalize_domain_rules,
    intersect_domain_allow_lists,
    union_domain_block_lists,
)

if TYPE_CHECKING:
    from gateway.core.config import GatewayConfig
    from gateway.models.tools import ResolvedWebSearchConfig


@dataclass(frozen=True)
class WebAccessGrant:
    """The web access a request may use once its workspace's policy applies."""

    search_tool_entry: dict[str, Any] | None
    fetch_policy: DomainPolicy


def apply_web_access_policy(
    policy: ResolvedWebSearchConfig | None,
    *,
    requested_tools: Sequence[str],
    search_tool_entry: dict[str, Any] | None,
    config: GatewayConfig,
) -> WebAccessGrant:
    """Narrow the declared web access to what ``policy`` permits.

    ``None`` is no policy, which narrows nothing.
    ``search_tool_entry`` is not mutated.

    Raises:
        WebAccessRefusedError: the policy refuses the request, and the subclass says why.
        WorkspaceWebSearchDomainsExcludedError: the Search domains share nothing with the workspace's.
    """
    fetch_requested = WEB_FETCH_TOOL_NAME in requested_tools
    workspace_domains = DomainPolicy()
    if policy is not None:
        if not policy.enabled:
            raise WebAccessNotEnabledError() if fetch_requested else WebSearchNotEnabledError()
        workspace_domains = DomainPolicy(
            allowed=canonicalize_domain_rules(policy.allowed_domains or ()),
            blocked=canonicalize_domain_rules(policy.blocked_domains or ()),
        )
        if policy.authorized_tools is not None and not set(requested_tools) <= policy.authorized_tools:
            raise WebAccessToolNotAuthorizedError()
    try:
        fetch_policy = _fetch_policy(workspace_domains, search_tool_entry if fetch_requested else None)
    except DisjointDomainAllowListsError as exc:
        raise WebAccessDomainsExcludedError() from exc
    if policy is not None and search_tool_entry is not None:
        search_tool_entry = narrow_web_search_tool_entry(
            search_tool_entry,
            policy,
            baseline_max_results=web_search_max_results_baseline(config),
        )
    return WebAccessGrant(search_tool_entry=search_tool_entry, fetch_policy=fetch_policy)


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
