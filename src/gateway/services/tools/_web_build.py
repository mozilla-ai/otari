"""One request's web tool backends on the libraries (design H).

The search backend, and the fetch backend when the request declared the fetch
tool, share the request's call allowance and open as two members of the
request's pool, so each rendering reads the backend that ran its call. A
request whose search runs on the old backend, or that declared only the fetch
tool, keeps the old backend's own builder, which takes its fetch backend from
:func:`build_web_fetch_backend`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gateway.core.env import otari_env
from gateway.services.tools._web_fetch import WebFetchBackend
from gateway.services.tools._web_search import WebSearchBackend, builtin_fetch_resolved, resolve_fetch
from gateway.services.tools._web_search_results import web_search_max_results_baseline
from gateway.services.web_retrieval_policy import DomainPolicy, canonicalize_domain_rules

if TYPE_CHECKING:
    from collections.abc import Mapping

    import httpx

    from gateway.core.config import GatewayConfig
    from gateway.core.settings.tools import ToolSettings
    from gateway.services.tool_usage import ToolUsageTally
    from gateway.services.tools._web_call_cap import WebToolCallCap
    from gateway.services.tools._web_search import InLoopSearch
    from gateway.services.traces import RequestTrace


def build_web_search_backends(
    config: GatewayConfig,
    search: InLoopSearch,
    *,
    search_tool_entry: Mapping[str, Any],
    fetch_declared: bool,
    fetch_policy: DomainPolicy,
    cap: WebToolCallCap,
    purpose_hint: str | None,
    web_provider_client: httpx.AsyncClient | None,
    tally: ToolUsageTally | None,
    trace: RequestTrace | None,
) -> list[WebSearchBackend | WebFetchBackend]:
    """The search backend on ``search``, and the fetch backend when the fetch tool was declared.

    ``search_tool_entry`` is the search declaration once the workspace's policy
    narrowed it: its count and domain rules.
    """
    trust_env_proxy = config.web_retrieval_trust_env_proxy
    backends: list[WebSearchBackend | WebFetchBackend] = [
        WebSearchBackend(
            search,
            cap=cap,
            max_results=_max_results(config, search_tool_entry, search.search.max_results),
            policy=DomainPolicy(
                allowed=canonicalize_domain_rules(_domains(search_tool_entry, "allowed_domains")),
                blocked=canonicalize_domain_rules(_domains(search_tool_entry, "blocked_domains")),
            ),
            enrich=_enrich(config),
            allow_private=config.web_search_allow_private_hosts,
            trust_env_proxy=trust_env_proxy,
            purpose_hint=purpose_hint,
            web_provider_client=web_provider_client,
            tally=tally,
            trace=trace,
        )
    ]
    if fetch_declared:
        backends.append(
            build_web_fetch_backend(
                config,
                policy=fetch_policy,
                trust_env_proxy=trust_env_proxy,
                web_provider_client=web_provider_client,
                tally=tally,
                trace=trace,
                cap=cap,
            )
        )
    return backends


def build_web_fetch_backend(
    config: ToolSettings | None,
    *,
    policy: DomainPolicy | None,
    trust_env_proxy: bool,
    web_provider_client: httpx.AsyncClient | None = None,
    tally: ToolUsageTally | None = None,
    trace: RequestTrace | None = None,
    cap: WebToolCallCap | None = None,
) -> WebFetchBackend:
    """One request's fetch tool, on the fetch instance the resolver chooses, or on ``builtin_fetch`` without a config.

    No workspace policy is passed to the resolver yet, so no fetch credential serves the call.
    ``cap`` is the request's shared search and fetch allowance, ``None`` where the old
    backend claims its own counter before delegating.
    """
    resolved = (
        resolve_fetch(config, None, requested=None, for_search=None) if config is not None else builtin_fetch_resolved()
    )
    return WebFetchBackend(
        resolved,
        policy=policy,
        trust_env_proxy=trust_env_proxy,
        web_provider_client=web_provider_client,
        tally=tally,
        trace=trace,
        cap=cap,
    )


def _max_results(config: GatewayConfig, entry: Mapping[str, Any], resolved: int | None) -> int:
    """The declaration's count, else the deployment's, lowered by a count the resolved options name (design G)."""
    requested = entry.get("max_results")
    if not isinstance(requested, int) or isinstance(requested, bool) or requested <= 0:
        requested = web_search_max_results_baseline(config)
    return min(requested, resolved) if resolved is not None else requested


def _domains(entry: Mapping[str, Any], field: str) -> tuple[str, ...]:
    values = entry.get(field)
    return tuple(str(value) for value in values) if isinstance(values, list) else ()


def _enrich(config: GatewayConfig) -> bool:
    """Whether hits without page text are enriched: the setting, else the environment, else yes."""
    if config.web_search_extract is not None:
        return config.web_search_extract
    extract_env = otari_env("WEB_SEARCH_EXTRACT")
    return extract_env is None or extract_env.lower() not in {"0", "false", "no", "off"}
