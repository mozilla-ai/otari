"""Run a policy's router backend for one request.

The bridge between the API layer, which knows the request, and a backend, which
knows how to rank models. Everything request-shaped arrives as a
:class:`RoutingSignal` of plain values, so this module (like the compiler) never
touches FastAPI, and a caller can simulate any request by constructing one.

The candidate pool is filtered *before* the backend sees it: a router that picked
a model the caller's allow-list forbids would have its choice dropped by the
compiler and silently serve something else, which reads as the router
misbehaving. Filtering first means the backend only ever ranks models this
request could actually use.
"""

from __future__ import annotations

from dataclasses import dataclass

from any_llm.exceptions import AnyLLMError

from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.routing import PolicySpec
from gateway.services.model_access import is_model_allowed
from gateway.services.provider_kwargs import resolve_provider_selector
from gateway.services.routing.backends import (
    KNN_BACKEND,
    NOOP_BACKEND,
    RoutingContext,
    get_router_backend,
    owes_missing_backend_warning,
)
from gateway.services.routing.compiler import RouterOrdering

__all__ = ["RoutingSignal", "decide_ordering"]


@dataclass(frozen=True)
class RoutingSignal:
    """What a router needs to know about one request, in format-neutral form.

    Built by the endpoint from its own wire format (see
    ``api/routes/_helpers.py``), so a backend never learns which endpoint it is
    serving.
    """

    task_signal: str = ""
    """This turn's prompt text."""
    trace_signal: str = ""
    """The conversation's opening prompt text, stable across its turns."""
    trace_anchor: str = ""
    """Text identifying the conversation when the client sends no id."""
    conversation_id: str | None = None
    """The client's ``Otari-Conversation-Id``, the explicit trace identity."""
    task_id: str | None = None
    """The client's ``Otari-Router-Task``: which partition to vote over."""
    has_tools: bool = False
    is_continuation: bool = False
    """Whether the conversation already has an assistant turn, i.e. this is not
    the request that makes the trace-sticky decision."""
    opted_out: bool = False
    """The client sent ``Otari-Router: off`` for this request."""


async def decide_ordering(
    config: GatewayConfig,
    spec: PolicySpec,
    *,
    policy_name: str,
    user_id: str | None,
    allowlist: list[str] | None,
    signal: RoutingSignal | None,
) -> RouterOrdering | None:
    """Ask the policy's router to rank its candidates for this request.

    Returns ``None`` when there is nothing to ask: the policy names no router,
    this build has no such backend, or the surface has no request to route. The
    compiler treats all three as "no ordering" and serves the default target; only
    the unknown-backend case is logged, once per policy.

    Returns an *empty* :class:`RouterOrdering` when a router was asked and
    declined, or when the caller opted out. That also serves the default target,
    but deliberately without the warning: a decline is normal operation.
    """
    backend_name = spec.router_backend
    if backend_name is None:
        return None
    if signal is None:
        # A synchronous surface (the model catalog, `explain`) has no request.
        return None
    if signal.opted_out:
        return RouterOrdering([], rationale="caller sent Otari-Router: off")

    backend = get_router_backend(config, backend_name)
    if backend is None:
        # This build has no such backend, which is the one "no ordering" case that
        # is a misconfiguration rather than normal operation, and the only one worth
        # a log line. Warned here rather than in the compiler because the compiler
        # also runs where there is no request at all (`explain`, the CLI), and
        # warning there reported a problem that did not exist.
        if owes_missing_backend_warning(policy_name, backend_name):
            logger.warning(
                "Routing policy '%s' names router backend '%s', which this build does not have, so the "
                "policy serves '%s' on every request. Available backends: %s. Logged once per policy per "
                "process.",
                policy_name,
                backend_name,
                spec.default_target,
                ", ".join((KNN_BACKEND, NOOP_BACKEND)),
            )
        return None

    pool = _usable_candidates(config, spec.router_candidates, user_id=user_id, allowlist=allowlist)
    default_model = spec.default_target
    if not pool:
        return RouterOrdering([], rationale="no candidate in the pool is usable by this caller")

    decision = await backend.rank(
        RoutingContext(
            user_id=user_id or "",
            default_model=default_model,
            candidate_pool=pool,
            task_signal=signal.task_signal,
            trace_signal=signal.trace_signal,
            trace_anchor=signal.trace_anchor,
            task_id=signal.task_id,
            has_tools=signal.has_tools,
            is_trace_continuation=signal.is_continuation,
            trace_key=signal.conversation_id,
        )
    )
    # One line per routed request, at info: which model the money went to and why
    # is the first thing an operator asks when a bill or a quality complaint
    # arrives, and the decision is not otherwise reconstructable.
    logger.info(
        "Router '%s' on policy '%s': %s (confidence=%.2f) -> %s",
        backend_name,
        policy_name,
        decision.rationale,
        decision.confidence,
        decision.ordered_models[0] if decision.ordered_models else "policy default",
    )
    return RouterOrdering(
        selectors=list(decision.ordered_models),
        confidence=decision.confidence,
        rationale=decision.rationale,
    )


def _usable_candidates(
    config: GatewayConfig,
    candidates: list[str],
    *,
    user_id: str | None,
    allowlist: list[str] | None,
) -> list[str]:
    """The candidates this caller may actually be served, in declared order.

    Selectors are kept in the form the policy wrote them (the compiler resolves
    them again for dispatch); resolution here is only to test the allow-list,
    which matches on the canonical ``instance:model``.
    """
    usable: list[str] = []
    for selector in candidates:
        try:
            resolved = resolve_provider_selector(config, selector, user_id)
        except (ValueError, AnyLLMError):
            # The compiler records this as a dropped candidate with a reason; here
            # it just means the router should not rank it.
            continue
        if not is_model_allowed(allowlist, f"{resolved.instance}:{resolved.model}"):
            continue
        usable.append(selector)
    return usable
