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

import asyncio
import logging
import uuid
from dataclasses import dataclass

from any_llm.exceptions import AnyLLMError

from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.routing import PolicySpec
from gateway.services.model_access import is_model_allowed
from gateway.services.provider_kwargs import resolve_provider_selector
from gateway.services.routing.backends import (
    RoutingContext,
    backend_is_weighted,
    get_router_backend,
    known_backends,
    owes_missing_backend_warning,
)
from gateway.services.routing.compiler import RouterOrdering
from gateway.services.routing.weighted import declared_shares, describe_split, explain_ordering

__all__ = [
    "ROUTER_DEADLINE_SECONDS",
    "RoutingSignal",
    "WeightedShare",
    "decide_ordering",
    "explain_router_ordering",
    "usable_candidates",
]

# A ranking that has not answered by now is not worth the caller's latency. The
# work behind it is an outbound embedding call plus a read of the stored examples,
# both on the request path and neither with a deadline of its own, so a hung
# embedding provider would otherwise hold the request open for as long as its own
# client allows. Expiring here costs the cheaper model, not the request.
ROUTER_DEADLINE_SECONDS = 5.0


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
    workspace_id: uuid.UUID | None = None,
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
                ", ".join(known_backends()),
            )
        return None

    pool = usable_candidates(
        config, spec.router_candidates, user_id=user_id, allowlist=allowlist, workspace_id=workspace_id
    )
    default_model = spec.default_target
    if not pool:
        return RouterOrdering([], rationale="no candidate in the pool is usable by this caller")

    context = RoutingContext(
        user_id=user_id or "",
        default_model=default_model,
        candidate_pool=pool,
        workspace_id=workspace_id,
        task_signal=signal.task_signal,
        trace_signal=signal.trace_signal,
        trace_anchor=signal.trace_anchor,
        task_id=signal.task_id,
        has_tools=signal.has_tools,
        is_trace_continuation=signal.is_continuation,
        trace_key=signal.conversation_id,
        weights=spec.router_weights,
    )
    try:
        async with asyncio.timeout(ROUTER_DEADLINE_SECONDS):
            decision = await backend.rank(context)
    except Exception as exc:
        # A router is an optimization, so it must never be the reason a request
        # cannot be served, or the reason it hangs. Backends already decline on the
        # failures they can name
        # (cold pool, embedding error, missing pricing), but ranking also reads the
        # database, and a broad guard here is what makes the claim true for every
        # backend and every failure rather than for the ones each backend thought
        # of. Declining costs the caller the cheaper model, not the request.
        logger.warning(
            "Router '%s' on policy '%s' failed or timed out (%s); serving '%s'",
            backend_name,
            policy_name,
            type(exc).__name__,
            spec.default_target,
        )
        return RouterOrdering([], rationale=f"router error ({type(exc).__name__})")
    # One line per routed request, at info: which model the money went to and why
    # is the first thing an operator asks when a bill or a quality complaint
    # arrives, and the decision is not otherwise reconstructable. A backend whose
    # decision *is* reconstructable (the weighted split is written in the policy,
    # and the usage row names what served) opts down to debug, because at load
    # balancer volume this line would be the log.
    logger.log(
        logging.INFO if decision.log_decision else logging.DEBUG,
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


@dataclass(frozen=True)
class WeightedShare:
    """One candidate's share of a weighted policy's traffic, for display.

    Carries both spellings because the two readers need different ones: the API
    reports ``selector`` (what the policy document says, so a form can bind it to
    the field the operator edits), and the CLI matches ``canonical`` against a
    compiled attempt, which identifies itself as ``instance:model``.
    """

    selector: str
    canonical: str
    share_pct: float


def explain_router_ordering(
    config: GatewayConfig,
    spec: PolicySpec,
    *,
    user_id: str | None = None,
    allowlist: list[str] | None = None,
    workspace_id: uuid.UUID | None = None,
) -> tuple[RouterOrdering | None, list[WeightedShare]]:
    """The ordering to show on a surface that has no request to route.

    Returns ``(None, [])`` for a policy whose router needs request state, which is
    every backend but one: kNN has no prompt to embed here, so the compiler falls
    through to the default target, and that decline path is what ``explain``
    documents.

    The weighted backend is the exception, because its split is written in the
    policy rather than derived from the request. So explain shows the real answer:
    candidates heaviest share first, with shares normalized over the pool that
    survived this caller's allow-list. Deliberately not a sample of the draw, which
    would make the same command print a different plan on every run.

    The *ordering* covers the whole declared pool, unlike the request path, which
    filters before ranking. Filtering here too would hand the compiler a pool with
    the forbidden candidates already gone, so they would never appear in
    :attr:`CompiledPlan.dropped` with their reason, and that list is most of the
    point of ``explain``: a balanced policy that compiles down to one provider for
    this caller is exactly what an operator needs to find before an outage does.

    That holds when *nothing* in the pool survives, which is the case the operator
    most needs to see. The shares are empty, because a split over no candidate is
    not a split, but the ordering still carries the declared pool so every filtered
    candidate is named in ``dropped``. Returning nothing here instead would compile
    the policy down to its ``on_failure`` chain with the whole split silently
    missing from both lists.
    """
    if not backend_is_weighted(spec.router_backend):
        return None, []
    declared = spec.router_candidates
    usable = usable_candidates(
        config, declared, user_id=user_id, allowlist=allowlist, workspace_id=workspace_id
    )
    weights = spec.router_weights
    shares = declared_shares(weights, usable)
    displayed: list[WeightedShare] = []
    for selector in explain_ordering(usable, weights):
        try:
            resolved = resolve_provider_selector(config, selector, user_id, workspace_id=workspace_id)
        except (ValueError, AnyLLMError):  # pragma: no cover - usable_candidates already resolved these
            continue
        displayed.append(
            WeightedShare(
                selector=selector,
                canonical=f"{resolved.instance}:{resolved.model}",
                share_pct=shares[selector],
            )
        )
    ordering = RouterOrdering(
        selectors=explain_ordering(declared, weights),
        confidence=(displayed[0].share_pct / 100.0) if displayed else 0.0,
        rationale=(
            f"weighted split ({describe_split(usable, weights)})"
            if usable
            else "no candidate in the split is usable by this caller"
        ),
    )
    return ordering, displayed


def usable_candidates(
    config: GatewayConfig,
    candidates: list[str],
    *,
    user_id: str | None,
    allowlist: list[str] | None,
    workspace_id: uuid.UUID | None = None,
) -> list[str]:
    """The candidates this caller may actually be served, in declared order.

    Selectors are kept in the form the policy wrote them (the compiler resolves
    them again for dispatch); resolution here is only to test the allow-list,
    which matches on the canonical ``instance:model``.

    ``workspace_id`` is passed for the same reason the compiler takes one: a
    candidate selector can itself name an alias, and resolving it in a different
    workspace than dispatch will would filter the pool against a canonical key
    the request never sees.

    Public because ``explain`` needs the same pool: a weighted policy's shares are
    normalized over what survived filtering, so computing them from the declared
    pool would report a split that is not the one running.
    """
    usable: list[str] = []
    for selector in candidates:
        try:
            resolved = resolve_provider_selector(config, selector, user_id, workspace_id=workspace_id)
        except (ValueError, AnyLLMError):
            # The compiler records this as a dropped candidate with a reason; here
            # it just means the router should not rank it.
            continue
        if not is_model_allowed(allowlist, f"{resolved.instance}:{resolved.model}"):
            continue
        usable.append(selector)
    return usable
