"""Compile a routing policy, or a catalog ID's offerings, into an ordered plan of attempts.

Pure and synchronous. Everything it needs about the request arrives as arguments
(:class:`BudgetState` and the allow-list), so it takes no database session, is
trivially testable, and adds no queries of its own. The caller decides whether
the budget numbers are even worth fetching, via :func:`needs_budget_state`: a
plain failover policy has no conditions, so the common case costs zero extra
queries.

What it deliberately does **not** do:

* **No pricing lookups.** Pricing is one query per model, so checking every
  candidate would be an N+1 on the request path. The existing pricing gate keys on
  the selected head candidate exactly as it does for a plain model name, and the
  ``explain`` surface is where an operator sees an unpriced fallback before it
  matters.
* **No dispatch.** It returns candidates; the walker tries them.

Drops are values, not silence. Every candidate removed from the plan is recorded
in :attr:`CompiledPlan.dropped` with a reason, so the caller can log it, return it
from ``explain``, and record it on the usage row. A "failover" policy that
silently compiled down to one attempt is the failure mode this exists to prevent.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field, replace

from any_llm.exceptions import AnyLLMError

from gateway.core.config import GatewayConfig
from gateway.core.error_codes import MODEL_NOT_ALLOWED, MODEL_NOT_FOUND, MODEL_NOT_SERVING
from gateway.log_config import logger
from gateway.models.guardrails import GuardrailConfig
from gateway.models.routing import MAX_CANDIDATES, PolicySpec, WhenClause
from gateway.services.model_access import is_model_allowed, org_model_refusal
from gateway.services.provider_kwargs import (
    credential_ladder_exhausted,
    resolve_catalog_fallback,
    resolve_provider_selector,
)
from gateway.types.attempt import Attempt
from gateway.types.budget_state import BudgetState

__all__ = [
    "BudgetState",
    "CompiledPlan",
    "DroppedCandidate",
    "NoEligibleCandidatesError",
    "RouterOrdering",
    "compile_catalog_plan",
    "compile_policy",
    "needs_budget_state",
    "selection_consults_router",
]


class NoEligibleCandidatesError(Exception):
    """Every candidate in a policy was filtered out before dispatch.

    Carries both audiences' text. ``caller_detail`` names the policy and nothing
    else, because a policy exists partly to keep its targets off the wire.
    ``operator_detail`` enumerates each candidate and why it went, for the
    activity log and for ``explain``, which are master-key surfaces.

    The status is derived from *why* the candidates went, because the two cases
    are not the same fault. Access rules dropping them is the caller being denied
    something that does exist: a 403 they can act on by asking for access. Nothing
    resolving is a gateway whose provider configuration no longer matches its
    policies, which is a 502: the caller did nothing wrong and there is nothing
    they can do. Sending 403 for both told an operator whose provider instance had
    been deleted to go audit their allow-lists.
    """

    def __init__(self, policy_name: str, dropped: list[DroppedCandidate]) -> None:
        self.policy_name = policy_name
        self.dropped = dropped
        unresolvable_only = bool(dropped) and all(item.reason == "unresolvable" for item in dropped)
        self.status_code = 502 if unresolvable_only else 403
        cause = (
            "None of its candidates resolve to a configured provider"
            if unresolvable_only
            else "Its candidates were all filtered out before dispatch"
        )
        self.caller_detail = (
            f"Routing policy '{policy_name}' has no usable candidate for this request, so it cannot be "
            f"served. {cause}. Ask your operator to check the policy."
        )
        reasons = "; ".join(f"'{item.selector}' {item.detail}" for item in dropped) or "no candidates declared"
        self.operator_detail = f"Routing policy '{policy_name}' compiled to 0 usable candidates. Dropped: {reasons}."
        super().__init__(self.operator_detail)


@dataclass(frozen=True)
class RouterOrdering:
    """What a router backend decided for one request.

    Passed *into* the compiler rather than fetched by it. Routing is asynchronous
    (an embedding call and a query over stored examples) and the compiler is pure
    and synchronous on purpose, so the I/O stays in the request pipeline and the
    ordering arrives as a value. That also means ``explain`` and the tests can
    simulate a router without one existing.

    ``selectors`` is the pool in the router's preferred order, best first. Empty
    means the router declined, which is a normal outcome (a cold pool, a
    low-confidence neighborhood) and compiles to the policy's default target.
    """

    selectors: list[str]
    confidence: float = 0.0
    rationale: str = ""


@dataclass(frozen=True)
class DroppedCandidate:
    """A candidate that did not make it into the plan."""

    selector: str
    reason: str
    """Machine-readable: ``unresolvable``, ``not_allowed``, ``duplicate``, ``over_cap``."""
    detail: str
    """Human-readable, for an operator."""


# A dropped candidate's detail reads after its selector ("'openai:x' is ..."),
# so the organization-key refusals are phrased as fragments here rather than
# reusing the full sentence the request routes answer with.
_ORG_REFUSAL_FRAGMENTS: dict[str, str] = {
    MODEL_NOT_ALLOWED: "is not in this workspace's organization-key model allow-list",
    MODEL_NOT_SERVING: "is offered by the organization's provider key but not serving",
    MODEL_NOT_FOUND: "is not offered by the organization's provider key",
}


@dataclass(frozen=True)
class CompiledPlan:
    """An ordered plan, plus everything that was left out and why."""

    policy_name: str
    attempts: list[Attempt]
    guardrails: list[GuardrailConfig] = field(default_factory=list)
    dropped: list[DroppedCandidate] = field(default_factory=list)
    router_ordering: RouterOrdering | None = None
    """The router decision this plan used, when a router entry supplied one.

    Kept on the plan so the rationale and confidence can be logged and shown in
    the activity log. A policy with a router that declined has ``None`` here and a
    ``default`` selection reason, which is how "the router chose the strong model"
    and "the router did not run" stay distinguishable after the fact.
    """

    @property
    def head(self) -> Attempt:
        """The candidate the request is priced and budgeted against."""
        return self.attempts[0]

    @property
    def selection_reason(self) -> str:
        """Why the head candidate was selected."""
        return self.attempts[0].selection_reason


def needs_budget_state(spec: PolicySpec) -> bool:
    """Whether any condition in ``spec`` reads budget numbers.

    Lets the caller skip the budget query entirely for a policy that only does
    failover, which is the common case.
    """
    return any(
        entry.when is not None
        and (entry.when.budget_used_pct is not None or entry.when.budget_remaining_usd is not None)
        for entry in spec.select
    )


def _matches(when: WhenClause, *, user_id: str | None, key_id: str | None, budget: BudgetState) -> bool:
    """Whether every condition present in ``when`` holds. Undefined never matches."""
    if when.budget_used_pct is not None:
        if budget.used_pct is None or not when.budget_used_pct.matches(budget.used_pct):
            return False
    if when.budget_remaining_usd is not None:
        if budget.remaining_usd is None or not when.budget_remaining_usd.matches(budget.remaining_usd):
            return False
    if when.user_id is not None:
        allowed = [when.user_id] if isinstance(when.user_id, str) else when.user_id
        if user_id is None or user_id not in allowed:
            return False
    if when.key_id is not None:
        allowed = [when.key_id] if isinstance(when.key_id, str) else when.key_id
        if key_id is None or key_id not in allowed:
            return False
    return True


def selection_consults_router(
    spec: PolicySpec,
    *,
    user_id: str | None = None,
    key_id: str | None = None,
    budget: BudgetState | None = None,
) -> bool:
    """Whether this request would actually reach the policy's ``router`` entry.

    Entries are evaluated in order, so a ``when`` entry ahead of the router wins
    outright and the router's ranking is discarded. Asking first keeps the caller
    from paying for a ranking (an embedding call and a scan of the user's stored
    examples) whose result nothing reads, and keeps the router's decision log line
    off requests it did not decide.

    Pure and synchronous like the rest of this module: it reads the same facts
    ``_select_head`` does, so the two cannot disagree about which entry wins.
    """
    budget = budget or BudgetState()
    for entry in spec.select:
        if entry.default is not None:
            return False
        if entry.router is not None:
            return True
        if entry.when is not None and _matches(entry.when, user_id=user_id, key_id=key_id, budget=budget):
            return False
    return False


def _select_head(
    spec: PolicySpec,
    *,
    policy_name: str,
    user_id: str | None,
    key_id: str | None,
    budget: BudgetState,
    router_ordering: RouterOrdering | None,
) -> list[tuple[str, str]]:
    """The selected candidates, in order, each with the reason it is there.

    Normally one head candidate: entries are evaluated in order and the first whose
    ``when`` matches wins, with the ``default`` entry (last, enforced by the schema)
    as the fallthrough.

    A ``router`` entry is the exception, and returns several. The router ranked the
    whole pool, and the walker can try candidates in order, so the ranking *is* the
    plan: its pick leads, the rest follow ahead of ``on_failure``. Nothing is
    discarded, so a routed request that fails over lands on the router's second
    choice rather than jumping straight to the operator's failure chain.
    """
    for entry in spec.select:
        if entry.default is not None:
            return [(entry.default, "default")]
        if entry.router is not None:
            if router_ordering is not None and router_ordering.selectors:
                reason = f"router:{entry.router}"
                return [(selector, reason) for selector in router_ordering.selectors]
            # No ordering: the router declined, this build has no such backend, or
            # this surface (``explain``, the model catalog) has no request to route.
            # Falling through to the default is the safe reading in all three: a
            # router is an optimization, and must never be the reason a request
            # cannot be served.
            #
            # Nothing is logged here on purpose. Only the caller knows which of the
            # three happened, and warning about all of them made ``explain`` (which
            # has no request by design) report a misconfiguration. The
            # unknown-backend warning lives in ``services/routing/decide``.
            continue
        if entry.when is not None and _matches(entry.when, user_id=user_id, key_id=key_id, budget=budget):
            assert entry.target is not None  # schema: a `when` entry always carries a target
            return [(entry.target, f"condition:{','.join(entry.when.conditions())}")]
    return [(spec.default_target, "default")]


def compile_policy(
    config: GatewayConfig,
    policy_name: str,
    spec: PolicySpec,
    *,
    user_id: str | None = None,
    key_id: str | None = None,
    allowlist: list[str] | None = None,
    budget: BudgetState | None = None,
    router_ordering: RouterOrdering | None = None,
    workspace_id: uuid.UUID | None = None,
) -> CompiledPlan:
    """Turn ``spec`` into an ordered plan for one request.

    Order: the selected candidate (or the router's whole ranking), then
    ``on_failure`` in declared order. Each selector is resolved locally (so the
    attempt carries this gateway's own credentials), then filtered by the caller's
    allow-list, deduplicated, and capped.

    ``router_ordering`` is the decision a router backend already made for this
    request; see :class:`RouterOrdering` for why it arrives as an argument. Omit
    it and a policy with a router compiles to its default target, which is what
    every synchronous surface (``explain``, the CLI) shows.

    ``workspace_id`` reaches ``resolve_provider_selector`` unchanged: it is a
    zero-I/O cache read (see that function), not a database call, so passing it
    here does not compromise this function's own "no DB and no I/O" contract.
    ``explain`` and the CLI omit it, which only affects a bare candidate
    selector with no matching ``config.providers`` instance.

    Raises :class:`NoEligibleCandidatesError` when nothing survives. That error
    derives its own status from why the candidates went: 403 when access rules
    filtered them, 502 when none of them resolve to a configured provider.
    """
    budget = budget or BudgetState()
    selected = _select_head(
        spec,
        policy_name=policy_name,
        user_id=user_id,
        key_id=key_id,
        budget=budget,
        router_ordering=router_ordering,
    )
    routed = selected[0][1].startswith("router:")

    ordered: list[tuple[str, str]] = list(selected)
    ordered.extend((selector, "on_failure") for selector in spec.on_failure)
    attempts, dropped = _resolve_candidates(
        config, ordered, display_model=policy_name, allowlist=allowlist, workspace_id=workspace_id
    )

    if not attempts:
        raise NoEligibleCandidatesError(policy_name, dropped)

    if dropped:
        logger.warning(
            "Routing policy '%s' compiled to %d of %d candidates; dropped %s",
            policy_name,
            len(attempts),
            len(ordered),
            "; ".join(f"{item.selector} ({item.reason})" for item in dropped),
        )

    return CompiledPlan(
        policy_name=policy_name,
        attempts=attempts,
        router_ordering=router_ordering if routed else None,
        guardrails=[
            GuardrailConfig(
                profile=guardrail.profile,
                url=guardrail.url,
                mode=guardrail.mode,
                on_unavailable=guardrail.on_unavailable,
                validate_kwargs=guardrail.validate_kwargs,
            )
            for guardrail in spec.guardrails
        ],
        dropped=dropped,
    )


def compile_catalog_plan(
    config: GatewayConfig,
    model_selector: str,
    *,
    user_id: str | None = None,
    allowlist: list[str] | None = None,
    workspace_id: uuid.UUID | None = None,
) -> CompiledPlan | None:
    """Turn a catalog ID into a failover plan over its offerings, best first, or ``None``.

    Each offering is filtered as a policy candidate is.
    ``None`` leaves the request to resolve as a plain model does: for a name an alias or a static policy claims,
    for a name with fewer than two offerings, when the filters leave no offering,
    and when an offering has no stored credential.
    """
    offerings = resolve_catalog_fallback(config, model_selector, user_id, workspace_id=workspace_id)
    if len(offerings) < 2:
        return None
    ordered = [(selector, "catalog") for selector in offerings]
    attempts, dropped = _resolve_candidates(
        config, ordered, display_model=model_selector, allowlist=allowlist, workspace_id=workspace_id
    )
    if dropped and logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Catalog ID '%s' compiled to %d of %d offerings; dropped %s",
            model_selector,
            len(attempts),
            len(offerings),
            "; ".join(f"{item.selector} ({item.reason})" for item in dropped),
        )
    if not attempts:
        return None
    # NOTE: Only the plain path asks the hosted-credential port, so an offering that needs it leaves the plan.
    if any(credential_ladder_exhausted(attempt.provider, attempt.kwargs) for attempt in attempts):
        return None
    # The first offering that survives the filters is the catalog's choice, and the rest follow it.
    attempts = [attempts[0], *(replace(attempt, selection_reason="on_failure") for attempt in attempts[1:])]
    return CompiledPlan(policy_name=model_selector, attempts=attempts, dropped=dropped)


def _resolve_candidates(
    config: GatewayConfig,
    ordered: list[tuple[str, str]],
    *,
    display_model: str,
    allowlist: list[str] | None,
    workspace_id: uuid.UUID | None,
) -> tuple[list[Attempt], list[DroppedCandidate]]:
    """The attempts ``ordered`` resolves to for this caller, and every candidate dropped on the way."""
    attempts: list[Attempt] = []
    dropped: list[DroppedCandidate] = []
    seen: set[str] = set()

    for selector, selection_reason in ordered:
        if len(attempts) >= MAX_CANDIDATES:
            dropped.append(DroppedCandidate(selector, "over_cap", f"exceeds the {MAX_CANDIDATES}-candidate cap"))
            continue
        try:
            resolved = resolve_provider_selector(config, selector, workspace_id=workspace_id)
        except (ValueError, AnyLLMError) as exc:
            # Startup validation rejects an unresolvable selector, so reaching
            # this means the provider set changed under a running gateway.
            dropped.append(DroppedCandidate(selector, "unresolvable", f"could not be resolved to a provider ({exc})"))
            continue

        canonical = f"{resolved.instance}:{resolved.model}"
        if canonical in seen:
            dropped.append(DroppedCandidate(selector, "duplicate", "already in the plan at an earlier position"))
            continue
        if not is_model_allowed(allowlist, canonical):
            dropped.append(DroppedCandidate(selector, "not_allowed", "is not in allowed_models for this caller"))
            continue
        # Organization-scoped model restriction (otari#643), same disjointness
        # condition `provider_kwargs.get_provider_kwargs` uses: only a selector
        # that named no configured instance can have resolved through an
        # organization key, so only that case is subject to its restriction.
        # Every candidate is checked here, not only the plan's head, so a
        # `router`/`on_failure` fallover cannot serve a model the workspace's
        # organization key excludes just because the head candidate passed.
        if workspace_id is not None and resolved.instance not in config.providers:
            refusal = org_model_refusal(workspace_id, resolved.provider.value, resolved.model, selector=selector)
            if refusal is not None:
                dropped.append(DroppedCandidate(selector, "not_allowed", _ORG_REFUSAL_FRAGMENTS[refusal.code]))
                continue
        seen.add(canonical)
        attempts.append(
            Attempt(
                position=len(attempts) + 1,
                instance=resolved.instance,
                provider=resolved.provider,
                model=resolved.model,
                kwargs=resolved.kwargs,
                display_model=display_model,
                selection_reason=selection_reason,
            )
        )

    return attempts, dropped
