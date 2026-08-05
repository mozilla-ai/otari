"""Compile a routing policy into an ordered plan of attempts.

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

from dataclasses import dataclass, field

from any_llm.exceptions import AnyLLMError

from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.guardrails import GuardrailConfig
from gateway.models.routing import MAX_CANDIDATES, PolicySpec, WhenClause
from gateway.services.model_access import is_model_allowed
from gateway.services.provider_kwargs import resolve_provider_selector
from gateway.types.attempt import Attempt
from gateway.types.budget_state import BudgetState

__all__ = [
    "BudgetState",
    "CompiledPlan",
    "DroppedCandidate",
    "NoEligibleCandidatesError",
    "RouterOrdering",
    "compile_policy",
    "needs_budget_state",
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

    attempts: list[Attempt] = []
    dropped: list[DroppedCandidate] = []
    seen: set[str] = set()

    for selector, selection_reason in ordered:
        try:
            resolved = resolve_provider_selector(config, selector)
        except (ValueError, AnyLLMError) as exc:
            # Startup validation rejects an unresolvable selector, so reaching
            # this means the provider set changed under a running gateway.
            dropped.append(
                DroppedCandidate(selector, "unresolvable", f"could not be resolved to a provider ({exc})")
            )
            continue

        canonical = f"{resolved.instance}:{resolved.model}"
        if canonical in seen:
            dropped.append(DroppedCandidate(selector, "duplicate", "already in the plan at an earlier position"))
            continue
        if not is_model_allowed(allowlist, canonical):
            dropped.append(
                DroppedCandidate(selector, "not_allowed", "is not in allowed_models for this caller")
            )
            continue
        if len(attempts) >= MAX_CANDIDATES:
            dropped.append(
                DroppedCandidate(selector, "over_cap", f"exceeds the {MAX_CANDIDATES}-candidate cap")
            )
            continue

        seen.add(canonical)
        attempts.append(
            Attempt(
                position=len(attempts) + 1,
                instance=resolved.instance,
                provider=resolved.provider,
                model=resolved.model,
                kwargs=resolved.kwargs,
                display_model=policy_name,
                selection_reason=selection_reason,
            )
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
