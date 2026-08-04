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
    "compile_policy",
    "needs_budget_state",
]


class NoEligibleCandidatesError(Exception):
    """Every candidate in a policy was filtered out before dispatch.

    Carries both audiences' text. ``caller_detail`` names the policy and nothing
    else, because a policy exists partly to keep its targets off the wire.
    ``operator_detail`` enumerates each candidate and why it went, for the
    activity log and for ``explain``, which are master-key surfaces.
    """

    def __init__(self, policy_name: str, dropped: list[DroppedCandidate], status_code: int) -> None:
        self.policy_name = policy_name
        self.dropped = dropped
        self.status_code = status_code
        self.caller_detail = (
            f"Routing policy '{policy_name}' has no usable candidate for this request, so it cannot be "
            "served. Every candidate was filtered out before dispatch by model-access rules. Ask your "
            "operator to check the policy."
        )
        reasons = "; ".join(f"'{item.selector}' {item.detail}" for item in dropped) or "no candidates declared"
        self.operator_detail = (
            f"Routing policy '{policy_name}' compiled to 0 usable candidates. Dropped: {reasons}."
        )
        super().__init__(self.operator_detail)


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
    user_id: str | None,
    key_id: str | None,
    budget: BudgetState,
) -> tuple[str, str]:
    """The head selector and the reason it was chosen.

    Entries are evaluated in order; the first whose ``when`` matches wins. The
    ``default`` entry is last (enforced by the schema), so it is the fallthrough.
    """
    for entry in spec.select:
        if entry.default is not None:
            return entry.default, "default"
        if entry.router is not None:
            # Router backends are not wired yet. Falling through to the default
            # is the safe reading: a router is an optimization, and it must never
            # be the reason a request cannot be served.
            logger.warning(
                "Routing policy names router '%s', which is not available yet; using the default target",
                entry.router,
            )
            continue
        if entry.when is not None and _matches(entry.when, user_id=user_id, key_id=key_id, budget=budget):
            assert entry.target is not None  # schema: a `when` entry always carries a target
            return entry.target, f"condition:{','.join(entry.when.conditions())}"
    return spec.default_target, "default"


def compile_policy(
    config: GatewayConfig,
    policy_name: str,
    spec: PolicySpec,
    *,
    user_id: str | None = None,
    key_id: str | None = None,
    allowlist: list[str] | None = None,
    budget: BudgetState | None = None,
) -> CompiledPlan:
    """Turn ``spec`` into an ordered plan for one request.

    Order: the selected head candidate, then ``on_failure`` in declared order.
    Each selector is resolved locally (so the attempt carries this gateway's own
    credentials), then filtered by the caller's allow-list, deduplicated, and
    capped.

    Raises :class:`NoEligibleCandidatesError` when nothing survives, with a 403
    (the only filter that can empty a validated plan is model access).
    """
    budget = budget or BudgetState()
    head, reason = _select_head(spec, user_id=user_id, key_id=key_id, budget=budget)

    ordered: list[tuple[str, str]] = [(head, reason)]
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
        raise NoEligibleCandidatesError(policy_name, dropped, status_code=403)

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
