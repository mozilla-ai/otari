"""Schema for the ``routing:`` config block: named routing policies.

A policy is a named model callers put in the ``model`` field. It decides which
real model serves the request, in what order candidates are tried, and which
guardrails always run. A one-target policy is the same thing as an alias, which
is why ``aliases:`` remains supported as its shorthand.

Two axes, deliberately separate:

* ``select`` decides where the plan *starts*. Entries are evaluated in order and
  the first whose ``when`` matches wins; the ``default`` entry is the
  fallthrough. A ``router`` entry hands ordering to a router backend instead.
* ``on_failure`` is what gets tried *after* a provider failure, in order.

Collapsing them into one list would make "did this entry not apply, or did it
fail?" ambiguous, and that ambiguity would surface in every log line and every
support thread. The names say which axis is which, and ``default`` is explicit
rather than positional so a misordered policy is refused instead of silently
having dead rules.

Every model here sets ``extra="forbid"``. ``GatewayConfig`` itself is
``extra="ignore"`` (a pydantic-settings default that also swallows stray env
vars), so a typo'd key inside a policy would otherwise vanish and the policy
would quietly not do what it says.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "MAX_CANDIDATES",
    "PolicyGuardrail",
    "PolicySpec",
    "RoutingConfig",
    "SelectEntry",
    "Threshold",
    "WhenClause",
]

# Ceiling on the compiled candidate list. Named rather than left implicit, and
# enforced by refusing the policy rather than by truncating it: a silently
# shortened chain is a failover policy that does less than it says.
MAX_CANDIDATES = 5

_COMPARATORS = ("gte", "gt", "lte", "lt")


class Threshold(BaseModel):
    """A single numeric comparison, e.g. ``{gte: 80}``."""

    model_config = ConfigDict(extra="forbid")

    gte: float | None = None
    gt: float | None = None
    lte: float | None = None
    lt: float | None = None

    @model_validator(mode="after")
    def _exactly_one_comparator(self) -> Threshold:
        set_comparators = [name for name in _COMPARATORS if getattr(self, name) is not None]
        if len(set_comparators) != 1:
            raise ValueError(
                f"a threshold needs exactly one of {', '.join(_COMPARATORS)}; got "
                f"{len(set_comparators)} ({', '.join(set_comparators) or 'none'})"
            )
        return self

    @property
    def comparator(self) -> str:
        for name in _COMPARATORS:
            if getattr(self, name) is not None:
                return name
        raise AssertionError("validated threshold always has one comparator")

    @property
    def value(self) -> float:
        return float(getattr(self, self.comparator))

    def matches(self, observed: float) -> bool:
        """Whether ``observed`` satisfies this comparison."""
        threshold = self.value
        if self.gte is not None:
            return observed >= threshold
        if self.gt is not None:
            return observed > threshold
        if self.lte is not None:
            return observed <= threshold
        return observed < threshold

    def describe(self) -> str:
        symbols = {"gte": ">=", "gt": ">", "lte": "<=", "lt": "<"}
        return f"{symbols[self.comparator]} {self.value:g}"


class WhenClause(BaseModel):
    """Conditions gating one ``select`` entry. All present conditions must match.

    A closed set on purpose: it is greppable, validatable at write time, and
    cannot grow into an expression language evaluated on the request path.
    """

    model_config = ConfigDict(extra="forbid")

    budget_used_pct: Threshold | None = Field(
        default=None,
        description=(
            "Percentage of the caller's budget already committed (spend + reserved). "
            "Undefined for a caller with no budget, an unlimited budget, or the master key; "
            "an undefined value never matches, so the default entry is used."
        ),
    )
    budget_remaining_usd: Threshold | None = Field(
        default=None,
        description="USD left in the caller's budget. Undefined in the same cases as budget_used_pct.",
    )
    user_id: str | list[str] | None = Field(default=None, description="Match one user id, or any in a list.")
    key_id: str | list[str] | None = Field(default=None, description="Match one API key id, or any in a list.")

    @model_validator(mode="after")
    def _at_least_one_condition(self) -> WhenClause:
        if not self.conditions():
            raise ValueError(
                "a `when` clause needs at least one condition "
                "(budget_used_pct, budget_remaining_usd, user_id, key_id); "
                "omit `when` entirely for an unconditional entry, or use `default`"
            )
        return self

    @model_validator(mode="after")
    def _budget_thresholds_stay_under_the_cap(self) -> WhenClause:
        """Refuse a rule that can only fire once the budget gate has already said no.

        The budget is enforced before selection: ``reserve_budget`` rejects a
        request whose estimate would push ``spend + reserved`` past ``max_budget``.
        So a rule written at 100% or above can never take effect, and an operator
        writing one believes they have configured "keep serving on a cheaper model
        after the budget runs out" when they have configured nothing. Tiering down
        keeps a caller *under* a cap; it is not a way past one.

        Only the upward comparators are unreachable. ``{lt: 100}`` and
        ``{lte: 100}`` mean "any caller still under the cap", which every request
        that gets as far as selection satisfies, so refusing those would reject a
        rule that works.
        """
        used = self.budget_used_pct
        if used is not None and used.comparator in ("gte", "gt") and used.value >= 100:
            raise ValueError(
                f"budget_used_pct {used.describe()} can never match: the budget gate rejects a request "
                "before selection once the cap is reached, so this rule would never take effect. "
                "Tiering down keeps a caller under a budget; it cannot serve traffic past one. "
                "Use a threshold below 100 (e.g. {gte: 80})."
            )
        return self

    def conditions(self) -> list[str]:
        """Names of the conditions actually set, for the selection reason."""
        names = ("budget_used_pct", "budget_remaining_usd", "user_id", "key_id")
        return [name for name in names if getattr(self, name) is not None]


class SelectEntry(BaseModel):
    """One entry in ``select``: a conditional target, the default, or a router."""

    model_config = ConfigDict(extra="forbid")

    when: WhenClause | None = None
    target: str | None = Field(default=None, description="Selector to use when `when` matches.")
    default: str | None = Field(default=None, description="Fallthrough selector. Exactly one entry must set this.")
    router: str | None = Field(default=None, description="Router backend that supplies the candidate ordering.")
    candidates: list[str] | None = Field(
        default=None,
        description=(
            "Selectors the router may order, for a `router` entry. Required there and meaningless "
            "elsewhere. The policy's `default` target is appended as the last resort if it is not "
            "already listed, so a router that declines can never leave the plan empty."
        ),
    )

    @model_validator(mode="after")
    def _exactly_one_destination(self) -> SelectEntry:
        chosen = [name for name in ("target", "default", "router") if getattr(self, name) is not None]
        if len(chosen) != 1:
            raise ValueError(
                f"a select entry needs exactly one of target, default, router; got "
                f"{len(chosen)} ({', '.join(chosen) or 'none'})"
            )
        return self

    @model_validator(mode="after")
    def _conditions_belong_to_targets(self) -> SelectEntry:
        if self.when is not None and self.default is not None:
            raise ValueError("the `default` entry is the fallthrough and cannot carry a `when` clause")
        if self.when is not None and self.router is not None:
            raise ValueError(
                "a `router` entry cannot carry a `when` clause: the router supplies the whole ordering, "
                "so combining it with a condition has no defined meaning"
            )
        return self

    @model_validator(mode="after")
    def _candidates_belong_to_routers(self) -> SelectEntry:
        """``candidates`` is the router's pool, so it only means something there.

        Required rather than optional, and at least two entries: a router asked to
        order one model has no decision to make, and an operator who wrote that
        believes they configured routing. Refusing says so at write time instead of
        serving the default forever and looking like a broken router.
        """
        if self.router is None:
            if self.candidates is not None:
                raise ValueError(
                    "`candidates` only applies to a `router` entry: it is the pool the router orders. "
                    "A static entry names its one model in `target` or `default`."
                )
            return self
        if not self.candidates:
            raise ValueError(
                f"a `router` entry needs `candidates`: the pool router '{self.router}' orders. "
                "Without it there is nothing to route among."
            )
        if len(self.candidates) < 2:
            raise ValueError(
                "a `router` entry needs at least 2 `candidates`: ordering a single model is not a "
                "routing decision. Name the models the router may choose between, cheapest included."
            )
        seen: set[str] = set()
        for selector in self.candidates:
            if selector in seen:
                raise ValueError(f"'{selector}' is listed twice in `candidates`; each candidate appears once")
            seen.add(selector)
        return self

    @property
    def selector(self) -> str | None:
        """The static selector this entry resolves to, if it is not a router."""
        return self.target if self.target is not None else self.default


class PolicyGuardrail(BaseModel):
    """A guardrail the operator mandates for every request through the policy.

    There is deliberately no ``on`` field. The request-level model accepts
    ``on: [output]`` and does not enforce it, so allowing it here would let an
    operator write a mandate that silently does nothing. Policy guardrails are
    input-direction only until output-direction enforcement exists.
    """

    model_config = ConfigDict(extra="forbid")

    profile: str = Field(min_length=1, max_length=128)
    mode: Literal["block", "monitor"] = Field(
        description=(
            "Required, with no default: the request-level field defaults to 'monitor', so an omitted "
            "mode here would read as a mandate and behave as shadow mode."
        )
    )
    on_unavailable: Literal["block", "monitor"] = Field(
        default="block",
        description=(
            "What to do when the guardrails service cannot be reached. 'block' (the default) fails "
            "closed, which means a guardrails outage rejects every request through this policy, ahead "
            "of the fallback chain. 'monitor' serves the request and records that the check was skipped."
        ),
    )
    url: str | None = Field(
        default=None,
        min_length=1,
        description="Override the operator-set guardrails service URL. SSRF-checked like the request-level field.",
    )
    validate_kwargs: dict[str, Any] = Field(default_factory=dict)


class PolicySpec(BaseModel):
    """One named routing policy."""

    model_config = ConfigDict(extra="forbid")

    spec_version: Literal[1] = 1
    select: list[SelectEntry] = Field(min_length=1)
    on_failure: list[str] = Field(
        default_factory=list,
        description="Selectors to try, in order, after a provider failure on the selected candidate.",
    )
    guardrails: list[PolicyGuardrail] = Field(default_factory=list)
    # No `limits` yet, deliberately. The only per-request deadline that exists is
    # the streaming first-chunk timeout, and it is applied solely by the hybrid
    # walker; standalone streaming (where policies apply) has none, so a
    # per-policy override would validate, store, and do nothing. It belongs here
    # once standalone streaming grows a deadline of its own.

    @model_validator(mode="after")
    def _one_default_and_it_comes_last(self) -> PolicySpec:
        default_positions = [index for index, entry in enumerate(self.select) if entry.default is not None]
        if len(default_positions) != 1:
            raise ValueError(
                f"select needs exactly one `default` entry (the fallthrough); found {len(default_positions)}"
            )
        if default_positions[0] != len(self.select) - 1:
            raise ValueError(
                "the `default` entry must come last in select: entries are evaluated in order, so any "
                "entry after the fallthrough could never be reached"
            )
        return self

    @model_validator(mode="after")
    def _candidate_count_within_cap(self) -> PolicySpec:
        # A router entry contributes its whole ordered pool at request time (the
        # walker cascades through it), so the cap counts the pool rather than one
        # head candidate. Without a router it is the selected candidate plus the
        # failure chain, as before.
        pool = self.router_candidates
        selected = len(pool) if pool else 1
        total = selected + len(self.on_failure)
        if total > MAX_CANDIDATES:
            detail = f"{selected} routed candidate(s) + on_failure" if pool else "1 selected + on_failure"
            raise ValueError(
                f"a policy may have at most {MAX_CANDIDATES} candidates ({detail}); this one has {total}"
            )
        return self

    @property
    def router_backend(self) -> str | None:
        """The router backend named by ``select``, if any."""
        for entry in self.select:
            if entry.router is not None:
                return entry.router
        return None

    @property
    def router_candidates(self) -> list[str]:
        """The router's pool, with the default target guaranteed present as the tail.

        Empty for a policy with no router. The default target is appended rather
        than assumed: it is what serves when the router declines, so it has to be
        in the plan even if the operator left it out of ``candidates``.
        """
        for entry in self.select:
            if entry.router is not None and entry.candidates:
                pool = list(entry.candidates)
                if self.default_target not in pool:
                    pool.append(self.default_target)
                return pool
        return []

    @property
    def is_dynamic(self) -> bool:
        """Whether the selected candidate depends on request state.

        A dynamic policy has no single target, so it cannot be resolved on the
        surfaces that need one synchronously (pricing, the model catalog, the
        non-completion endpoints).
        """
        return self.router_backend is not None or any(entry.when is not None for entry in self.select)

    @property
    def default_target(self) -> str:
        """The fallthrough selector. Always present once validated."""
        for entry in self.select:
            if entry.default is not None:
                return entry.default
        raise AssertionError("validated spec always has exactly one default entry")

    def static_selectors(self) -> list[str]:
        """Every statically declared selector, in plan order, deduplicated.

        A router's ``candidates`` count as static: they are written in the policy,
        so they get the same startup and write-time checks (resolvable, not another
        policy or alias) as any other target. What the router decides at request
        time is only their *order*.
        """
        ordered: list[str] = []
        for entry in self.select:
            selector = entry.selector
            if selector is not None and selector not in ordered:
                ordered.append(selector)
            for candidate in entry.candidates or []:
                if candidate not in ordered:
                    ordered.append(candidate)
        for selector in self.on_failure:
            if selector not in ordered:
                ordered.append(selector)
        return ordered


class RoutingConfig(BaseModel):
    """The ``routing:`` block."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=True,
        description=(
            "Master switch. False makes the gateway behave as though no policy were configured, "
            "so a misrouting policy can be turned off without editing or deleting it."
        ),
    )
    policies: dict[str, PolicySpec] = Field(default_factory=dict)
