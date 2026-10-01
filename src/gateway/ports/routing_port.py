"""Choosing the provider/model attempts for a request: which router backends a build has.

A routing policy's ``select`` may name a router backend (``{router: knn, ...}``).
The pipeline asks this port for the backend by that name, and the backend ranks
the policy's candidates for one request. Everything else about the policy
(conditions, ``on_failure``, the allow-list, guardrails) stays in the core
compiler, so a backend is one decision inside a policy rather than a second
routing system.

The core adapter answers with the backends this build ships. An overlay rebinds
the port to add a backend of its own, or to replace one, without editing the
compiler or the pipeline.

Lookups are synchronous on purpose: they pick an object, they do no I/O, and
synchronous surfaces (``explain``, a policy write) ask them too. The work is in
the backend, whose methods are asynchronous.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

__all__ = [
    "RouterBackend",
    "RouterTraits",
    "RoutingContext",
    "RoutingDecision",
    "RoutingPort",
]


@dataclass
class RoutingContext:
    """Inputs a backend may use to rank candidates for a single request.

    The prompt arrives as already-flattened text rather than as wire messages.
    Flattening is format-specific (chat, Anthropic messages, and responses all
    shape content differently) and the API layer already does it for guardrails,
    so a backend never has to know which endpoint it is serving.
    """

    user_id: str
    default_model: str
    """The policy's default target: what serves if the backend declines, and the
    safe choice a low-confidence decision leads with."""
    candidate_pool: list[str]
    """The policy's candidates, already filtered to what this caller may use."""
    workspace_id: uuid.UUID | None = None
    """The workspace this request bills to, when there is one. A backend that
    reads stored state partitions on it as well as on ``user_id``, so a user
    holding keys in two workspaces does not have one steer the other. ``None``
    only where there is no request (a synchronous surface), which is also where
    no backend is asked to rank."""
    task_signal: str = ""
    """This turn's prompt text. What ``step`` granularity routes on."""
    trace_signal: str = ""
    """The conversation's opening prompt text. What ``trace_sticky`` routes on, so
    every turn of one conversation embeds the same thing."""
    trace_anchor: str = ""
    """Stable text identifying the conversation when the client sends no id."""
    task_id: str | None = None
    has_tools: bool = False
    is_trace_continuation: bool = False
    trace_key: str | None = None
    weights: dict[str, float] = field(default_factory=dict)
    """Per-candidate traffic weights the policy declared, for a backend that takes
    its parameters from the policy document rather than from the environment. Empty
    for every other backend. Kept as declared: normalizing needs ``candidate_pool``,
    which is already filtered to this caller."""


@dataclass
class RoutingDecision:
    """A backend's ranking for one request, best first.

    An empty ``ordered_models`` is a decline: a normal outcome (cold pool, sparse
    neighborhood, no embeddable signal) that leaves the policy's default target to
    serve. ``rationale`` is operator-facing text saying which of those it was.
    """

    ordered_models: list[str]
    confidence: float
    rationale: str
    log_decision: bool = True
    """Whether this decision earns its INFO line. A learned router's pick is
    unreconstructable after the fact and worth one; a load balancer's draw is one
    line per request forever, and the usage row already records what served. The
    backend decides, because only it knows how often it is asked."""

    @classmethod
    def decline(cls, rationale: str) -> RoutingDecision:
        return cls(ordered_models=[], confidence=0.0, rationale=rationale)


@runtime_checkable
class RouterBackend(Protocol):
    """Contract a router backend implements."""

    async def rank(self, ctx: RoutingContext) -> RoutingDecision: ...


@dataclass(frozen=True)
class RouterTraits:
    """What the rest of the gateway needs to know about a backend without asking it to rank.

    ``teachable``: the policy's candidates are a pool the routing-memory API is
    taught about (``POST /routing/preferences/rank``), so their spellings decide
    which score keys a user may send. ``requires_pricing``: the backend weighs
    candidates against their price and declines every request while one is
    unpriced, so a policy write refuses an unpriced candidate.
    """

    teachable: bool
    requires_pricing: bool


class RoutingPort(Protocol):
    """The router backends this build can hand a policy's ``select``."""

    def backend(self, name: str) -> RouterBackend | None:
        """The backend a policy named, or ``None`` when this build has none by that name.

        ``None`` is not an error: the policy then serves its default target. The
        name is matched case-insensitively and with surrounding whitespace ignored,
        the way a policy document is validated.
        """
        ...

    def known_backends(self) -> tuple[str, ...]:
        """The backend names this build resolves, for a message that lists them."""
        ...

    def traits(self, name: str | None) -> RouterTraits:
        """What a policy naming ``name`` asks of the rest of the gateway.

        Answers for a name this build does not have as well, since a policy can
        name one.
        """
        ...
