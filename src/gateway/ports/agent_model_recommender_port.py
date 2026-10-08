"""Which model a coding agent's subagent should start on.

A harness about to spawn a subagent asks before it starts it, and runs the
whole subagent on the answer. Who answers is the seam. The core puts one
choice question to the decision model the deployment configured, which may be
a local one, and a hosted build binds a recommender of its own behind the same
port: one the caller never configures or sees, and whose price is the build's
to set (``ARCHITECTURE.md``, rule 7: a hosted service runs through it).

The port only recommends. Metering is the caller's: it asks :meth:`quote`
for the label and the price before the call, holds against them, and settles
what :class:`RecommendationUsage` reports after it.

Stability: this interface is not frozen while Otari is pre-1.0. Overlay authors
should pin a released tag and expect the shape to move.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from gateway.exceptions.routing_exceptions import (
    RecommendationFailedError,
    RecommenderNotConfiguredError,
    UnreadableRecommendationError,
)


@dataclass(frozen=True)
class SubagentSpawn:
    """What a harness knows at the moment it spawns a subagent."""

    harness: str
    """The harness asking, such as ``claude-code``."""
    session_id: str
    tool_use_id: str
    agent_type: str
    """A built-in such as ``Explore``, or a custom agent's name."""
    description: str
    prompt: str
    """The task the subagent is given. Read for the recommendation, never stored or logged."""
    parent_model: str
    requested_model: str | None
    """The model the harness asked for, if any: a fact, not a hint. The recommendation is the gateway's."""


@dataclass(frozen=True)
class Quote:
    """What a recommendation is metered as, known before the call.

    ``provider`` and ``model`` are the label the usage row, the allow-list and
    any pricing row key on. ``charge`` is the price of one recommendation
    where the recommender owns the price, a service fee the caller holds before
    the call. ``None`` is pass-through: the caller pays what the upstream
    charged, priced from the caller's own rows.
    """

    provider: str
    model: str
    charge: Decimal | None = None


@dataclass(frozen=True)
class RecommendationUsage:
    """What one recommendation consumed."""

    input_tokens: int = 0
    output_tokens: int = 0
    charge: Decimal | None = None
    """The final price where the recommender names one. ``None`` leaves it to the caller's rows, or to the quote."""


@dataclass(frozen=True)
class ModelRecommendation:
    """The model recommended for the subagent, and what deciding it consumed."""

    model: str
    """One of the candidates, as the harness spells it."""
    reason: str | None
    probabilities: Mapping[str, float] | None
    usage: RecommendationUsage


class AgentModelRecommenderPort(Protocol):
    """What a build must answer to pick a subagent's model."""

    def quote(self) -> Quote:
        """The label and, where the recommender owns the price, the price of one recommendation.

        Asked before every call and before anything is held, so it reads
        configuration and nothing else.

        Raises:
            RecommenderNotConfiguredError: nothing can answer on this deployment.

        """
        ...

    async def recommend(self, spawn: SubagentSpawn, *, candidates: Mapping[str, str | None]) -> ModelRecommendation:
        """Pick one of ``candidates`` for ``spawn``.

        ``candidates`` maps each model name, as the harness spells it, to what
        it is for, or ``None`` where the name says enough. The answer's
        ``model`` is one of its keys.

        Raises:
            RecommendationFailedError: the recommender could not answer.
            UnreadableRecommendationError: it answered, but named no candidate.

        """
        ...


__all__ = [
    "AgentModelRecommenderPort",
    "ModelRecommendation",
    "Quote",
    "RecommendationFailedError",
    "RecommendationUsage",
    "RecommenderNotConfiguredError",
    "SubagentSpawn",
    "UnreadableRecommendationError",
]
