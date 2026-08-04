"""Routing policies: turning a policy name into an ordered plan of attempts.

The decision half of routing. Something here decides which candidates to try and
in what order; the API layer's attempt walker executes the result and makes no
choices of its own.

Plain core code, with no port. ``ARCHITECTURE.md`` names a ``RoutingPort`` and
marks the routing capability line provisional, and whether that port should exist
at all is an open maintainer decision, so this does not presume one.
"""

from gateway.services.routing.compiler import (
    CompiledPlan,
    DroppedCandidate,
    NoEligibleCandidatesError,
    compile_policy,
    needs_budget_state,
)
from gateway.types.budget_state import BudgetState

__all__ = [
    "BudgetState",
    "CompiledPlan",
    "DroppedCandidate",
    "NoEligibleCandidatesError",
    "compile_policy",
    "needs_budget_state",
]
