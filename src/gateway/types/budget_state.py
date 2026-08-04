"""What a routing policy's budget conditions can read about a caller.

A leaf type so the budget service can produce it and the routing compiler can
consume it without either importing the other.

``None`` on a field means *undefined*, which is the honest answer for a caller
with no budget row, no budget attached, an unlimited budget, or the master key. An
undefined value never matches a condition, so a policy falls through to its
default rather than raising: "this user has no budget configured" must not become
a 500 on every request through the policy.
"""

from dataclasses import dataclass

__all__ = ["BudgetState"]


@dataclass(frozen=True)
class BudgetState:
    """Budget numbers a ``when`` clause may compare against."""

    used_pct: float | None = None
    """Percentage of the budget already committed (``spend + reserved``), which is
    the same total the budget gate enforces. Reading bare ``spend`` instead would
    let a tier-down rule see a smaller number than the gate does and fire late."""

    remaining_usd: float | None = None
    """USD left before the cap."""
