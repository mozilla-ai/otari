"""What a budget refusal says, for the per-user budget and every scoped ceiling alike."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

from gateway.services.budgets._periods import seconds_until

_MICRO_DOLLAR = Decimal("0.000001")


@dataclass(frozen=True)
class BlockedAxis:
    """The capped axis that refused a hold, and why.

    ``remaining`` is set only when the dollar axis was under its cap and this
    request's hold (``held``) did not fit in what was left. That refusal is about
    the size of the request, not about spend already made, so it reads differently.

    ``exceeds_cap`` is set when the hold alone is larger than the cap, so a fresh
    window would refuse it too. ``resets_at`` is when the refusing budget's window
    ends, None for a budget that never resets.
    """

    name: str
    remaining: Decimal | None = None
    held: Decimal | None = None
    exceeds_cap: bool = False
    resets_at: datetime | None = None

    def retry_after(self, now: datetime) -> int | None:
        """Seconds until the refusing budget's window ends, or None when waiting would not let the request in."""
        if self.resets_at is None or self.exceeds_cap:
            return None
        return seconds_until(self.resets_at, now)


def blocked_on(
    name: str,
    *,
    committed: Decimal,
    cap: Decimal,
    wanted: Decimal,
    new_request: bool,
) -> BlockedAxis | None:
    """Apply one axis's admission test, returning the refusal or None when it admits.

    The test is the gate's own, both clauses: a hold that would push the axis
    past its cap, and, for an arrival only, an axis already at or over its cap.
    """
    exceeds_cap = wanted > cap
    if committed + wanted > cap:
        if name == "budget" and committed < cap:
            return BlockedAxis(name, remaining=cap - committed, held=wanted, exceeds_cap=exceeds_cap)
        return BlockedAxis(name, exceeds_cap=exceeds_cap)
    if new_request and committed >= cap:
        return BlockedAxis(name, exceeds_cap=exceeds_cap)
    return None


def _usd(amount: Decimal, rounding: str) -> str:
    """A dollar amount to the micro-dollar, without trailing zeros past the cent."""
    text = f"{amount.quantize(_MICRO_DOLLAR, rounding=rounding):f}"
    whole, _, fraction = text.partition(".")
    return f"${whole}.{fraction.rstrip('0').ljust(2, '0')}"


def refusal_detail(
    subject: str,
    blocked: BlockedAxis,
    *,
    new_request: bool,
    reserved_output_tokens: int | None = None,
) -> str:
    """The 403 ``detail`` for a budget refusal.

    A spent allowance keeps the wording it has always had. A hold that does not
    fit says so, with both amounts: the estimate is rounded up and the remainder
    down, so the two never print as equal. ``reserved_output_tokens`` is the
    output allowance the estimate was priced at, named only where the caller's
    output token limit decided it. The field is not named, because each API
    spells it differently.
    """
    if blocked.remaining is None or blocked.held is None:
        return f"{subject} has exceeded {blocked.name} limit"
    estimate = _usd(blocked.held, ROUND_CEILING)
    remaining = _usd(blocked.remaining, ROUND_FLOOR)
    if not new_request:
        return (
            f"{subject} has {remaining} of budget remaining, "
            f"less than the {estimate} this request's estimated cost grew by"
        )
    detail = f"{subject} has {remaining} of budget remaining, less than this request's estimated cost of {estimate}"
    if reserved_output_tokens is not None and reserved_output_tokens > 0:
        detail += (
            f". The estimate assumes {reserved_output_tokens} output tokens; "
            "a lower output token limit on the request may let it fit"
        )
    return detail
