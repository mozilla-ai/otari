"""The ``Retry-After`` value a deployment relays from a peer it called."""

import math

MAX_RETRY_AFTER_SECONDS = 24 * 60 * 60


def bounded_retry_after(raw: object) -> str | None:
    """Return ``raw`` as whole seconds capped at one day, or ``None`` when it is not a usable delay.

    The result is a re-serialized number, so no text a peer chooses reaches a response header.
    Fractional seconds round up, so a client does not retry before the delay the peer named.
    An HTTP date is dropped, because a value that cannot be bounded is not safe to relay.
    """
    if not isinstance(raw, str):
        return None
    try:
        parsed = float(raw.strip())
    except ValueError:
        return None
    # NOTE: ``float`` accepts "inf" and "1e400", and ``math.ceil`` raises OverflowError on both.
    if not math.isfinite(parsed) or parsed < 0:
        return None
    return str(min(math.ceil(parsed), MAX_RETRY_AFTER_SECONDS))
