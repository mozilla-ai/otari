"""The inference domain: deduplicating completion requests that carry an idempotency key."""

from gateway.services.inference._idempotency import (
    Admission,
    Claimed,
    IdempotencyService,
    IdempotentRequest,
    KeyReused,
    Replay,
    StillInFlight,
    storage_available,
)
from gateway.services.inference._sweeper import run_idempotency_sweeper

__all__ = [
    "Admission",
    "Claimed",
    "IdempotencyService",
    "IdempotentRequest",
    "KeyReused",
    "Replay",
    "StillInFlight",
    "run_idempotency_sweeper",
    "storage_available",
]
