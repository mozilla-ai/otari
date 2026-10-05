"""The inference domain: idempotent completion requests and structured decisions."""

from gateway.services.inference._decisions import (
    DecisionProvider,
    DecisionProviderError,
    UnknownDecisionProviderError,
    close_decision_client,
    request_decision,
    resolve_decision_provider,
)
from gateway.services.inference._idempotency import (
    Admission,
    BlockedCaller,
    Claimed,
    IdempotencyService,
    IdempotentRequest,
    InvalidKey,
    KeyReused,
    Replay,
    StillInFlight,
    UnknownCaller,
)
from gateway.services.inference._lease import keep_claim_alive
from gateway.services.inference._sweeper import run_idempotency_sweeper

__all__ = [
    "Admission",
    "BlockedCaller",
    "Claimed",
    "DecisionProvider",
    "DecisionProviderError",
    "IdempotencyService",
    "IdempotentRequest",
    "InvalidKey",
    "KeyReused",
    "Replay",
    "StillInFlight",
    "UnknownDecisionProviderError",
    "UnknownCaller",
    "close_decision_client",
    "keep_claim_alive",
    "request_decision",
    "resolve_decision_provider",
    "run_idempotency_sweeper",
]
