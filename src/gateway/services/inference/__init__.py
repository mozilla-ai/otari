"""The inference domain: idempotent completion requests, structured decisions and dialect bridges."""

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
from gateway.services.inference._responses_bridge import (
    aresponses_via_chat_completions,
    call_responses,
    serves_responses,
    uses_chat_completions_bridge,
)
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
    "aresponses_via_chat_completions",
    "call_responses",
    "close_decision_client",
    "keep_claim_alive",
    "request_decision",
    "resolve_decision_provider",
    "run_idempotency_sweeper",
    "serves_responses",
    "uses_chat_completions_bridge",
]
