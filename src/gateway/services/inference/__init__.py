"""The inference domain: idempotent completion requests, structured decisions and dialect bridges."""

from gateway.services.inference._decisions import (
    DecisionProvider,
    DecisionProviderError,
    UnknownDecisionProviderError,
    close_decision_client,
    decision_body,
    reported_charge,
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
    REASONING_ITEM_ID_PREFIX,
    aresponses_via_chat_completions,
    call_responses,
    serves_responses,
    uses_chat_completions_bridge,
)
from gateway.services.inference._responses_input import strip_gateway_minted_items
from gateway.services.inference._sweeper import run_idempotency_sweeper

__all__ = [
    "REASONING_ITEM_ID_PREFIX",
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
    "decision_body",
    "keep_claim_alive",
    "reported_charge",
    "request_decision",
    "resolve_decision_provider",
    "run_idempotency_sweeper",
    "serves_responses",
    "strip_gateway_minted_items",
    "uses_chat_completions_bridge",
]
