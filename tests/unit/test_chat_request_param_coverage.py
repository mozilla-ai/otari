"""Guard against the gateway silently dropping OpenAI completion params.

The OpenAI-compatible chat request schema (``ChatCompletionRequest``) is a hand-maintained
allowlist. The gateway forwards only the fields it declares to ``acompletion(**kwargs)``, so a
parameter any-llm understands can be left out of the schema and silently dropped before the
provider call (this is how ``reasoning_effort`` ended up being ignored, mozilla-ai/otari#150).

These tests pin that gap against any-llm's ``CompletionParams`` (the gateway already depends on
any-llm): every completion param must either be accepted by the gateway schema or be listed,
with a reason, in ``KNOWN_UNSUPPORTED``. A new any-llm param then fails CI until someone makes a
conscious decision, instead of being dropped unnoticed.
"""

from any_llm.types.completion import CompletionParams

from gateway.api.routes.chat import ChatCompletionRequest

# any-llm ``CompletionParams`` fields the gateway's chat schema does not accept today. Each entry
# needs a reason: a naming difference, a deliberate non-support decision, or a tracked bug.
# Remove an entry once the field is added to ``ChatCompletionRequest`` (the accuracy test below
# enforces that the allowlist and the schema never overlap).
KNOWN_UNSUPPORTED: dict[str, str] = {
    "model_id": "any-llm's internal name; the gateway exposes it as `model`",
    # Standard OpenAI completion params the gateway does not currently forward; triage (forward or
    # document as intentional) is tracked in mozilla-ai/otari#152. Remove each entry as it is resolved.
    "n": "not currently forwarded (mozilla-ai/otari#152)",
    "stop": "not currently forwarded (mozilla-ai/otari#152)",
    "seed": "not currently forwarded (mozilla-ai/otari#152)",
    "presence_penalty": "not currently forwarded (mozilla-ai/otari#152)",
    "frequency_penalty": "not currently forwarded (mozilla-ai/otari#152)",
    "parallel_tool_calls": "not currently forwarded (mozilla-ai/otari#152)",
    "logprobs": "not currently forwarded (mozilla-ai/otari#152)",
    "top_logprobs": "not currently forwarded (mozilla-ai/otari#152)",
    "logit_bias": "not currently forwarded (mozilla-ai/otari#152)",
}


def test_chat_request_covers_any_llm_completion_params() -> None:
    """Every any-llm completion param is accepted by the gateway or explicitly allow-listed."""
    any_llm_fields = set(CompletionParams.model_fields)
    gateway_fields = set(ChatCompletionRequest.model_fields)

    uncovered = any_llm_fields - gateway_fields - set(KNOWN_UNSUPPORTED)

    assert not uncovered, (
        f"any-llm CompletionParams the gateway neither accepts nor allow-lists, so they are "
        f"silently dropped before the provider call: {sorted(uncovered)}. Add them to "
        f"ChatCompletionRequest (they are then forwarded), or add to KNOWN_UNSUPPORTED with a reason."
    )


def test_known_unsupported_allowlist_stays_accurate() -> None:
    """The allowlist never lists a field that is actually supported or no longer exists."""
    any_llm_fields = set(CompletionParams.model_fields)
    gateway_fields = set(ChatCompletionRequest.model_fields)

    now_supported = set(KNOWN_UNSUPPORTED) & gateway_fields
    assert not now_supported, (
        f"these are now accepted by ChatCompletionRequest; remove them from KNOWN_UNSUPPORTED: {sorted(now_supported)}"
    )

    stale = set(KNOWN_UNSUPPORTED) - any_llm_fields
    assert not stale, (
        f"these are no longer any-llm CompletionParams; remove them from KNOWN_UNSUPPORTED: {sorted(stale)}"
    )
