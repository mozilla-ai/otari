"""Unit tests for the chat endpoint's output-cap fold (mozilla-ai/otari-ai#1062).

OpenAI renamed the output cap to ``max_completion_tokens`` and deprecated
``max_tokens``, so an OpenAI-compatible client sends the current name and
any-llm forwards both fields verbatim. A provider whose param conversion
predates the rename then either rejects the request (Anthropic's SDK raises a
TypeError, which surfaced as a 502) or silently drops the cap (Google's honors
``max_tokens`` alone), so the gateway collapses the two spellings into one
before dispatch.

The fold is also what the budget reservation estimates against, so the value
tested here is the value the request reserves for and the value the provider is
told, in every combination a client can send.
"""

import logging

import pytest
from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.types.completion import CompletionParams

from gateway.api.routes.chat import _effective_output_cap
from gateway.log_config import logger as gateway_logger


def test_current_name_alone_is_the_cap() -> None:
    """The field an unmodified OpenAI-compatible client sends."""
    assert _effective_output_cap(None, 20) == 20


def test_legacy_name_alone_is_the_cap() -> None:
    """The deprecated spelling keeps working, unchanged.

    An explicit ``"max_completion_tokens": null`` reaches this helper as the same
    call, since a null and an unset field are indistinguishable once pydantic has
    parsed the body; that they differ on the wire is pinned where it is visible,
    in ``tests/integration/test_output_cap_fold.py``.
    """
    assert _effective_output_cap(20, None) == 20


def test_current_name_wins_when_both_are_sent() -> None:
    """Matches OpenAI's deprecation and the precedence any-llm's OpenAI layer
    applies to the same pair, so the cap the gateway reserves against and the
    cap the provider enforces agree."""
    assert _effective_output_cap(300, 20) == 20


def test_neither_leaves_the_cap_unset() -> None:
    """No cap stays no cap: the provider applies its own default, and the budget
    estimate falls back to its configured default output tokens."""
    assert _effective_output_cap(None, None) is None


def test_zero_is_a_cap_and_not_an_absent_value() -> None:
    """Zero is falsy but explicit; the fold keys on ``is None`` so a caller's
    zero reaches the provider (and the estimate) rather than reading as unset."""
    assert _effective_output_cap(None, 0) == 0
    assert _effective_output_cap(0, None) == 0


def _caps_with_logs(
    max_tokens: int | None, max_completion_tokens: int | None, caplog: pytest.LogCaptureFixture
) -> tuple[int | None, str]:
    """Fold a pair of caps, capturing what the (non-propagating) gateway logger said."""
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.WARNING, logger="gateway")
    try:
        return _effective_output_cap(max_tokens, max_completion_tokens), caplog.text
    finally:
        gateway_logger.removeHandler(caplog.handler)


def test_two_different_caps_are_logged_with_both_values(caplog: pytest.LogCaptureFixture) -> None:
    """The discarded value leaves no trace on the wire, so the log is the only
    place an operator can learn why a request stopped at a limit the caller
    thought it had raised. Both numbers are named, since which one lost is the
    whole question."""
    cap, text = _caps_with_logs(300, 20, caplog)
    assert cap == 20
    assert "max_completion_tokens=20" in text
    assert "max_tokens=300" in text


def test_two_equal_caps_are_not_a_contradiction(caplog: pytest.LogCaptureFixture) -> None:
    """Nothing is discarded when both fields say the same thing, so warning
    would train an operator to ignore the message that matters."""
    cap, text = _caps_with_logs(20, 20, caplog)
    assert cap == 20
    assert text == ""


@pytest.mark.parametrize(
    ("max_tokens", "max_completion_tokens"),
    [(20, None), (None, 20), (None, None)],
)
def test_one_cap_or_none_stays_quiet(
    max_tokens: int | None, max_completion_tokens: int | None, caplog: pytest.LogCaptureFixture
) -> None:
    """The ordinary requests: nothing ambiguous, nothing to say."""
    _, text = _caps_with_logs(max_tokens, max_completion_tokens, caplog)
    assert text == ""


def test_any_llm_still_remaps_the_legacy_name_for_openai_providers() -> None:
    """The upstream behavior the fold depends on, pinned because it is not ours.

    Folding onto the deprecated spelling is only lossless because any-llm's
    OpenAI layer renames it back on the way out, which is what lets an OpenAI
    reasoning model (which rejects ``max_tokens`` outright) keep working through
    this gateway. That layer belongs to any-llm, so a version bump could drop the
    remap and the fold would start sending the deprecated name to the one
    provider family that refuses it, with nothing else in the suite noticing.

    Asked in review on mozilla-ai/otari#769: why fold onto the deprecated field.
    """
    converted = BaseOpenAIProvider._convert_completion_params(
        CompletionParams(model_id="gpt-5", messages=[{"role": "user", "content": "hi"}], max_tokens=20)
    )
    assert converted.get("max_completion_tokens") == 20
    assert "max_tokens" not in converted
