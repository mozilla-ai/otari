"""Unit tests for provider-reported latency capture (mozilla-ai/otari#337).

Provider timing (Groq's total_time, Ollama's prompt_eval_duration +
eval_duration) survives in ``usage.model_extra`` because any-llm's usage types
allow extra fields, but ``GatewayUsage.from_completion_usage`` used to rebuild
a fresh object naming only its own explicit fields, silently dropping any
extras from the source. These tests cover the fix in both directions: the
extras now survive the rebuild, and ``provider_latency_ms_of`` normalizes what
it finds there. Ollama's own ``total_duration`` is deliberately not used here:
it includes model load time, which is not compute time.
"""

from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionUsage

from gateway.api.routes.chat import _ChatAdapter
from gateway.core.usage import GatewayUsage, provider_latency_ms_of


def test_provider_latency_ms_of_normalizes_groq_seconds() -> None:
    usage = CompletionUsage.model_construct(
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
        queue_time=0.01,
        prompt_time=0.02,
        completion_time=0.03,
        total_time=0.06,
    )
    assert provider_latency_ms_of(usage, "groq") == 60


def test_provider_latency_ms_of_normalizes_ollama_nanoseconds() -> None:
    usage = CompletionUsage.model_construct(
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
        prompt_eval_duration=23456789,
        eval_duration=100000000,
    )
    assert provider_latency_ms_of(usage, "ollama") == 123


def test_provider_latency_ms_of_excludes_ollama_load_duration() -> None:
    """total_duration includes load_duration; only the two eval fields count."""
    usage = CompletionUsage.model_construct(
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
        prompt_eval_duration=23456789,
        eval_duration=100000000,
        load_duration=5_000_000_000,
        total_duration=5_123_456_789,
    )
    assert provider_latency_ms_of(usage, "ollama") == 123


def test_provider_latency_ms_of_none_when_ollama_reports_only_total_duration() -> None:
    """Without both eval fields there is nothing to sum; never fall back to total_duration."""
    usage = CompletionUsage.model_construct(
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
        total_duration=123456789,
    )
    assert provider_latency_ms_of(usage, "ollama") is None


def test_provider_latency_ms_of_none_for_unmapped_provider() -> None:
    usage = CompletionUsage.model_construct(prompt_tokens=1, completion_tokens=1, total_tokens=2, total_time=0.06)
    assert provider_latency_ms_of(usage, "openai") is None
    assert provider_latency_ms_of(usage, None) is None


def test_provider_latency_ms_of_none_when_field_absent() -> None:
    usage = CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    assert provider_latency_ms_of(usage, "groq") is None
    assert provider_latency_ms_of(usage, "ollama") is None


def test_provider_latency_ms_of_none_on_malformed_value() -> None:
    """A field present but not a plain number never raises or fakes a number."""
    usage = CompletionUsage.model_construct(prompt_tokens=1, completion_tokens=1, total_tokens=2, total_time="soon")
    assert provider_latency_ms_of(usage, "groq") is None
    boolean = CompletionUsage.model_construct(prompt_tokens=1, completion_tokens=1, total_tokens=2, total_time=True)
    assert provider_latency_ms_of(boolean, "groq") is None


def test_provider_latency_ms_of_none_on_non_finite_value() -> None:
    """inf/nan pass isinstance(float) and used to reach round(), which raises on both."""
    inf_usage = CompletionUsage.model_construct(
        prompt_tokens=1, completion_tokens=1, total_tokens=2, total_time=float("inf")
    )
    assert provider_latency_ms_of(inf_usage, "groq") is None
    nan_usage = CompletionUsage.model_construct(
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
        prompt_eval_duration=float("nan"),
        eval_duration=100000000,
    )
    assert provider_latency_ms_of(nan_usage, "ollama") is None


def test_provider_latency_ms_of_none_when_scaling_overflows() -> None:
    """A finite raw value can still overflow to inf after the unit multiplier, and round() raises on inf."""
    huge_usage = CompletionUsage.model_construct(
        prompt_tokens=1, completion_tokens=1, total_tokens=2, total_time=1e308
    )
    assert provider_latency_ms_of(huge_usage, "groq") is None


def test_provider_latency_ms_of_none_on_negative_value() -> None:
    """A negative duration is malformed, not a fast response; zero stays valid."""
    negative_usage = CompletionUsage.model_construct(
        prompt_tokens=1, completion_tokens=1, total_tokens=2, total_time=-0.01
    )
    assert provider_latency_ms_of(negative_usage, "groq") is None
    zero_usage = CompletionUsage.model_construct(prompt_tokens=1, completion_tokens=1, total_tokens=2, total_time=0)
    assert provider_latency_ms_of(zero_usage, "groq") == 0


def test_provider_latency_ms_of_none_when_int_multiply_overflows() -> None:
    """An oversized int raw value raises OverflowError on int*float, not caught by isfinite on a float."""
    oversized_usage = CompletionUsage.model_construct(
        prompt_tokens=1, completion_tokens=1, total_tokens=2, total_time=10**400
    )
    assert provider_latency_ms_of(oversized_usage, "groq") is None


def test_from_completion_usage_forwards_provider_extras() -> None:
    base = CompletionUsage.model_construct(
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        total_time=0.06,
    )
    usage = GatewayUsage.from_completion_usage(base)
    assert provider_latency_ms_of(usage, "groq") == 60


def test_from_completion_usage_explicit_fields_win_over_same_named_extra() -> None:
    """A same-named extra can never shadow an explicit cache-count kwarg."""
    base = CompletionUsage.model_construct(
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        cache_read_tokens=999,
    )
    usage = GatewayUsage.from_completion_usage(base, cache_read_tokens=7)
    assert usage.cache_read_tokens == 7


def test_chat_non_stream_forwards_groq_timing_to_settlement() -> None:
    result = ChatCompletion.model_construct(
        usage=CompletionUsage.model_construct(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            total_time=0.25,
        ),
    )
    usage = _ChatAdapter().extract_usage(result)
    assert isinstance(usage, GatewayUsage)
    assert provider_latency_ms_of(usage, "groq") == 250


def test_chat_stream_forwards_ollama_timing_to_settlement() -> None:
    chunk = ChatCompletionChunk.model_construct(
        usage=CompletionUsage.model_construct(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            prompt_eval_duration=500_000,
            eval_duration=1_500_000,
        ),
    )
    usage = _ChatAdapter().extract_stream_usage(chunk)
    assert isinstance(usage, GatewayUsage)
    assert provider_latency_ms_of(usage, "ollama") == 2


def test_chat_stream_never_lets_an_extra_set_a_real_accounting_field() -> None:
    """A provider-controlled extra must never reach a GatewayUsage billing field.

    extract_stream_usage's own update() only pins prompt/completion/total tokens,
    prompt_tokens_details and cache_read_tokens; before external_extras this left
    cache_write_tokens, cache_write_1h_tokens and cache_tokens_in_prompt open to
    whatever a provider put under the same name in model_extra.
    """
    chunk = ChatCompletionChunk.model_construct(
        usage=CompletionUsage.model_construct(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            cache_write_tokens=999,
            cache_write_1h_tokens=999,
            cache_tokens_in_prompt=False,
        ),
    )
    usage = _ChatAdapter().extract_stream_usage(chunk)
    assert isinstance(usage, GatewayUsage)
    assert usage.cache_write_tokens == 0
    assert usage.cache_write_1h_tokens == 0
    assert usage.cache_tokens_in_prompt is True
