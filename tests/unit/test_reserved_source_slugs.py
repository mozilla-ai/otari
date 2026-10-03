"""Unit tests for the provenance-slug reservation (``reserved_source_reason``).

Two callers read it and neither is a good place to pin the case handling: the
batch schema turns a reason into a 422 and the OTLP route swallows it into a
fallback source, so both hide which spellings are reserved behind a status code.
"""

import pytest

from gateway.services.external_usage_service import reserved_source_reason


@pytest.mark.parametrize(
    "value",
    [
        "gateway",
        "GATEWAY",
        "Gateway",
    ],
)
def test_reserved_slugs_are_refused(value: str) -> None:
    reason = reserved_source_reason(value)
    assert reason is not None
    assert "reserved" in reason


@pytest.mark.parametrize(
    "value",
    [
        # A colon is an ordinary slug character.
        "foo:bar",
        "claude_code",
        "gateway-mirror",
        "not-gateway",
        "otel",
    ],
)
def test_free_slugs_are_allowed(value: str) -> None:
    assert reserved_source_reason(value) is None


def test_exact_match_message_names_the_matched_slug() -> None:
    # The message is built from the value that matched, so a second entry in
    # RESERVED_SOURCES cannot inherit "gateway" in its own rejection.
    assert reserved_source_reason("GATEWAY") == (
        "source 'gateway' is reserved for usage Otari served itself; pick another slug."
    )
