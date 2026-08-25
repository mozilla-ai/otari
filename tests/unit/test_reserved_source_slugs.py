"""Unit tests for the provenance-slug reservation (``reserved_source_reason``).

Two callers read it and neither is a good place to pin the case handling: the
batch schema turns a reason into a 422 and the OTLP route swallows it into a
fallback source, so both hide which spellings are reserved behind a status code.

The boundary that matters is the colon. otari.ai's reconciliation scopes the rows
it wrote with ``source LIKE 'otari-ai:%'``, so `otari-ai:` and anything under it
is what collides; a bare `otari-ai` matches nothing there and stays a legal slug
for an importer to use.
"""

import pytest

from gateway.services.external_usage_service import reserved_source_reason


@pytest.mark.parametrize(
    "value",
    [
        "gateway",
        "GATEWAY",
        "Gateway",
        "otari-ai:",
        "otari-ai:gateway",
        "otari-ai:claude_code",
        "OTARI-AI:gateway",
        "Otari-AI:Claude_Code",
    ],
)
def test_reserved_slugs_are_refused(value: str) -> None:
    reason = reserved_source_reason(value)
    assert reason is not None
    assert "reserved" in reason


@pytest.mark.parametrize(
    "value",
    [
        # The colon is the boundary: reconciliation scopes on `otari-ai:%`, so the
        # bare name and a name that merely starts with it collide with nothing.
        "otari-ai",
        "otari-ai-mirror",
        "otari_ai:gateway",
        # A colon elsewhere is ordinary: the guard is a prefix check, not a ban on
        # the character the slug pattern already allows.
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
