"""Unit tests for the provenance-slug reservation (``is_reserved_source``).

Two callers read it and neither is a good place to pin the case handling: the
batch schema turns it into a 422 and the OTLP route swallows it into a fallback
source, so both hide which spellings are reserved behind a status code.
"""

import pytest

from gateway.services.external_usage_service import is_reserved_source


@pytest.mark.parametrize("value", ["gateway", "GATEWAY", "Gateway"])
def test_reserved_slugs_are_refused(value: str) -> None:
    assert is_reserved_source(value)


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
    assert not is_reserved_source(value)
