"""The API root is one constant, and it has the shape a mount prefix needs."""

from gateway.core.config import API_ROOT, OTLP_ROOT


def test_api_root_is_the_documented_value() -> None:
    assert API_ROOT == "/api/v1"


def test_otlp_root_is_a_sibling_not_a_child() -> None:
    assert OTLP_ROOT == "/otlp"
    assert not OTLP_ROOT.startswith(API_ROOT)


def test_roots_are_valid_fastapi_prefixes() -> None:
    # FastAPI rejects a prefix that ends with "/" or does not start with one.
    for root in (API_ROOT, OTLP_ROOT):
        assert root.startswith("/")
        assert not root.endswith("/")
