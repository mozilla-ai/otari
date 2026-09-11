"""The API root is one constant, and it has the shape a mount prefix needs."""

from gateway.core.config import API_ROOT, API_VERSION, OTLP_ROOT


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


def test_the_root_is_built_from_the_version_rather_than_parsed_back_out() -> None:
    """Anything needing the version reads it, instead of guessing where it sits in a path.

    Splitting it back out of the root works only while the root is exactly two
    segments ending in the version. It would answer "api" for a root of
    "/api", and "" for one with a trailing slash, and both silently.
    """
    assert API_VERSION == "v1"
    assert API_ROOT.endswith(f"/{API_VERSION}")
    assert API_ROOT == f"/api/{API_VERSION}"
    # No separator inside it, or a path built from the pair gains one.
    assert "/" not in API_VERSION
