"""Cross-SDK control-plane error-mapping conformance anchor (issue #226).

All four language SDKs (Python, TS, Rust, Go) wrap a generated ``_client`` core
with a hand-written shell that maps HTTP errors to the SDK's typed error
hierarchy. On the inference path the shell maps a 401 to the SDK's typed
authentication error; on the control-plane path (keys, users, budgets, pricing,
usage) it historically did not, so management calls leaked the raw
generated/transport error type. The per-SDK fixes are tracked in the SDK repos;
the cross-SDK conformance assertion that keeps them aligned is documented in
``tests/integration/README.md``.

This test pins the **server-side half** of that contract: the gateway must
return one uniform error contract on the control-plane path that is identical to
the inference path for the same auth failure, a 401 with a JSON
``{"detail": <str>}`` body. If the control-plane path ever diverged (a different
status, or a non-JSON / differently shaped body), the SDK shells could not map
both paths to the same typed error and the documented conformance assertion
would be unsatisfiable no matter what the shells did.
"""

from fastapi.testclient import TestClient

from gateway.core.config import API_KEY_HEADER

# A well-formed but unknown key: it passes ``validate_api_key_format`` so it
# reaches the hash lookup and misses, the closest inference-path analog to an
# invalid master key (both are recognized-shape-but-wrong credentials).
INVALID_API_KEY = "gw-" + "a" * 60

# The inference path guards on ``verify_api_key``; an unknown key yields 401.
INFERENCE_ENDPOINT = ("POST", "/v1/chat/completions")
INFERENCE_BODY = {"model": "openai:gpt-4o-mini", "messages": [{"role": "user", "content": "Hello"}]}

# One read endpoint on every standalone control-plane router, each guarded by
# ``verify_master_key``. Pricing's GET also accepts an API key, so its
# master-key-only POST stands in for the pricing surface.
CONTROL_PLANE_ENDPOINTS = [
    ("GET", "/v1/keys", None),
    ("GET", "/v1/users", None),
    ("GET", "/v1/budgets", None),
    ("GET", "/v1/usage", None),
    ("POST", "/v1/pricing", {"model_key": "openai:gpt-4o-mini", "input_cost_per_token": 0.0}),
]


def _invalid_auth_header() -> dict[str, str]:
    return {API_KEY_HEADER: f"Bearer {INVALID_API_KEY}"}


def _assert_typed_auth_error_shape(response_json: object) -> None:
    """The body must be a JSON object carrying a string ``detail``.

    This is the shape an SDK shell maps onto its typed authentication error; a
    missing/empty or non-JSON body is what forces the shell to surface the raw
    transport error instead.
    """
    assert isinstance(response_json, dict), f"expected a JSON object, got {type(response_json).__name__}"
    assert isinstance(response_json.get("detail"), str) and response_json["detail"]


def test_inference_invalid_credential_is_typed_auth_error(client: TestClient) -> None:
    """Baseline: the inference path returns 401 with a mappable JSON body."""
    method, path = INFERENCE_ENDPOINT
    response = client.request(method, path, json=INFERENCE_BODY, headers=_invalid_auth_header())

    assert response.status_code == 401
    assert response.headers["content-type"].startswith("application/json")
    _assert_typed_auth_error_shape(response.json())


def test_control_plane_shares_the_inference_auth_error_contract(client: TestClient) -> None:
    """Conformance: every control-plane endpoint mirrors the inference path.

    An invalid master key on any control-plane call surfaces the *same* status
    and body shape the inference path raises on an invalid key, so an SDK shell
    can map both to one typed authentication error rather than leaking the raw
    generated/transport error on management calls.
    """
    inference_method, inference_path = INFERENCE_ENDPOINT
    baseline = client.request(inference_method, inference_path, json=INFERENCE_BODY, headers=_invalid_auth_header())
    assert baseline.status_code == 401

    for method, path, body in CONTROL_PLANE_ENDPOINTS:
        response = client.request(method, path, json=body, headers=_invalid_auth_header())

        assert response.status_code == baseline.status_code, (
            f"{method} {path} returned {response.status_code}, "
            f"but the inference path returns {baseline.status_code} for the same auth failure"
        )
        assert response.headers["content-type"].startswith("application/json"), (
            f"{method} {path} returned a non-JSON error body: {response.headers.get('content-type')!r}"
        )
        _assert_typed_auth_error_shape(response.json())
