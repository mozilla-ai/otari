"""Unit tests for building Bedrock's ``client_args`` for a hybrid-mode attempt."""

from __future__ import annotations

from typing import Any

from gateway.services.bedrock_gateway_auth import build_bedrock_client_args


class _FakeRequest:
    def __init__(self) -> None:
        self.headers: dict[str, str] = {}


def test_classic_shape_aliases_secret_and_forwards_region() -> None:
    """The classic IAM access-key/secret-key shape (aws_access_key_id present
    in extra_params) forwards region_name/aws_access_key_id unchanged and
    aliases the resolved attempt's api_key into aws_secret_access_key, since
    any-llm-sdk's BedrockProvider never reads a plain api_key when building
    its boto3 client."""
    client_args = build_bedrock_client_args(
        "my-secret-access-key",
        {"region_name": "us-east-1", "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE"},
    )

    assert client_args == {
        "region_name": "us-east-1",
        "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
        "aws_secret_access_key": "my-secret-access-key",
    }


def test_bearer_shape_builds_client_with_injected_authorization_header() -> None:
    """The bearer-token ("Bedrock API key") shape (no aws_access_key_id) gets
    a pre-built, unsigned boto3 client instead of plain credential kwargs;
    the client injects `Authorization: Bearer <token>` on every request via a
    before-sign hook, since this boto3 version has no native support for
    AWS_BEARER_TOKEN_BEDROCK / an aws_bearer_token constructor kwarg."""
    client_args = build_bedrock_client_args("bearer-token-value", {"region_name": "us-west-2"})

    assert client_args["region_name"] == "us-west-2"
    client = client_args["client"]
    assert client.meta.region_name == "us-west-2"

    fake_request = _FakeRequest()
    client.meta.events.emit("before-sign.bedrock-runtime.Converse", request=fake_request)
    assert fake_request.headers["Authorization"] == "Bearer bearer-token-value"


def test_bearer_shape_builds_a_distinct_client_per_call() -> None:
    """Building a fresh client per attempt (rather than caching/sharing one)
    keeps concurrent requests for different bearer tokens from racing on
    shared state, matching otari.ai's own per-request client construction."""
    first: dict[str, Any] = build_bedrock_client_args("token-a", {"region_name": "us-east-1"})
    second: dict[str, Any] = build_bedrock_client_args("token-b", {"region_name": "us-east-1"})

    assert first["client"] is not second["client"]

    req_a, req_b = _FakeRequest(), _FakeRequest()
    first["client"].meta.events.emit("before-sign.bedrock-runtime.Converse", request=req_a)
    second["client"].meta.events.emit("before-sign.bedrock-runtime.Converse", request=req_b)
    assert req_a.headers["Authorization"] == "Bearer token-a"
    assert req_b.headers["Authorization"] == "Bearer token-b"
