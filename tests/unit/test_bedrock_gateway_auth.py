"""Unit tests for building Bedrock's ``client_args`` for a hybrid-mode attempt."""

from __future__ import annotations

from typing import Any

import pytest
from botocore.config import Config

from gateway.services.bedrock_gateway_auth import (
    BEDROCK_READ_TIMEOUT_SECONDS,
    build_bedrock_client_args,
    with_bedrock_read_timeout,
)


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

    config = client_args.pop("config")
    assert client_args == {
        "region_name": "us-east-1",
        "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
        "aws_secret_access_key": "my-secret-access-key",
    }
    assert config.read_timeout == BEDROCK_READ_TIMEOUT_SECONDS


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


def test_bearer_shape_client_outlasts_botocores_default_read_timeout() -> None:
    """botocore's 60s default fails a stream whose model goes quiet for a minute."""
    client = build_bedrock_client_args("bearer-token-value", {"region_name": "us-west-2"})["client"]

    assert client.meta.config.read_timeout == BEDROCK_READ_TIMEOUT_SECONDS


def test_the_read_timeout_is_added_when_no_config_is_given() -> None:
    client_args = with_bedrock_read_timeout({"region_name": "us-east-1"})

    assert client_args["region_name"] == "us-east-1"
    assert client_args["config"].read_timeout == BEDROCK_READ_TIMEOUT_SECONDS


def test_an_operator_config_keeps_its_own_settings_and_gains_the_read_timeout() -> None:
    operator = Config(retries={"max_attempts": 2})

    merged = with_bedrock_read_timeout({"config": operator})["config"]

    assert merged.retries == {"max_attempts": 2}
    assert merged.read_timeout == BEDROCK_READ_TIMEOUT_SECONDS


@pytest.mark.parametrize("operator", [Config(read_timeout=30), {"read_timeout": 30}])
def test_an_operator_read_timeout_wins(operator: Config | dict[str, Any]) -> None:
    merged = with_bedrock_read_timeout({"config": operator})["config"]

    assert merged.read_timeout == 30
    assert isinstance(merged, Config)


def test_a_prebuilt_client_and_the_callers_mapping_are_left_alone() -> None:
    prebuilt = object()
    original: dict[str, Any] = {"client": prebuilt}

    assert with_bedrock_read_timeout(original) == {"client": prebuilt}
    assert with_bedrock_read_timeout({"region_name": "us-east-1"}) is not original
    no_config: dict[str, Any] = {"region_name": "us-east-1"}
    with_bedrock_read_timeout(no_config)
    assert "config" not in no_config


def test_a_config_botocore_cannot_take_is_refused() -> None:
    with pytest.raises(ValueError, match="botocore Config or a mapping"):
        with_bedrock_read_timeout({"config": 120})
