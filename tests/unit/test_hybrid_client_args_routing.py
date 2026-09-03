"""Regression tests that a hybrid-mode attempt's ``extra_params`` actually
reach the provider's client constructor through the real any-llm SDK, not
just that our own code produces a kwargs dict that *looks* right.

``default_attempt_kwargs`` merges ``extra_params`` under ``client_args``
specifically because any-llm's ``acompletion()`` only forwards a
``client_args`` mapping to the provider's client constructor (everything
else in ``**kwargs`` goes to the completion *call* instead). A test that
monkeypatches ``acompletion`` directly (as the hybrid-mode integration tests
do) can't catch a regression back to flat kwargs, because the fake
``acompletion`` never exercises any-llm's own kwarg-splitting logic. These
tests call into the real SDK, stopping only at boto3's own session client
construction (patched so no network call or real AWS credentials are
needed), to verify the values actually land where boto3 expects them.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import anthropic
import httpx
import pytest
from anthropic import APIConnectionError
from any_llm import acompletion

from gateway.api.routes._platform import ResolvedAttempt, default_attempt_kwargs
from gateway.services.provider_kwargs import ANTHROPIC_DEFAULT_CONNECT_TIMEOUT_SECONDS


@pytest.mark.asyncio
async def test_bedrock_classic_shape_client_args_reach_real_boto3_client() -> None:
    """Reproduces the exact bug this test guards against: merging
    region_name/aws_access_key_id flat into acompletion()'s kwargs still
    raises botocore's NoRegionError, because any-llm forwards unrecognized
    flat kwargs to the completion call, not boto3's client constructor.
    Nesting them under client_args (what default_attempt_kwargs now does) is
    the only way they reach the dedicated Session.client() kwargs."""
    attempt = ResolvedAttempt(
        attempt_id="a0",
        position=0,
        provider="bedrock",
        model="anthropic.claude-3-5-sonnet-20241022-v2:0",
        api_key="secret-access-key",
        managed=False,
        extra_params={"region_name": "us-east-1", "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE"},
    )
    kwargs = default_attempt_kwargs(attempt, {"messages": [{"role": "user", "content": "hi"}]})

    captured: dict[str, Any] = {}

    class _StopBeforeNetworkCall(Exception):
        pass

    def fake_boto3_client(service_name: str, **client_kwargs: Any) -> Any:
        captured["service_name"] = service_name
        captured.update(client_kwargs)
        raise _StopBeforeNetworkCall

    with patch("boto3.session.Session.client", side_effect=fake_boto3_client):
        with pytest.raises(_StopBeforeNetworkCall):
            await acompletion(**kwargs)

    assert captured["service_name"] == "bedrock-runtime"
    assert captured["region_name"] == "us-east-1"
    assert captured["aws_access_key_id"] == "AKIAIOSFODNN7EXAMPLE"
    assert captured["aws_secret_access_key"] == "secret-access-key"


@pytest.mark.asyncio
async def test_bedrock_bearer_shape_uses_custom_client_not_flat_kwargs() -> None:
    """The bearer-token shape's pre-built client (client_args["client"]) is
    what any-llm's BedrockProvider actually uses; Session.client() is never
    called a second time for it."""
    attempt = ResolvedAttempt(
        attempt_id="a0",
        position=0,
        provider="bedrock",
        model="anthropic.claude-3-5-sonnet-20241022-v2:0",
        api_key="bearer-token-value",
        managed=False,
        extra_params={"region_name": "us-west-2"},
    )
    kwargs = default_attempt_kwargs(attempt, {"messages": [{"role": "user", "content": "hi"}]})

    client_args = kwargs["client_args"]
    assert "client" in client_args
    injected_client = client_args["client"]

    class _StopBeforeNetworkCall(Exception):
        pass

    def fake_converse(**_call_kwargs: Any) -> Any:
        raise _StopBeforeNetworkCall

    with patch("boto3.session.Session.client") as fake_boto3_client:
        with patch.object(injected_client, "converse", side_effect=fake_converse):
            with pytest.raises(_StopBeforeNetworkCall):
                await acompletion(**kwargs)
        # any-llm-sdk's BedrockProvider._init_client honors the pre-built
        # client and skips constructing its own; Session.client() is
        # untouched for the runtime client in this shape.
        fake_boto3_client.assert_not_called()


@pytest.mark.asyncio
async def test_anthropic_default_timeout_reaches_the_real_sdk_client() -> None:
    """otari#533: with no extra_params, the anthropic client used to be built
    with the SDK's own default timeout, which makes AsyncAnthropic's
    non-streaming pre-flight guard raise a bare ValueError for a large
    max_tokens before any request is even attempted. default_attempt_kwargs now
    carries an explicit timeout through client_args, which the real SDK's guard
    treats identically to an operator-supplied one and skips."""
    attempt = ResolvedAttempt(
        attempt_id="a0",
        position=0,
        provider="anthropic",
        model="claude-opus-5",
        api_key="sk-test",
        managed=False,
        extra_params=None,
    )
    kwargs = default_attempt_kwargs(
        attempt,
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 65536, "stream": False},
    )

    class _Sentinel(Exception):
        pass

    captured_client_timeout: list[Any] = []

    def fake_client_init(self: Any, *args: Any, **client_kwargs: Any) -> None:
        # otari#799 review, finding 1: capture the *real* httpx client's timeout
        # at construction time, before any request goes out, so this proves the
        # end-to-end value the real SDK ends up with rather than only what our
        # own dict construction says.
        real_init(self, *args, **client_kwargs)
        captured_client_timeout.append(self.timeout)

    real_init = httpx.AsyncClient.__init__

    async def fake_send(*_args: Any, **_kwargs: Any) -> Any:
        raise _Sentinel

    # any-llm/anthropic wraps a non-HTTP send failure in APIConnectionError after
    # retries; reaching that (rather than the pre-flight ValueError) is what
    # proves the guard was skipped and the request reached the transport layer.
    with patch("httpx.AsyncClient.__init__", fake_client_init):
        with patch("httpx.AsyncClient.send", side_effect=fake_send):
            with pytest.raises(APIConnectionError):
                await acompletion(**kwargs)

    assert captured_client_timeout
    assert all(t.connect == ANTHROPIC_DEFAULT_CONNECT_TIMEOUT_SECONDS for t in captured_client_timeout)
    assert all(t != anthropic.DEFAULT_TIMEOUT for t in captured_client_timeout)


@pytest.mark.asyncio
async def test_anthropic_default_timeout_reaches_the_real_sdk_client_in_hybrid_mode() -> None:
    """otari#533 point 1: an attempt with its own extra_params (the hybrid-mode
    shape) took a different branch of build_attempt_client_args than the bare
    no-extra_params case above, and that branch never filled in a default
    timeout, so a platform-routed Anthropic attempt with any extra_params at
    all (e.g. a base_url override) still hit the pre-flight guard."""
    attempt = ResolvedAttempt(
        attempt_id="a0",
        position=0,
        provider="anthropic",
        model="claude-opus-5",
        api_key="sk-test",
        managed=False,
        extra_params={"auth_token": "operator-supplied-bearer-token"},
    )
    kwargs = default_attempt_kwargs(
        attempt,
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 65536, "stream": False},
    )
    # otari#799 review, finding 1: this must not be a flat float, or the SDK
    # expands it into an httpx.Timeout that also raises connect from 5s to
    # 600s. connect must stay at the SDK's own default.
    assert kwargs["client_args"]["timeout"].connect == ANTHROPIC_DEFAULT_CONNECT_TIMEOUT_SECONDS
    assert kwargs["client_args"]["auth_token"] == "operator-supplied-bearer-token"

    class _Sentinel(Exception):
        pass

    async def fake_send(*_args: Any, **_kwargs: Any) -> Any:
        raise _Sentinel

    with patch("httpx.AsyncClient.send", side_effect=fake_send):
        with pytest.raises(APIConnectionError):
            await acompletion(**kwargs)
