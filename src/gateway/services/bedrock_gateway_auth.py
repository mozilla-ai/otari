"""Builds the ``client_args`` any-llm's Bedrock provider needs for a
hybrid-mode attempt, for both AWS credential shapes a BYO Bedrock provider
key can use.

AWS Bedrock is unusual among any-llm providers in two ways any other
provider's ``ResolvedAttempt.extra_params`` never has to deal with:

1. any-llm-sdk's ``BedrockProvider`` never forwards ``api_key`` into the boto3
   client it builds (it is only used for a presence check); the real secret
   must be passed under boto3's own constructor kwarg, ``aws_secret_access_key``.
2. AWS has two distinct credential shapes: a classic IAM access-key/secret-key
   pair, and a newer "Bedrock API key" (bearer token). The pinned boto3
   version here (see ``pyproject.toml``) has no built-in support for the
   latter (no ``AWS_BEARER_TOKEN_BEDROCK`` auto-detection, no ``aws_bearer_token``
   constructor kwarg), so it is authenticated the same way otari.ai's own
   in-process completion path does it: a boto3 client with signing disabled
   and the ``Authorization: Bearer <token>`` header injected via a
   ``before-sign`` event hook. Building that client is a per-process, in-memory
   step, so it works equally well here in the gateway as it does in a
   platform's own completion service; the bearer token itself is a plain
   string, forwarded like any other credential.

``build_bedrock_client_args`` is the single entry point: it inspects
``extra_params`` (``region_name`` and, for the classic shape,
``aws_access_key_id``) to pick the shape, and returns the dict any-llm's
``acompletion(client_args=...)`` should receive. Every other provider's
``extra_params`` is nested under ``client_args`` unchanged, without going
through this module at all (see ``default_attempt_kwargs``).
"""

from __future__ import annotations

from typing import Any

import boto3
import botocore
from botocore.config import Config

# any-llm-sdk's BedrockProvider never forwards ``api_key`` into the boto3
# client it builds (it's only used for a non-empty presence check) — the
# secret must instead be forwarded under boto3's real constructor kwarg.
# Verified against any-llm-sdk's bedrock.py: ``_init_client`` calls
# ``boto3.client("bedrock-runtime", endpoint_url=api_base, **kwargs)`` with no
# reference to ``api_key`` at all.
_SECRET_ACCESS_KEY_KWARG = "aws_secret_access_key"


def _build_unsigned_bedrock_runtime_client(region_name: str, token: str) -> Any:
    """Build a ``bedrock-runtime`` boto3 client that authenticates every
    request with ``Authorization: Bearer <token>``, regardless of the
    process's ambient AWS credentials.

    Building a client with signature-version disabled skips SigV4 credential
    resolution entirely (which would otherwise raise when no ambient AWS
    credentials are present); the ``before-sign`` hook then stamps the bearer
    header on every outgoing request. State lives on the returned client
    object, not on any global/session-wide object, so building one client per
    request is safe under concurrent load. Only the ``bedrock-runtime``
    client (used for ``converse``/``converse_stream``) is needed here: model
    listing and batch ops use a second, separate ``bedrock`` control-plane
    client any-llm-sdk builds itself, but those aren't reachable through the
    gateway's hybrid-mode chat/messages/responses routes.

    The service name is inlined as a literal at both call sites (rather than
    a shared module constant) because ``boto3.client``'s type stubs overload
    on a ``Literal`` service-name argument; a ``str``-typed constant, even
    with this exact value, would fail every overload.
    """
    client = boto3.client(
        "bedrock-runtime",
        region_name=region_name,
        config=Config(signature_version=botocore.UNSIGNED),
    )

    def _inject_bearer_token(request: Any, **kwargs: Any) -> None:  # noqa: ARG001
        request.headers["Authorization"] = f"Bearer {token}"

    client.meta.events.register("before-sign.bedrock-runtime.*", _inject_bearer_token)
    return client


def build_bedrock_client_args(api_key: str, extra_params: dict[str, str]) -> dict[str, Any]:
    """Build the ``client_args`` any-llm needs to construct a Bedrock client
    for one hybrid-mode attempt.

    ``extra_params`` always carries ``region_name`` (Bedrock's boto3 client
    has no default region fallback) and, for the classic IAM access-key-pair
    shape, ``aws_access_key_id``; its absence signals the bearer-token shape,
    mirroring the detection otari.ai's own resolve service uses.

    Classic shape: returns ``extra_params`` with the secret access key
    aliased in under its real boto3 kwarg name, so a plain ``region_name`` +
    ``aws_access_key_id`` + ``aws_secret_access_key`` dict reaches
    ``boto3.client()`` via any-llm's constructor-kwargs channel.

    Bearer-token shape: returns ``region_name`` plus a pre-built, per-request
    unsigned client (under the ``client`` key any-llm-sdk's ``BedrockProvider``
    already supports overriding its client with).
    """
    if extra_params.get("aws_access_key_id"):
        return {**extra_params, _SECRET_ACCESS_KEY_KWARG: api_key}

    region_name = extra_params.get("region_name", "")
    return {
        "region_name": region_name,
        "client": _build_unsigned_bedrock_runtime_client(region_name, api_key),
    }
