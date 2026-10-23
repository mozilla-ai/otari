"""Otari's own data-key wrappers: ``OTARI_SECRET_KEY``, and a key held in AWS KMS.

``build_content_key_port`` puts the ones a deployment configured on one keyring:
new keys are made with the wrapper ``trace_content_key_backend`` names, and a
key either wrapper made opens while that wrapper is configured, so moving a
deployment from one to the other leaves what is already stored readable.

No error raised here carries key material, a wrapped key, a context value or an
AWS error's message: an AWS failure is named by its error code alone.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import re
import secrets
from typing import TYPE_CHECKING, Any

from gateway.ports.content_key_port import ContentKeyError, ContentKeyFailure, ContentKeyPort, DataKey, DataKeySource
from gateway.services.secret_box import (
    SecretBoxUnavailableError,
    SecretDecryptionError,
    decrypt_secret,
    encrypt_secret,
    secret_box_configured,
)

if TYPE_CHECKING:
    from gateway.core.config import GatewayConfig

DATA_KEY_BYTES = 32

SECRET_KEY_WRAPPER = "sk"
KMS_WRAPPER = "kms"

# Error codes that mean this process may not use the key, as opposed to KMS being
# unreachable or busy. A disabled or deleted key is refused the same way: it will
# not open anything until an operator acts.
_KMS_DENIED = frozenset({"AccessDeniedException", "DisabledException", "NotFoundException", "KMSInvalidStateException"})
# The wrapped key is not one this key opens for this context.
_KMS_REJECTED = frozenset({"InvalidCiphertextException", "IncorrectKeyException"})


class SecretKeyContentKeys(DataKeySource):
    """Wraps a data key with ``OTARI_SECRET_KEY``, sealing its context beside it.

    Fernet has no associated data, so the context travels inside the token and
    is compared on open: a key made for one context opens for no other.
    """

    wrapper = SECRET_KEY_WRAPPER

    async def new_data_key(self, context: dict[str, str]) -> DataKey:
        plaintext = secrets.token_bytes(DATA_KEY_BYTES)
        body = json.dumps({"k": base64.b64encode(plaintext).decode(), "c": context}, sort_keys=True)
        try:
            token = encrypt_secret(body)
        except SecretBoxUnavailableError:
            raise ContentKeyError("unavailable", "OTARI_SECRET_KEY is not set") from None
        return DataKey(wrapper=self.wrapper, plaintext=plaintext, wrapped=token.encode())

    async def open_data_key(self, wrapped: bytes, context: dict[str, str]) -> bytes:
        try:
            body = json.loads(decrypt_secret(wrapped.decode()))
        except SecretBoxUnavailableError:
            raise ContentKeyError("unavailable", "OTARI_SECRET_KEY is not set") from None
        except (SecretDecryptionError, UnicodeDecodeError, ValueError):
            raise ContentKeyError("rejected", "no configured OTARI_SECRET_KEY opens this data key") from None
        if not isinstance(body, dict) or body.get("c") != context or not isinstance(body.get("k"), str):
            raise ContentKeyError("rejected", "the data key was made for another context")
        try:
            plaintext = base64.b64decode(body["k"], validate=True)
        except (binascii.Error, ValueError):
            raise ContentKeyError("rejected", "the data key is malformed") from None
        if len(plaintext) != DATA_KEY_BYTES:
            raise ContentKeyError("rejected", "the data key is malformed")
        return plaintext


class AwsKmsContentKeys(DataKeySource):
    """Makes data keys under a customer-managed AWS KMS key, with the context as encryption context.

    New keys are made under ``key_id``. A wrapped key opens only when KMS reports
    it was made under ``key_id`` or one of ``previous_key_ids``, so replacing the
    key keeps what the old one wrapped readable while it stays listed, and a
    ciphertext made under any other key opens as nothing. Each id may be a key
    id, a key ARN or an alias; they are resolved to key ARNs on first use.
    """

    wrapper = KMS_WRAPPER

    def __init__(
        self,
        key_id: str,
        previous_key_ids: tuple[str, ...] = (),
        *,
        region: str | None = None,
        endpoint_url: str | None = None,
        client: Any = None,
    ) -> None:
        if client is None:
            try:
                import boto3
                from botocore.config import Config
            except ImportError as exc:
                msg = "trace_content_key_backend 'aws_kms' requires boto3. Install the kms extra: uv sync --extra kms"
                raise ImportError(msg) from exc
            # Tight timeouts and few retries: a KMS outage drops content, and must
            # not hold the trace writer's batch past its own time limit.
            config = Config(connect_timeout=2, read_timeout=3, retries={"max_attempts": 2, "mode": "standard"})
            client = boto3.client("kms", region_name=region, endpoint_url=endpoint_url, config=config)
        self._client = client
        self._key_id = key_id
        self._key_ids = (key_id, *previous_key_ids)
        self._allowed_arns: frozenset[str] | None = None

    async def new_data_key(self, context: dict[str, str]) -> DataKey:
        response = await self._call(
            "GenerateDataKey",
            self._client.generate_data_key,
            KeyId=self._key_id,
            KeySpec="AES_256",
            EncryptionContext=context,
        )
        return DataKey(wrapper=self.wrapper, plaintext=response["Plaintext"], wrapped=response["CiphertextBlob"])

    async def open_data_key(self, wrapped: bytes, context: dict[str, str]) -> bytes:
        allowed = await self._allowed()
        response = await self._call("Decrypt", self._client.decrypt, CiphertextBlob=wrapped, EncryptionContext=context)
        if response.get("KeyId") not in allowed:
            raise ContentKeyError("rejected", "the data key was made under a KMS key this deployment does not list")
        plaintext: bytes = response["Plaintext"]
        return plaintext

    async def _allowed(self) -> frozenset[str]:
        if self._allowed_arns is None:
            arns = []
            for key_id in self._key_ids:
                described = await self._call("DescribeKey", self._client.describe_key, KeyId=key_id)
                arns.append(described["KeyMetadata"]["Arn"])
            self._allowed_arns = frozenset(arns)
        return self._allowed_arns

    @staticmethod
    async def _call(operation: str, method: Any, **kwargs: Any) -> Any:
        from botocore.exceptions import BotoCoreError, ClientError

        try:
            return await asyncio.to_thread(method, **kwargs)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", "unknown"))
            reason: ContentKeyFailure = (
                "denied" if code in _KMS_DENIED else "rejected" if code in _KMS_REJECTED else "unavailable"
            )
            raise ContentKeyError(reason, f"AWS KMS {operation} failed ({code})") from None
        except BotoCoreError as exc:
            raise ContentKeyError("unavailable", f"AWS KMS {operation} failed ({type(exc).__name__})") from None


class ContentKeyring(ContentKeyPort):
    """The :class:`ContentKeyPort` over the wrappers a deployment configured."""

    def __init__(self, writer: str, wrappers: dict[str, DataKeySource]) -> None:
        self._writer = writer
        self._wrappers = wrappers

    @property
    def wrapper(self) -> str | None:
        # Asked each time, so OTARI_SECRET_KEY set or unset after startup is what decides.
        if self._writer == SECRET_KEY_WRAPPER and not secret_box_configured():
            return None
        return self._writer

    async def new_data_key(self, context: dict[str, str]) -> DataKey:
        writer = self.wrapper
        if writer is None:
            raise ContentKeyError("unavailable", "no content key is configured")
        return await self._wrappers[writer].new_data_key(context)

    async def open_data_key(self, wrapper: str, wrapped: bytes, context: dict[str, str]) -> bytes:
        source = self._wrappers.get(wrapper)
        if source is None:
            raise ContentKeyError("unavailable", "the wrapper this data key names is not configured")
        return await source.open_data_key(wrapped, context)


def _key_ids(raw: str | None) -> tuple[str, ...]:
    return tuple(key_id for key_id in re.split(r"[\s,]+", (raw or "").strip()) if key_id)


def build_content_key_port(config: GatewayConfig) -> ContentKeyPort:
    """Build the keyring ``trace_content_key_backend`` and the KMS settings describe.

    ``OTARI_SECRET_KEY`` always opens what it wrapped. A KMS key, where one is
    set, opens what it wrapped even while new keys are made with the secret key,
    so a deployment can move back as well as forward.

    Raises:
        ValueError: ``aws_kms`` is selected without ``trace_content_kms_key_id``.
        ImportError: a KMS key is configured and boto3 is not installed.
    """
    wrappers: dict[str, DataKeySource] = {SECRET_KEY_WRAPPER: SecretKeyContentKeys()}
    key_id = config.trace_content_kms_key_id
    if config.trace_content_key_backend == "aws_kms" and not key_id:
        msg = "trace_content_kms_key_id is required when trace_content_key_backend is 'aws_kms'"
        raise ValueError(msg)
    if key_id:
        wrappers[KMS_WRAPPER] = AwsKmsContentKeys(
            key_id,
            _key_ids(config.trace_content_kms_previous_key_ids),
            region=config.trace_content_kms_region,
            endpoint_url=config.trace_content_kms_endpoint_url,
        )
    writer = KMS_WRAPPER if config.trace_content_key_backend == "aws_kms" else SECRET_KEY_WRAPPER
    return ContentKeyring(writer, wrappers)
