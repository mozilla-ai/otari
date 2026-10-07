"""The core DataKeyPort adapters: a KEK from ``OTARI_SECRET_KEY``, and one held in AWS KMS.

``SecretBoxDataKeys`` wraps each data key with the deployment's Fernet key, the
one that already encrypts stored provider credentials. Fernet has no associated
data, so the context is sealed inside the wrapped payload and checked on unwrap.

``AwsKmsDataKeys`` asks KMS for each data key, with the context as the KMS
encryption context, so KMS itself refuses to decrypt it under another one. This
build seals and reads content in the same process, so that process needs both
``kms:GenerateDataKey`` and ``kms:Decrypt`` on the key. boto3 is imported only
when this adapter is built, behind the ``kms`` extra, so the plain build runs
without it.
"""

import asyncio
import json
import os
import uuid
from typing import Any

from cryptography.fernet import InvalidToken

from gateway.log_config import logger
from gateway.ports.data_key_port import (
    DataKey,
    DataKeyContext,
    DataKeyContextMismatchError,
    DataKeyPort,
    DataKeyUnavailableError,
)
from gateway.services.secret_box import SecretBoxUnavailableError, get_secret_box

_KEY_BYTES = 32


class SecretBoxDataKeys(DataKeyPort):
    """Data keys wrapped by ``OTARI_SECRET_KEY``.

    The KEK is read when a key is generated or unwrapped, not when the adapter is
    built, so a deployment that never captures content needs no secret key.
    """

    key_ref = "secret_box"

    @staticmethod
    def _box() -> Any:
        try:
            return get_secret_box()
        except SecretBoxUnavailableError as exc:
            msg = "Set OTARI_SECRET_KEY to a valid Fernet key to capture trace content"
            raise DataKeyUnavailableError(msg) from exc

    async def available(self) -> bool:
        try:
            self._box()
        except DataKeyUnavailableError as exc:
            logger.warning("Trace content key backend unavailable: %s", exc)
            return False
        return True

    async def generate(self, context: DataKeyContext) -> DataKey:
        plaintext = os.urandom(_KEY_BYTES)
        payload = json.dumps({"context": context.as_mapping(), "key": plaintext.hex()}).encode()
        return DataKey(plaintext=plaintext, wrapped=self._box().encrypt(payload), key_ref=self.key_ref)

    async def unwrap(self, wrapped: bytes, context: DataKeyContext) -> bytes:
        box = self._box()
        try:
            payload = json.loads(box.decrypt(wrapped))
            bound = payload["context"]
            key = bytes.fromhex(payload["key"])
        except (InvalidToken, ValueError, KeyError, TypeError) as exc:
            raise DataKeyContextMismatchError("The data key does not unwrap with this deployment's key") from exc
        if bound != context.as_mapping():
            raise DataKeyContextMismatchError("The data key belongs to another workspace or session")
        return key


class AwsKmsDataKeys(DataKeyPort):
    """Data keys generated and decrypted by one AWS KMS key.

    ``client`` is a boto3 KMS client, or a stand-in with the same two methods;
    left out, one is built from the environment's AWS credentials. Calls run in a
    worker thread, because boto3 blocks.
    """

    def __init__(self, key_id: str, *, client: Any | None = None) -> None:
        if not key_id:
            raise DataKeyUnavailableError("Set trace_content_kms_key_id to use the aws_kms key backend")
        self._key_id = key_id
        self._client = client if client is not None else _kms_client()

    @property
    def key_ref(self) -> str:
        return f"aws_kms:{self._key_id}"

    async def available(self) -> bool:
        # A throwaway key, discarded: the one call that proves the credentials, the key
        # and this process's permission to generate with it, and no permission beyond that.
        probe = DataKeyContext(workspace_id=uuid.UUID(int=0), session="availability-probe")
        try:
            await self.generate(probe)
        except Exception as exc:
            logger.warning("Trace content key backend unavailable: %s", type(exc).__name__)
            return False
        return True

    async def generate(self, context: DataKeyContext) -> DataKey:
        try:
            response = await asyncio.to_thread(
                self._client.generate_data_key,
                KeyId=self._key_id,
                KeySpec="AES_256",
                EncryptionContext=context.as_mapping(),
            )
        except Exception as exc:
            raise _unavailable_from(exc) from None
        return DataKey(plaintext=response["Plaintext"], wrapped=response["CiphertextBlob"], key_ref=self.key_ref)

    async def unwrap(self, wrapped: bytes, context: DataKeyContext) -> bytes:
        try:
            response = await asyncio.to_thread(
                self._client.decrypt,
                CiphertextBlob=wrapped,
                KeyId=self._key_id,
                EncryptionContext=context.as_mapping(),
            )
        except Exception as exc:
            # KMS answers a wrong encryption context with InvalidCiphertextException, and
            # its client errors are botocore types this module does not import.
            if type(exc).__name__ in {"InvalidCiphertextException", "IncorrectKeyException"}:
                raise DataKeyContextMismatchError("The data key belongs to another workspace or session") from exc
            raise _unavailable_from(exc) from None
        plaintext: bytes = response["Plaintext"]
        return plaintext


def _unavailable_from(exc: BaseException) -> DataKeyUnavailableError:
    """A KMS failure as the port's own error: its type names the cause, its text could name the account."""
    return DataKeyUnavailableError(f"AWS KMS refused or could not be reached ({type(exc).__name__})")


def _kms_client() -> Any:
    try:
        import boto3  # noqa: PLC0415 - optional extra, imported only when this backend is chosen
    except ImportError as exc:
        msg = "The aws_kms key backend needs the kms extra: install otari[kms]"
        raise DataKeyUnavailableError(msg) from exc
    return boto3.client("kms")


def build_data_key_port(backend: str, kms_key_id: str | None) -> DataKeyPort:
    """The adapter a deployment's ``trace_content_key_backend`` names."""
    if backend == "aws_kms":
        return AwsKmsDataKeys(kms_key_id or "")
    return SecretBoxDataKeys()
