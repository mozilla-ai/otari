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
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from time import monotonic
from typing import TYPE_CHECKING, Any
from weakref import WeakKeyDictionary

from gateway.log_config import logger
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
# How long the key ARNs a deployment's ids resolve to are trusted before they are resolved again.
_ALLOWED_KEYS_TTL_S = 300.0
_PARTIAL_KEYS_TTL_S = 15.0
# KMS calls in flight at once, and the client's connections for them.
_KMS_CONCURRENCY = 32

SECRET_KEY_WRAPPER = "sk"
KMS_WRAPPER = "kms"

# Error codes that mean this process may not use the key, as opposed to KMS being
# unreachable or busy. A disabled or deleted key is refused the same way: it will
# not open anything until an operator acts.
_KMS_DENIED = frozenset({"AccessDeniedException", "DisabledException", "NotFoundException", "KMSInvalidStateException"})
# The wrapped key is not one this key opens for this context.
_KMS_REJECTED = frozenset({"InvalidCiphertextException", "IncorrectKeyException"})


# One resolution of the keys content may open under: the ARNs, when they were
# resolved, and whether every configured id answered. Read as one value, so a read
# classifying a miss cannot take the flag from a later resolution than the set.
_Allowed = tuple[frozenset[str], float, bool]


def _stale(allowed: _Allowed) -> bool:
    """Whether a resolution is old enough to make again.

    A partial set is trusted only briefly, so one throttled DescribeKey does not
    shut a key out for minutes.
    """
    _, at, complete = allowed
    return monotonic() - at >= (_ALLOWED_KEYS_TTL_S if complete else _PARTIAL_KEYS_TTL_S)


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

    def close(self) -> None:
        """Nothing: the secret key holds nothing open."""


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
                from botocore.exceptions import BotoCoreError
            except ImportError as exc:
                msg = "trace_content_key_backend 'aws_kms' requires boto3. Install the kms extra: uv sync --extra kms"
                raise ImportError(msg) from exc
            # Tight timeouts and few retries: a KMS outage drops content, and must
            # not hold the trace writer's batch past its own time limit.
            # Enough connections for a batch fetching many sessions' keys at once.
            config = Config(
                connect_timeout=1,
                read_timeout=1,
                retries={"max_attempts": 2, "mode": "standard"},
                max_pool_connections=_KMS_CONCURRENCY,
            )
            try:
                client = boto3.client("kms", region_name=region, endpoint_url=endpoint_url, config=config)
            except BotoCoreError as exc:
                # A missing region, most often: a configuration error, reported as one.
                msg = (
                    f"the trace content KMS client cannot be built ({type(exc).__name__}); set trace_content_kms_region"
                )
                raise ValueError(msg) from None
        self._client = client
        self._executor: ThreadPoolExecutor | None = None
        self._key_id = key_id
        # Deduplicated, keeping the order given: a key listed twice would be
        # described twice in the one round trip below.
        self._key_ids = tuple(dict.fromkeys((key_id, *previous_key_ids)))
        self._allowed: _Allowed | None = None
        # One per loop: a lock binds to the loop that first waits on it, and this
        # adapter outlives any one loop. Weakly keyed, so a loop that is gone takes
        # its lock with it.
        self._resolving: WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = WeakKeyDictionary()

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
        # Neither call needs the other's answer, so both go at once: a read costs one
        # round trip however many keys the deployment lists, which is what keeps it
        # inside the deadline its caller allows for opening one key.
        answers = await asyncio.gather(
            self._allowed_keys(),
            self._call("Decrypt", self._client.decrypt, CiphertextBlob=wrapped, EncryptionContext=context),
            return_exceptions=True,
        )
        allowed, response = answers
        if isinstance(response, BaseException):
            # The ciphertext's own refusal outranks a key that could not be described:
            # it says this row does not open, where the other says not yet.
            raise response
        if isinstance(allowed, BaseException):
            raise allowed
        arns, _, complete = allowed
        if response.get("KeyId") not in arns:
            # ``complete`` comes from the same resolution as the ARNs, because a refresh
            # running beside this read would otherwise decide which answer this is.
            if not complete:
                # It may be under the key that could not be described: an outage, not a tampered row.
                raise ContentKeyError("unavailable", "the KMS key this data key was made under could not be described")
            raise ContentKeyError("rejected", "the data key was made under a KMS key this deployment does not list")
        plaintext: bytes = response["Plaintext"]
        return plaintext

    def close(self) -> None:
        """Stop the pool KMS calls run on, abandoning any still queued; the next call starts a new one."""
        executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    async def _allowed_keys(self) -> _Allowed:
        """The keys content may open under, resolved from the configured ids every few minutes.

        One resolution serves every reader waiting on it: a read happens on the
        request path, so readers arrive together and each starting its own would
        ask the key source once per reader for the same answer.
        """
        held = self._allowed
        if held is not None and not _stale(held):
            return held
        async with self._resolution_lock():
            # Resolved while this read waited for the lock, which is the usual case
            # for every reader but the first.
            held = self._allowed
            if held is not None and not _stale(held):
                return held
            self._allowed = await self._describe_keys()
            return self._allowed

    def _resolution_lock(self) -> asyncio.Lock:
        """The lock readers of this loop queue on while the key set is resolved."""
        loop = asyncio.get_running_loop()
        lock = self._resolving.get(loop)
        if lock is None:
            lock = self._resolving[loop] = asyncio.Lock()
        return lock

    async def _describe_keys(self) -> _Allowed:
        """Describe every configured id at once, keeping the answers that came back.

        Each answer stands on its own, so a listed key that is gone or not
        describable costs only what it sealed, an alias pointed at a new key is
        followed without a restart, and the resolution takes one round trip rather
        than one per key.
        """
        described = await asyncio.gather(
            *(self._call("DescribeKey", self._client.describe_key, KeyId=key_id) for key_id in self._key_ids),
            return_exceptions=True,
        )
        arns: set[str] = set()
        failure: ContentKeyError | None = None
        for answer in described:
            if isinstance(answer, ContentKeyError):
                failure = failure or answer
            elif isinstance(answer, BaseException):
                raise answer
            else:
                arns.add(answer["KeyMetadata"]["Arn"])
        if not arns:
            raise failure or ContentKeyError("unavailable", "no configured KMS key could be described")
        if failure is not None:
            logger.warning("A configured trace content KMS key could not be described and is skipped (%s)", failure)
        return frozenset(arns), monotonic(), failure is None

    async def _call(self, operation: str, method: Any, **kwargs: Any) -> Any:
        from botocore.exceptions import BotoCoreError, ClientError

        try:
            # A pool of its own, as wide as the client's connections, so a batch fetching many
            # sessions' keys is not queued behind the process's small default executor.
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=_KMS_CONCURRENCY, thread_name_prefix="otari-kms")
            return await asyncio.get_running_loop().run_in_executor(self._executor, partial(method, **kwargs))
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
            # The row's fault, not the source's: it names a wrapper this deployment no longer configures.
            raise ContentKeyError("rejected", "the wrapper this data key names is not configured")
        return await source.open_data_key(wrapped, context)

    def close(self) -> None:
        for source in self._wrappers.values():
            source.close()


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
