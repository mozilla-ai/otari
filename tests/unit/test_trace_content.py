"""A span's content is sealed under a data key for its row, and opens only there.

KMS is moto's, or a fake client for the failures moto does not model; never real AWS.
"""

import asyncio
import base64
import json
import logging
import time
import uuid
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, cast

import boto3
import pytest
from botocore.exceptions import ClientError, EndpointConnectionError
from moto import mock_aws

from gateway.adapters.content_key_adapter import (
    KMS_WRAPPER,
    SECRET_KEY_WRAPPER,
    AwsKmsContentKeys,
    ContentKeyring,
    SecretKeyContentKeys,
    build_content_key_port,
)
from gateway.core.config import GatewayConfig
from gateway.ports.content_key_port import ContentKeyError, DataKey
from gateway.services.secret_box import encrypt_secret, generate_secret_key
from gateway.services.traces import _content
from gateway.services.traces._content import UNOPENED, UNSEALED, SealItem, SpanContent

pytestmark = pytest.mark.asyncio

_WORKSPACE = uuid.UUID("00000000-0000-0000-0000-000000000001")
_OTHER_WORKSPACE = uuid.UUID("00000000-0000-0000-0000-000000000002")
_CONTENT = {"command": "cat ~/.aws/credentials"}
_REGION = "us-east-1"
_TRACE = "session-1"
_CONTEXT = {"otari:purpose": "trace-content", "otari:workspace": str(_WORKSPACE), "otari:session": _TRACE}


@pytest.fixture
def key(monkeypatch: pytest.MonkeyPatch) -> str:
    value = generate_secret_key()
    monkeypatch.setenv("OTARI_SECRET_KEY", value)
    return value


@pytest.fixture
def kms(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    monkeypatch.setenv("AWS_DEFAULT_REGION", _REGION)
    # moto answers in this process, so a busy CI runner can hold a call past the
    # production budget and drop the content a test then expects to read back.
    # The budget itself is pinned against a fake key source below.
    monkeypatch.setattr(_content, "_KEY_SOURCE_BUDGET_S", 30.0)
    with mock_aws():
        yield boto3.client("kms", region_name=_REGION)


def _secret_key_content() -> SpanContent:
    return SpanContent(ContentKeyring(SECRET_KEY_WRAPPER, {SECRET_KEY_WRAPPER: SecretKeyContentKeys()}))


def _kms_keyring(client: Any, key_id: str, previous: tuple[str, ...] = ()) -> ContentKeyring:
    return ContentKeyring(
        KMS_WRAPPER,
        {SECRET_KEY_WRAPPER: SecretKeyContentKeys(), KMS_WRAPPER: AwsKmsContentKeys(key_id, previous, client=client)},
    )


def _new_kms_key(client: Any) -> str:
    key_id: str = client.create_key()["KeyMetadata"]["KeyId"]
    return key_id


def _wrapped(sealed: str | None) -> bytes:
    assert sealed is not None
    part = sealed.split(".")[2]
    return base64.urlsafe_b64decode(part + "=" * (-len(part) % 4))


def _count(counter: Any, reason: str) -> float:
    value: float = counter.labels(reason=reason)._value.get()
    return value


class _CountingKeys:
    """A key source that counts its calls, over the secret-key wrapper."""

    wrapper = SECRET_KEY_WRAPPER

    def __init__(self) -> None:
        self._inner = SecretKeyContentKeys()
        self.made = 0
        self.opened = 0

    async def new_data_key(self, context: dict[str, str]) -> DataKey:
        self.made += 1
        return await self._inner.new_data_key(context)

    async def open_data_key(self, wrapper: str, wrapped: bytes, context: dict[str, str]) -> bytes:
        self.opened += 1
        return await self._inner.open_data_key(wrapped, context)

    def close(self) -> None:
        self.closed = True


class _FailingKmsClient:
    """A KMS client every call to which fails with ``error``."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    def _fail(self, **_: Any) -> Any:
        self.calls += 1
        raise self.error

    generate_data_key = decrypt = describe_key = _fail


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "arn:aws:kms:secret-detail"}}, "Decrypt")


# Sealing under OTARI_SECRET_KEY, the default.


async def test_sealed_content_hides_what_it_holds_and_opens_on_its_row(key: str) -> None:
    content = _secret_key_content()
    sealed = await content.seal(_CONTENT, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1")

    assert sealed is not None and sealed.startswith("otc1.sk.") and "credentials" not in sealed
    assert await content.open(sealed, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1") == _CONTENT


@pytest.mark.parametrize(
    ("workspace_id", "trace_id", "span_id"),
    [(_WORKSPACE, _TRACE, "tool-2"), (_WORKSPACE, "session-2", "tool-1"), (_OTHER_WORKSPACE, _TRACE, "tool-1")],
)
async def test_content_copied_onto_another_row_opens_as_nothing(
    key: str, workspace_id: uuid.UUID, trace_id: str, span_id: str
) -> None:
    sealed = await _secret_key_content().seal(_CONTENT, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1")

    assert await _secret_key_content().open(sealed, workspace_id=workspace_id, trace_id=trace_id, span_id=span_id) == {}


async def test_a_tampered_value_opens_as_nothing(key: str) -> None:
    content = _secret_key_content()
    sealed = await content.seal(_CONTENT, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1")
    assert sealed is not None
    head, body = sealed.rsplit(".", 1)
    flipped = body[:10] + ("A" if body[10] != "A" else "B") + body[11:]

    assert await content.open(f"{head}.{flipped}", workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1") == {}
    assert await content.open("otc1.sk.not-base64!", workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1") == {}


async def test_without_a_key_nothing_is_sealed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTARI_SECRET_KEY", raising=False)

    assert (
        await _secret_key_content().seal(_CONTENT, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1") is None
    )


async def test_content_opens_under_a_rotated_key_list_and_not_once_its_key_is_gone(
    key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    sealed = await _secret_key_content().seal(_CONTENT, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1")
    newer = generate_secret_key()

    monkeypatch.setenv("OTARI_SECRET_KEY", f"{newer},{key}")
    assert (
        await _secret_key_content().open(sealed, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1") == _CONTENT
    )
    monkeypatch.setenv("OTARI_SECRET_KEY", newer)
    assert await _secret_key_content().open(sealed, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1") == {}


async def test_no_content_seals_to_nothing(key: str) -> None:
    content = _secret_key_content()

    assert await content.seal({}, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1") is None
    assert await content.open(None, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1") == {}


async def test_content_disabled_seals_nothing(key: str) -> None:
    assert (
        await SpanContent.disabled().seal(_CONTENT, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1") is None
    )


# Rows sealed before data keys.


def _legacy(workspace_id: uuid.UUID, span_id: str) -> str:
    return encrypt_secret(json.dumps({"workspace_id": str(workspace_id), "span_id": span_id, "content": _CONTENT}))


async def test_an_old_fernet_row_still_opens_on_its_row_and_nowhere_else(key: str) -> None:
    legacy = _legacy(_WORKSPACE, "tool-1")
    content = _secret_key_content()

    assert await content.open(legacy, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1") == _CONTENT
    assert await content.open(legacy, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-2") == {}
    assert await content.open(legacy, workspace_id=_OTHER_WORKSPACE, trace_id=_TRACE, span_id="tool-1") == {}


async def test_old_rows_open_after_the_deployment_moves_to_kms(key: str, kms: Any) -> None:
    legacy = _legacy(_WORKSPACE, "tool-1")
    sealed_with_secret_key = await _secret_key_content().seal(
        _CONTENT, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-2"
    )
    content = SpanContent(_kms_keyring(kms, _new_kms_key(kms)))

    new = await content.seal(_CONTENT, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-3")

    assert new is not None and new.startswith("otc1.kms.")
    assert await content.open(legacy, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1") == _CONTENT
    assert (
        await content.open(sealed_with_secret_key, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-2")
        == _CONTENT
    )
    assert await content.open(new, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-3") == _CONTENT


# Data-key caching.


async def test_a_busy_workspace_makes_one_data_key_and_opens_it_once(key: str) -> None:
    keys = _CountingKeys()
    content = SpanContent(keys)

    sealed = [
        await content.seal(_CONTENT, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id=f"tool-{n}") for n in range(500)
    ]
    await content.seal(_CONTENT, workspace_id=_OTHER_WORKSPACE, trace_id=_TRACE, span_id="tool-1")
    reader = SpanContent(keys)
    for n, value in enumerate(sealed):
        assert await reader.open(value, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id=f"tool-{n}") == _CONTENT

    assert keys.made == 2
    assert keys.opened == 1


async def test_a_write_key_is_replaced_once_old_or_used_enough(key: str, monkeypatch: pytest.MonkeyPatch) -> None:
    keys = _CountingKeys()
    content = SpanContent(keys)
    now = [1_000.0]
    monkeypatch.setattr(_content, "monotonic", lambda: now[0])
    monkeypatch.setattr(_content, "_WRITE_KEY_MAX_USES", 3)

    for n in range(3):
        await content.seal(_CONTENT, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id=f"tool-{n}")
    assert keys.made == 1
    await content.seal(_CONTENT, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-3")
    assert keys.made == 2

    now[0] += _content._KEY_MAX_AGE_S
    await content.seal(_CONTENT, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-4")
    assert keys.made == 3


# AWS KMS.


async def test_kms_seals_under_the_workspace_as_encryption_context(kms: Any) -> None:
    key_id = _new_kms_key(kms)
    sealed = await SpanContent(_kms_keyring(kms, key_id)).seal(
        _CONTENT, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1"
    )

    assert kms.decrypt(CiphertextBlob=_wrapped(sealed), EncryptionContext=_CONTEXT)["KeyId"].endswith(key_id)
    reader = SpanContent(_kms_keyring(kms, key_id))
    assert await reader.open(sealed, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1") == _CONTENT


async def test_kms_content_moved_to_another_workspace_is_refused_by_kms(kms: Any) -> None:
    key_id = _new_kms_key(kms)
    sealed = await SpanContent(_kms_keyring(kms, key_id)).seal(
        _CONTENT, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1"
    )
    before = _count(UNOPENED, "rejected")

    reader = SpanContent(_kms_keyring(kms, key_id))
    assert await reader.open(sealed, workspace_id=_OTHER_WORKSPACE, trace_id=_TRACE, span_id="tool-1") == {}
    assert _count(UNOPENED, "rejected") == before + 1
    # A rejected key is the row's fault, not the source's, so the next read still asks KMS.
    assert await reader.open(sealed, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1") == _CONTENT


async def test_kms_key_replaced_keeps_old_content_open_while_listed(kms: Any) -> None:
    old, new = _new_kms_key(kms), _new_kms_key(kms)
    kms.create_alias(AliasName="alias/otari-trace-content", TargetKeyId=old)
    sealed = await SpanContent(_kms_keyring(kms, "alias/otari-trace-content")).seal(
        _CONTENT, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1"
    )
    listed = SpanContent(_kms_keyring(kms, new, previous=("alias/otari-trace-content",)))
    unlisted = SpanContent(_kms_keyring(kms, new))

    assert await listed.open(sealed, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1") == _CONTENT
    assert await unlisted.open(sealed, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1") == {}
    resealed = await listed.seal(_CONTENT, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-2")
    assert kms.decrypt(CiphertextBlob=_wrapped(resealed), EncryptionContext=_CONTEXT)["KeyId"].endswith(new)


async def test_kms_automatic_rotation_leaves_content_readable(kms: Any) -> None:
    key_id = _new_kms_key(kms)
    sealed = await SpanContent(_kms_keyring(kms, key_id)).seal(
        _CONTENT, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1"
    )

    kms.enable_key_rotation(KeyId=key_id)
    kms.rotate_key_on_demand(KeyId=key_id)

    reader = SpanContent(_kms_keyring(kms, key_id))
    assert await reader.open(sealed, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1") == _CONTENT


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (EndpointConnectionError(endpoint_url="https://kms.us-east-1.amazonaws.com"), "unavailable"),
        (_client_error("ThrottlingException"), "unavailable"),
        (_client_error("AccessDeniedException"), "denied"),
        (_client_error("DisabledException"), "denied"),
    ],
)
async def test_kms_failing_stores_no_content_and_pauses_before_asking_again(
    error: Exception, reason: str, caplog: pytest.LogCaptureFixture
) -> None:
    client = _FailingKmsClient(error)
    content = SpanContent(_kms_keyring(client, "alias/otari-trace-content"))
    before, paused = _count(UNSEALED, reason), _count(UNSEALED, "paused")

    with caplog.at_level(logging.WARNING):
        assert await content.seal(_CONTENT, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1") is None
        assert await content.seal(_CONTENT, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-2") is None

    assert client.calls == 1
    assert _count(UNSEALED, reason) == before + 1
    assert _count(UNSEALED, "paused") == paused + 1
    assert "credentials" not in caplog.text and "secret-detail" not in caplog.text


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (EndpointConnectionError(endpoint_url="https://kms.us-east-1.amazonaws.com"), "unavailable"),
        (_client_error("AccessDeniedException"), "denied"),
    ],
)
async def test_kms_failing_on_read_returns_empty_content(kms: Any, error: Exception, reason: str) -> None:
    sealed = await SpanContent(_kms_keyring(kms, _new_kms_key(kms))).seal(
        _CONTENT, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1"
    )
    before = _count(UNOPENED, reason)

    content = SpanContent(_kms_keyring(_FailingKmsClient(error), "alias/otari-trace-content"))

    assert await content.open(sealed, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1") == {}
    assert _count(UNOPENED, reason) == before + 1


async def test_kms_errors_name_the_code_and_never_the_message() -> None:
    keys = AwsKmsContentKeys("alias/x", client=_FailingKmsClient(_client_error("AccessDeniedException")))

    with pytest.raises(ContentKeyError) as raised:
        await keys.new_data_key(_CONTEXT)

    assert raised.value.reason == "denied"
    assert "AccessDeniedException" in str(raised.value)
    assert "secret-detail" not in str(raised.value) and str(_WORKSPACE) not in str(raised.value)
    assert raised.value.__cause__ is None


async def test_a_data_key_never_shows_its_key_material() -> None:
    data_key = DataKey(wrapper="sk", plaintext=b"\x01" * 32, wrapped=b"wrapped-bytes")

    assert repr(data_key) == "DataKey(wrapper='sk')"


# Configuration.


def _config(**values: Any) -> GatewayConfig:
    settings = {
        "trace_content_key_backend": "secret_key",
        "trace_content_kms_key_id": None,
        "trace_content_kms_previous_key_ids": None,
        "trace_content_kms_region": _REGION,
        "trace_content_kms_endpoint_url": None,
        **values,
    }
    return cast(GatewayConfig, SimpleNamespace(**settings))


async def test_aws_kms_without_a_key_id_refuses_to_build() -> None:
    with pytest.raises(ValueError, match="trace_content_kms_key_id"):
        build_content_key_port(_config(trace_content_key_backend="aws_kms"))


async def test_the_secret_key_backend_writes_only_while_a_key_is_set(key: str, monkeypatch: pytest.MonkeyPatch) -> None:
    keys = build_content_key_port(_config())
    assert keys.wrapper == SECRET_KEY_WRAPPER

    monkeypatch.delenv("OTARI_SECRET_KEY")
    assert keys.wrapper is None


async def test_a_kms_key_beside_the_secret_key_backend_still_opens_what_it_sealed(key: str, kms: Any) -> None:
    key_id = _new_kms_key(kms)
    sealed = await SpanContent(_kms_keyring(kms, key_id)).seal(
        _CONTENT, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1"
    )

    keys = build_content_key_port(_config(trace_content_kms_key_id=key_id))

    assert keys.wrapper == SECRET_KEY_WRAPPER
    assert await SpanContent(keys).open(sealed, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1") == _CONTENT


async def test_kms_only_without_a_key_id_refuses_to_build() -> None:
    with pytest.raises(ValueError, match="trace_content_kms_key_id"):
        build_content_key_port(_config(trace_content_key_backend="aws_kms_only"))


async def test_kms_only_opens_nothing_the_secret_key_sealed(key: str, kms: Any) -> None:
    key_id = _new_kms_key(kms)
    by_secret_key = await _secret_key_content().seal(_CONTENT, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="t1")
    by_kms = await SpanContent(_kms_keyring(kms, key_id)).seal(
        _CONTENT, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="t2"
    )

    keys = build_content_key_port(_config(trace_content_key_backend="aws_kms_only", trace_content_kms_key_id=key_id))
    content = SpanContent(keys)

    assert keys.wrapper == KMS_WRAPPER
    assert await content.open(by_secret_key, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="t1") == {}
    assert await content.open(by_kms, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="t2") == _CONTENT


async def test_previous_key_ids_accept_commas_and_whitespace(kms: Any) -> None:
    keys = build_content_key_port(
        _config(
            trace_content_key_backend="aws_kms",
            trace_content_kms_key_id="k1",
            trace_content_kms_previous_key_ids="k2, k3\nk4",
        )
    )

    assert keys.wrapper == KMS_WRAPPER
    assert cast(Any, keys)._wrappers[KMS_WRAPPER]._key_ids == ("k1", "k2", "k3", "k4")


# Outages, slowness and idle keys.


class _FlakyKeys(_CountingKeys):
    """A key source that fails while ``down``, and can be made slow."""

    def __init__(self) -> None:
        super().__init__()
        self.down = False
        self.delay = 0.0

    async def new_data_key(self, context: dict[str, str]) -> DataKey:
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.down:
            raise ContentKeyError("unavailable", "down")
        return await super().new_data_key(context)

    async def open_data_key(self, wrapper: str, wrapped: bytes, context: dict[str, str]) -> bytes:
        if self.down:
            self.opened += 1
            raise ContentKeyError("unavailable", "down")
        return await super().open_data_key(wrapper, wrapped, context)


async def test_polling_through_an_outage_does_not_hold_the_pause_open(
    key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = [1_000.0]
    monkeypatch.setattr(_content, "monotonic", lambda: now[0])
    keys = _FlakyKeys()
    sealed = await SpanContent(keys).seal(_CONTENT, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1")
    reader = SpanContent(keys)
    keys.down = True
    assert await reader.open(sealed, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1") == {}
    asked = keys.opened

    # The dashboard polls every five seconds; the source recovers ten seconds in.
    opened: dict[str, str] = {}
    for tick in range(1, 14):
        now[0] += 5
        if tick == 2:
            keys.down = False
        opened = await reader.open(sealed, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1")
        if opened:
            break

    assert opened == _CONTENT
    # Asked again once the pause ran out, thirty seconds after the failure, and not before.
    assert now[0] - 1_000.0 == _content._PAUSE_AFTER_FAILURE_S
    assert keys.opened == asked + 1


async def test_a_slow_key_source_costs_the_content_and_not_the_batch(key: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_content, "_KEY_SOURCE_BUDGET_S", 0.05)
    keys = _FlakyKeys()
    keys.delay = 1.0
    content = SpanContent(keys)
    before, paused = _count(UNSEALED, "unavailable"), _count(UNSEALED, "paused")

    started = asyncio.get_running_loop().time()
    assert await content.seal(_CONTENT, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1") is None
    assert await content.seal(_CONTENT, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-2") is None

    assert asyncio.get_running_loop().time() - started < 0.5
    assert _count(UNSEALED, "unavailable") == before + 1
    assert _count(UNSEALED, "paused") == paused + 1


async def test_an_idle_process_holds_no_key_past_its_age(key: str, monkeypatch: pytest.MonkeyPatch) -> None:
    now = [1_000.0]
    monkeypatch.setattr(_content, "monotonic", lambda: now[0])
    content = SpanContent(_CountingKeys())
    sealed = await content.seal(_CONTENT, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1")
    await content.open(sealed, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1")
    assert content._write_keys and content._open_keys
    assert content._sweep is not None

    now[0] += _content._KEY_MAX_AGE_S
    content._run_sweep()

    assert not content._write_keys and not content._open_keys
    assert content._sweep is None


async def test_each_idle_key_goes_at_its_own_age_however_staggered(key: str, monkeypatch: pytest.MonkeyPatch) -> None:
    now = [1_000.0]
    monkeypatch.setattr(_content, "monotonic", lambda: now[0])
    content = SpanContent(_CountingKeys())
    loop = asyncio.get_running_loop()

    def due_in() -> float:
        assert content._sweep is not None
        return content._sweep.when() - loop.time()

    await content.seal(_CONTENT, workspace_id=_WORKSPACE, trace_id="first", span_id="tool-1")
    now[0] += 1
    later = await content.seal(_CONTENT, workspace_id=_WORKSPACE, trace_id="second", span_id="tool-1")
    now[0] += 1
    await content.open(later, workspace_id=_WORKSPACE, trace_id="second", span_id="tool-1")
    try:
        # Woken as the first key ages out, the sweep comes back for each later one at its own age.
        now[0] = 1_000.0 + _content._KEY_MAX_AGE_S
        content._run_sweep()
        assert list(content._write_keys) == [(_WORKSPACE, "second")] and due_in() <= 1.1
        now[0] += 1
        content._run_sweep()
        assert not content._write_keys and content._open_keys and due_in() <= 1.1
        now[0] += 1
        content._run_sweep()
        assert not content._open_keys and content._sweep is None
    finally:
        content.clear()


async def test_stopping_drops_every_cached_key(key: str) -> None:
    content = SpanContent(_CountingKeys())
    sealed = await content.seal(_CONTENT, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1")
    await content.open(sealed, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1")

    content.clear()

    assert not content._write_keys and not content._open_keys and content._sweep is None


async def test_the_write_cache_is_bounded(key: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_content, "_KEY_CACHE_SIZE", 2)
    content = SpanContent(_CountingKeys())

    for n in range(3):
        await content.seal(_CONTENT, workspace_id=uuid.uuid4(), trace_id=_TRACE, span_id=f"tool-{n}")

    assert len(content._write_keys) == 2


# Batches, separate pauses, and the KMS key list.


async def test_a_batch_fetches_its_keys_together_under_one_deadline(key: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_content, "_KEY_SOURCE_BUDGET_S", 0.5)
    keys = _FlakyKeys()
    keys.delay = 0.2
    items = [SealItem(_CONTENT, uuid.uuid4(), _TRACE, f"tool-{n}") for n in range(5)]

    started = asyncio.get_running_loop().time()
    sealed = await SpanContent(keys).seal_many(items)

    # Five workspaces at 0.2s each, fetched one after another, would take a second.
    assert asyncio.get_running_loop().time() - started < 0.45
    assert all(value is not None for value in sealed) and keys.made == 5


async def test_a_slow_source_costs_a_wide_batch_its_content_within_the_deadline(
    key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_content, "_KEY_SOURCE_BUDGET_S", 0.05)
    keys = _FlakyKeys()
    keys.delay = 1.0
    items = [SealItem(_CONTENT, uuid.uuid4(), _TRACE, f"tool-{n}") for n in range(5)]

    started = asyncio.get_running_loop().time()
    sealed = await SpanContent(keys).seal_many(items)

    assert asyncio.get_running_loop().time() - started < 0.5
    assert sealed == [None] * 5


async def test_a_failed_read_does_not_stop_capture(key: str) -> None:
    keys = _FlakyKeys()
    sealed = await SpanContent(keys).seal(_CONTENT, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1")
    content = SpanContent(keys)
    keys.down = True
    assert await content.open(sealed, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1") == {}
    keys.down = False

    assert await content.seal(_CONTENT, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-2") is not None


async def test_a_wrapper_no_longer_configured_is_rejected_and_pauses_nothing(kms: Any, key: str) -> None:
    sealed = await SpanContent(_kms_keyring(kms, _new_kms_key(kms))).seal(
        _CONTENT, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1"
    )
    content = _secret_key_content()
    before = _count(UNOPENED, "rejected")

    assert await content.open(sealed, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1") == {}
    assert _count(UNOPENED, "rejected") == before + 1
    assert await content.seal(_CONTENT, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-2") is not None


async def test_a_listed_key_that_cannot_be_described_is_skipped(kms: Any) -> None:
    current = _new_kms_key(kms)
    sealed = await SpanContent(_kms_keyring(kms, current)).seal(
        _CONTENT, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1"
    )

    reader = SpanContent(_kms_keyring(kms, current, previous=("alias/gone",)))

    assert await reader.open(sealed, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1") == _CONTENT


async def test_the_listed_keys_are_described_together_not_one_after_another() -> None:
    class _SlowDescribe:
        def describe_key(self, KeyId: str) -> dict[str, Any]:  # noqa: N803 - boto3's own keyword
            time.sleep(0.4)
            return {"KeyMetadata": {"Arn": f"arn:aws:kms:{_REGION}:000000000000:key/{KeyId}"}}

    keys = AwsKmsContentKeys("k1", ("k2", "k3", "k4"), client=_SlowDescribe())
    started = time.monotonic()
    try:
        allowed = await keys._allowed()
    finally:
        keys.close()

    assert len(allowed) == 4
    # One after another would take 1.6 s; together, about one call's 0.4 s.
    assert time.monotonic() - started < 1.0


async def test_an_alias_pointed_at_a_new_key_is_followed_without_a_restart(
    kms: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gateway.adapters import content_key_adapter

    now = [1_000.0]
    monkeypatch.setattr(content_key_adapter, "monotonic", lambda: now[0])
    old, new = _new_kms_key(kms), _new_kms_key(kms)
    kms.create_alias(AliasName="alias/otari-trace-content", TargetKeyId=old)
    keyring = _kms_keyring(kms, "alias/otari-trace-content")
    writer = SpanContent(keyring)
    first = await writer.seal(_CONTENT, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-0")
    # Opening resolves the alias, which is what a retarget has to outlive.
    assert await SpanContent(keyring).open(first, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-0")

    kms.update_alias(AliasName="alias/otari-trace-content", TargetKeyId=new)
    other = uuid.uuid4()
    resealed = await writer.seal(_CONTENT, workspace_id=other, trace_id=_TRACE, span_id="tool-1")
    assert await SpanContent(keyring).open(resealed, workspace_id=other, trace_id=_TRACE, span_id="tool-1") == {}
    now[0] += content_key_adapter._ALLOWED_KEYS_TTL_S

    assert await SpanContent(keyring).open(resealed, workspace_id=other, trace_id=_TRACE, span_id="tool-1") == _CONTENT


async def test_kms_opens_a_sessions_data_key_for_that_session_alone(kms: Any) -> None:
    sealed = await SpanContent(_kms_keyring(kms, _new_kms_key(kms))).seal(
        _CONTENT, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1"
    )

    with pytest.raises(ClientError) as refused:
        kms.decrypt(CiphertextBlob=_wrapped(sealed), EncryptionContext={**_CONTEXT, "otari:session": "session-2"})

    assert refused.value.response["Error"]["Code"] == "InvalidCiphertextException"


async def test_each_session_gets_a_data_key_of_its_own(key: str) -> None:
    keys = _CountingKeys()
    content = SpanContent(keys)

    for session in ("session-1", "session-2", "session-3"):
        for n in range(10):
            await content.seal(_CONTENT, workspace_id=_WORKSPACE, trace_id=session, span_id=f"tool-{n}")

    assert keys.made == 3


class _ThrottledOnceKms:
    """A KMS client whose first DescribeKey of ``throttled`` fails, the rest passing through."""

    def __init__(self, inner: Any, throttled: str) -> None:
        self._inner, self._throttled, self._failed = inner, throttled, False
        self.generate_data_key = inner.generate_data_key
        self.decrypt = inner.decrypt

    def describe_key(self, **kwargs: Any) -> Any:
        if kwargs["KeyId"] == self._throttled and not self._failed:
            self._failed = True
            raise _client_error("ThrottlingException")
        return self._inner.describe_key(**kwargs)


async def test_one_throttled_describe_does_not_shut_out_the_current_key(
    kms: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gateway.adapters import content_key_adapter

    now = [1_000.0]
    monkeypatch.setattr(content_key_adapter, "monotonic", lambda: now[0])
    monkeypatch.setattr(_content, "monotonic", lambda: now[0])
    current, old = _new_kms_key(kms), _new_kms_key(kms)
    sealed = await SpanContent(_kms_keyring(kms, current)).seal(
        _CONTENT, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1"
    )
    reader = SpanContent(_kms_keyring(_ThrottledOnceKms(kms, current), current, previous=(old,)))
    before = _count(UNOPENED, "unavailable")

    # The current key could not be described: an outage, which pauses reads, not tampering.
    assert await reader.open(sealed, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1") == {}
    assert _count(UNOPENED, "unavailable") == before + 1

    now[0] += max(content_key_adapter._PARTIAL_KEYS_TTL_S, _content._PAUSE_AFTER_FAILURE_S)
    assert await reader.open(sealed, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1") == _CONTENT


async def test_a_reading_with_its_own_keys_asks_the_source_afresh_and_leaves_nothing_cached(key: str) -> None:
    keys = _CountingKeys()
    content = SpanContent(keys)
    sealed = [await content.seal(_CONTENT, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id=f"t{n}") for n in range(3)]
    for n, value in enumerate(sealed):
        await content.open(value, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id=f"t{n}")
    assert keys.opened == 1

    for _ in range(2):
        opened: dict[Any, bytes] = {}
        for n, value in enumerate(sealed):
            assert (
                await content.open(value, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id=f"t{n}", keys_opened=opened)
                == _CONTENT
            )

    # Once per reading, however warm the shared cache was.
    assert keys.opened == 3


# The startup check, and stopping.


async def test_the_startup_check_passes_with_a_key_it_can_use(kms: Any) -> None:
    content = SpanContent(_kms_keyring(kms, _new_kms_key(kms)))

    assert await content.check() is None
    # It caches nothing, so no session holds the check's key.
    assert not content._write_keys and not content._open_keys


@pytest.mark.parametrize(
    ("error", "reason"),
    [(_client_error("AccessDeniedException"), "denied"), (EndpointConnectionError(endpoint_url="x"), "unavailable")],
)
async def test_the_startup_check_reports_a_key_it_cannot_use(error: Exception, reason: str) -> None:
    content = SpanContent(_kms_keyring(_FailingKmsClient(error), "alias/otari"))

    failure = await content.check()

    assert failure is not None and failure.reason == reason
    assert "secret-detail" not in str(failure)


async def test_the_startup_check_reports_no_key_at_all(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTARI_SECRET_KEY", raising=False)
    content = _secret_key_content()

    assert content.sealing is False
    failure = await content.check()
    assert failure is not None and failure.reason == "unavailable"


async def test_a_kms_key_with_no_region_is_a_configuration_error(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("AWS_REGION", "AWS_DEFAULT_REGION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", "/nonexistent")

    with pytest.raises(ValueError, match="trace_content_kms_region"):
        AwsKmsContentKeys("alias/otari")


async def test_closing_drops_the_keys_and_stops_the_kms_pool_which_the_next_call_starts_again(kms: Any) -> None:
    source = AwsKmsContentKeys(_new_kms_key(kms), client=kms)
    content = SpanContent(
        ContentKeyring(KMS_WRAPPER, {SECRET_KEY_WRAPPER: SecretKeyContentKeys(), KMS_WRAPPER: source})
    )
    await content.seal(_CONTENT, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1")
    pool = source._executor
    assert pool is not None

    content.close()

    assert not content._write_keys and content._sweep is None
    assert pool._shutdown and source._executor is None
    # A process whose app starts again (as the tests' does) keeps sealing.
    sealed = await content.seal(_CONTENT, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-2")
    assert sealed is not None


async def test_a_failed_startup_check_withdraws_content_until_a_key_is_made(key: str) -> None:
    keys = _FlakyKeys()
    keys.down = True
    content = SpanContent(keys)

    assert await content.check() is not None
    assert content.sealing is False

    keys.down = False
    sealed = await content.seal(_CONTENT, workspace_id=_WORKSPACE, trace_id=_TRACE, span_id="tool-1")

    assert sealed is not None and content.sealing is True


# One workspace's new sessions, against a key source every workspace shares.


async def test_one_workspace_cannot_ask_for_more_than_its_share_of_new_keys(
    key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_content, "_NEW_KEYS_BURST", 3.0)
    monkeypatch.setattr(_content, "_NEW_KEYS_PER_S", 0.0)
    keys = _CountingKeys()
    content = SpanContent(keys)
    limited_before = _count(UNSEALED, "limited")
    # One export naming many sessions, each needing a key of its own.
    items = [SealItem(_CONTENT, _WORKSPACE, f"session-{n}", "tool-1") for n in range(10)]

    sealed = await content.seal_many(items)

    assert keys.made == 3
    assert sum(value is not None for value in sealed) == 3
    assert _count(UNSEALED, "limited") - limited_before == 7
    # Another workspace's sessions are not held back by it.
    other = await content.seal(_CONTENT, workspace_id=_OTHER_WORKSPACE, trace_id=_TRACE, span_id="tool-1")
    assert other is not None
    # Nor are the sessions that already have a key.
    again = await content.seal(_CONTENT, workspace_id=_WORKSPACE, trace_id="session-0", span_id="tool-2")
    assert again is not None and keys.made == 4


async def test_a_workspaces_allowance_of_new_keys_refills(key: str, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [1_000.0]
    monkeypatch.setattr(_content, "monotonic", lambda: clock[0])
    monkeypatch.setattr(_content, "_NEW_KEYS_BURST", 1.0)
    monkeypatch.setattr(_content, "_NEW_KEYS_PER_S", 1.0)
    content = SpanContent(_CountingKeys())

    first = await content.seal(_CONTENT, workspace_id=_WORKSPACE, trace_id="session-1", span_id="tool-1")
    refused = await content.seal(_CONTENT, workspace_id=_WORKSPACE, trace_id="session-2", span_id="tool-1")
    clock[0] += 1.0
    later = await content.seal(_CONTENT, workspace_id=_WORKSPACE, trace_id="session-2", span_id="tool-1")

    assert first is not None and refused is None and later is not None
