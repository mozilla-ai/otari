"""Both core data-key adapters keep the port's contract.

A data key unwraps only under the context it was generated for, so a wrapped
key copied into another workspace's or another session's row yields nothing. The
KMS adapter is run against a stand-in client that enforces the encryption
context the way KMS does.
"""

import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from cryptography.fernet import Fernet

from gateway.adapters.data_key_adapter import AwsKmsDataKeys, SecretBoxDataKeys
from gateway.container import build_container
from gateway.core.config import GatewayConfig
from gateway.ports.data_key_port import (
    DataKeyContext,
    DataKeyContextMismatchError,
    DataKeyPort,
    DataKeyUnavailableError,
)

_CONTEXT = DataKeyContext(workspace_id=uuid.uuid4(), session="s-session-1")


class InvalidCiphertextException(Exception):
    """Named as botocore names the error KMS returns for a wrong encryption context."""


class _FakeKms:
    """Wraps a key by remembering it with its encryption context, and refuses any other context."""

    def __init__(self) -> None:
        self._keys: dict[bytes, tuple[bytes, dict[str, str]]] = {}

    def generate_data_key(self, *, KeyId: str, KeySpec: str, EncryptionContext: dict[str, str]) -> dict[str, Any]:
        plaintext = uuid.uuid4().bytes * 2
        blob = uuid.uuid4().bytes
        self._keys[blob] = (plaintext, dict(EncryptionContext))
        return {"Plaintext": plaintext, "CiphertextBlob": blob}

    def decrypt(self, *, CiphertextBlob: bytes, KeyId: str, EncryptionContext: dict[str, str]) -> dict[str, Any]:
        stored = self._keys.get(CiphertextBlob)
        if stored is None or stored[1] != EncryptionContext:
            raise InvalidCiphertextException
        return {"Plaintext": stored[0]}


@pytest.fixture
def secret_key(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("OTARI_SECRET_KEY", Fernet.generate_key().decode())
    yield


@pytest.fixture(params=["secret_box", "aws_kms"])
def keys(request: pytest.FixtureRequest, secret_key: None) -> DataKeyPort:
    if request.param == "aws_kms":
        return AwsKmsDataKeys("alias/otari-traces", client=_FakeKms())
    return SecretBoxDataKeys()


@pytest.mark.asyncio
async def test_a_data_key_unwraps_under_its_own_context(keys: DataKeyPort) -> None:
    key = await keys.generate(_CONTEXT)

    assert len(key.plaintext) == 32
    assert key.wrapped != key.plaintext
    assert await keys.unwrap(key.wrapped, _CONTEXT) == key.plaintext


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "other",
    [
        DataKeyContext(workspace_id=uuid.uuid4(), session=_CONTEXT.session),
        DataKeyContext(workspace_id=_CONTEXT.workspace_id, session="s-session-2"),
    ],
    ids=["another workspace", "another session"],
)
async def test_a_data_key_refuses_any_other_context(keys: DataKeyPort, other: DataKeyContext) -> None:
    key = await keys.generate(_CONTEXT)

    with pytest.raises(DataKeyContextMismatchError):
        await keys.unwrap(key.wrapped, other)


@pytest.mark.asyncio
async def test_secret_box_refuses_to_issue_keys_without_a_secret_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTARI_SECRET_KEY", raising=False)

    with pytest.raises(DataKeyUnavailableError, match="OTARI_SECRET_KEY"):
        await SecretBoxDataKeys().generate(_CONTEXT)


@pytest.mark.asyncio
async def test_secret_box_refuses_a_key_another_secret_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", Fernet.generate_key().decode())
    key = await SecretBoxDataKeys().generate(_CONTEXT)
    monkeypatch.setenv("OTARI_SECRET_KEY", Fernet.generate_key().decode())

    with pytest.raises(DataKeyContextMismatchError):
        await SecretBoxDataKeys().unwrap(key.wrapped, _CONTEXT)


def test_the_kms_backend_needs_a_key_id() -> None:
    with pytest.raises(DataKeyUnavailableError, match="trace_content_kms_key_id"):
        AwsKmsDataKeys("", client=_FakeKms())


def test_the_kms_backend_must_name_its_key_in_config() -> None:
    with pytest.raises(ValueError, match="trace_content_kms_key_id"):
        GatewayConfig(trace_content_key_backend="aws_kms")


def test_the_configured_backend_is_what_the_container_binds(secret_key: None) -> None:
    container = build_container(config=GatewayConfig(), workspace_listener=None)

    assert isinstance(container.resolve(DataKeyPort, None), SecretBoxDataKeys)


@pytest.mark.asyncio
async def test_a_data_keys_repr_shows_neither_the_plaintext_nor_the_wrapped_key(keys: DataKeyPort) -> None:
    key = await keys.generate(_CONTEXT)

    assert key.plaintext.hex() not in repr(key)
    assert repr(key.plaintext) not in repr(key)
    assert repr(key.wrapped) not in repr(key)


@pytest.mark.asyncio
async def test_a_configured_backend_is_available(keys: DataKeyPort) -> None:
    assert await keys.available() is True


@pytest.mark.asyncio
async def test_secret_box_is_unavailable_without_a_secret_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTARI_SECRET_KEY", raising=False)

    assert await SecretBoxDataKeys().available() is False


class _DeniedKms(_FakeKms):
    def generate_data_key(self, *, KeyId: str, KeySpec: str, EncryptionContext: dict[str, str]) -> dict[str, Any]:
        raise PermissionError("AccessDeniedException")


@pytest.mark.asyncio
async def test_kms_is_unavailable_when_it_refuses_to_generate() -> None:
    assert await AwsKmsDataKeys("alias/otari-traces", client=_DeniedKms()).available() is False


@pytest.mark.asyncio
async def test_secret_box_refuses_a_payload_of_another_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = Fernet.generate_key()
    monkeypatch.setenv("OTARI_SECRET_KEY", secret.decode())

    with pytest.raises(DataKeyContextMismatchError):
        await SecretBoxDataKeys().unwrap(Fernet(secret).encrypt(b'["not", "a", "key"]'), _CONTEXT)


def test_content_capture_is_off_unless_the_operator_opts_in() -> None:
    assert GatewayConfig().trace_content_capture_max == "off"


class _RefusingKms:
    class AccessDeniedException(Exception):
        pass

    def generate_data_key(self, **kwargs: object) -> dict[str, bytes]:
        raise self.AccessDeniedException("arn:aws:kms:eu-west-1:123456789012:key/secret")

    def decrypt(self, **kwargs: object) -> dict[str, bytes]:
        raise self.AccessDeniedException("arn:aws:kms:eu-west-1:123456789012:key/secret")


@pytest.mark.asyncio
async def test_a_kms_refusal_is_the_ports_unavailable_error() -> None:
    keys = AwsKmsDataKeys("arn:aws:kms:eu-west-1:123456789012:key/abc", client=_RefusingKms())

    with pytest.raises(DataKeyUnavailableError) as generated:
        await keys.generate(_CONTEXT)
    with pytest.raises(DataKeyUnavailableError) as unwrapped:
        await keys.unwrap(b"wrapped", _CONTEXT)

    for caught in (generated, unwrapped):
        assert "AccessDeniedException" in str(caught.value)
        assert "123456789012" not in str(caught.value)


@pytest.mark.parametrize("key_id", ["alias/otari-traces", "arn:aws:kms:eu-west-1:123456789012:alias/otari-traces"])
def test_the_kms_key_is_named_by_id_not_alias(key_id: str) -> None:
    with pytest.raises(ValueError, match="not by alias"):
        GatewayConfig(trace_content_key_backend="aws_kms", trace_content_kms_key_id=key_id)
