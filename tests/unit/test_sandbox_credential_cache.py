"""The per-worker cache of the stored sandbox key, against a fake repository."""

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from gateway.models.tools import SandboxProvider
from gateway.services.tools import _sandbox_credential as credentials


class _Uow:
    async def __aenter__(self) -> "_Uow":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None


class _Repository:
    """Holds one row; a read can be held open to stage a write in the middle of it."""

    row: Any = None
    read_started = asyncio.Event()
    release_read: asyncio.Event | None = None

    def __init__(self, _uow: object) -> None:
        pass

    async def get_committed(self, _provider: str) -> Any:
        row = type(self).row
        type(self).read_started.set()
        release = type(self).release_read
        if release is not None:
            await release.wait()
        return row

    async def put(self, provider: str, *, encrypted_api_key: str, last4: str) -> None:
        type(self).row = SimpleNamespace(provider=provider, encrypted_api_key=encrypted_api_key, last4=last4)

    async def remove(self, _provider: str) -> bool:
        type(self).row = None
        return True


@pytest.fixture(autouse=True)
def _fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    _Repository.row = None
    _Repository.read_started = asyncio.Event()
    _Repository.release_read = None
    monkeypatch.setattr(credentials, "SandboxCredentialRepository", _Repository)
    monkeypatch.setattr(credentials, "encrypt_secret", lambda plaintext: f"enc:{plaintext}")
    monkeypatch.setattr(credentials, "decrypt_secret", lambda ciphertext: ciphertext.removeprefix("enc:"))
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    credentials.reset_sandbox_credential_cache()


@pytest.mark.asyncio
async def test_a_refresh_that_read_before_a_write_does_not_publish_over_it() -> None:
    await credentials.store_sandbox_api_key(_Uow(), SandboxProvider.E2B, "e2b_old_aaaa")  # type: ignore[arg-type]
    _Repository.release_read = asyncio.Event()

    refresh = asyncio.create_task(credentials.refresh_sandbox_credential_cache(_Uow()))  # type: ignore[arg-type]
    await _Repository.read_started.wait()
    await credentials.store_sandbox_api_key(_Uow(), SandboxProvider.E2B, "e2b_new_bbbb")  # type: ignore[arg-type]
    _Repository.release_read.set()
    await refresh

    assert credentials.sandbox_api_key(SandboxProvider.E2B) == "e2b_new_bbbb"


@pytest.mark.asyncio
async def test_a_refresh_with_no_write_in_between_publishes_what_it_read() -> None:
    _Repository.row = SimpleNamespace(provider="e2b", encrypted_api_key="enc:e2b_other_cccc", last4="cccc")

    await credentials.refresh_sandbox_credential_cache(_Uow())  # type: ignore[arg-type]

    assert credentials.sandbox_api_key(SandboxProvider.E2B) == "e2b_other_cccc"
    assert credentials.sandbox_credential_status(SandboxProvider.E2B).source == "stored"
