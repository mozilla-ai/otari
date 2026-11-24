"""The hosted name index keeps a write this worker made while a refresh was reading."""

import asyncio
import uuid
from collections.abc import Iterator
from typing import Any

import pytest

from gateway.core.config import GatewayConfig
from gateway.services.providers import _hosted_name_index as index


class _Row:
    def __init__(self, name: str, provider: str) -> None:
        self.id = uuid.uuid4()
        self.name = name
        self.provider = provider
        self.zero_data_retention = False
        self.data_location = None


class _Uow:
    async def __aenter__(self) -> "_Uow":
        return self

    async def __aexit__(self, *_: object) -> None:
        return None


class _NoRegions:
    def __init__(self, _uow: object) -> None:
        pass

    async def list_for_providers(self, _ids: object) -> dict[uuid.UUID, list[object]]:
        return {}


@pytest.fixture(autouse=True)
def _clean_index(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(index, "HostedProviderRegionRepository", _NoRegions)
    index.reset_hosted_name_index()
    yield
    index.reset_hosted_name_index()


def _repository_reading(rows: list[_Row], during_read: Any) -> type:
    class _Repository:
        def __init__(self, _uow: object) -> None:
            pass

        async def list_all(self) -> list[_Row]:
            during_read()
            return rows

    return _Repository


@pytest.mark.asyncio
async def test_a_provider_created_while_the_refresh_read_survives_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """The read began before the create committed, so its snapshot lacks it; the create must not be lost."""
    monkeypatch.setattr(
        index,
        "HostedProviderRepository",
        _repository_reading([], lambda: index.remember_hosted_name("openai-eu", "openai")),
    )

    await index.refresh_hosted_name_index(_Uow(), GatewayConfig())  # type: ignore[arg-type]

    assert index.hosted_implementation("openai-eu") == "openai"


@pytest.mark.asyncio
async def test_a_provider_removed_while_the_refresh_read_stays_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    index.remember_hosted_name("openai-eu", "openai")
    monkeypatch.setattr(
        index,
        "HostedProviderRepository",
        _repository_reading([_Row("openai-eu", "openai")], lambda: index.forget_hosted_name("openai-eu")),
    )

    await index.refresh_hosted_name_index(_Uow(), GatewayConfig())  # type: ignore[arg-type]

    assert index.hosted_implementation("openai-eu") is None


@pytest.mark.asyncio
async def test_a_refresh_after_the_write_reads_it_from_the_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Once a read sees the write, the replay is dropped and the rows decide."""
    index.remember_hosted_name("openai-eu", "openai")
    monkeypatch.setattr(index, "HostedProviderRepository", _repository_reading([], lambda: None))

    await index.refresh_hosted_name_index(_Uow(), GatewayConfig())  # type: ignore[arg-type]

    assert index.hosted_implementation("openai-eu") is None


@pytest.mark.asyncio
async def test_a_retention_declaration_stored_while_the_refresh_read_survives_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        index,
        "HostedProviderRepository",
        _repository_reading(
            [_Row("openai-eu", "openai")], lambda: index.remember_hosted_zero_data_retention("openai-eu", True)
        ),
    )

    await index.refresh_hosted_name_index(_Uow(), GatewayConfig())  # type: ignore[arg-type]

    assert index.hosted_zero_data_retention("openai-eu")


@pytest.mark.asyncio
async def test_a_refresh_after_a_retention_write_reads_it_from_the_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    index.remember_hosted_zero_data_retention("openai-eu", True)
    monkeypatch.setattr(
        index, "HostedProviderRepository", _repository_reading([_Row("openai-eu", "openai")], lambda: None)
    )

    await index.refresh_hosted_name_index(_Uow(), GatewayConfig())  # type: ignore[arg-type]

    assert not index.hosted_zero_data_retention("openai-eu")


@pytest.mark.asyncio
async def test_a_retention_declaration_withdrawn_while_the_refresh_read_stays_withdrawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    declared = _Row("openai-eu", "openai")
    declared.zero_data_retention = True
    monkeypatch.setattr(
        index,
        "HostedProviderRepository",
        _repository_reading([declared], lambda: index.remember_hosted_zero_data_retention("openai-eu", False)),
    )

    await index.refresh_hosted_name_index(_Uow(), GatewayConfig())  # type: ignore[arg-type]

    assert not index.hosted_zero_data_retention("openai-eu")


@pytest.mark.asyncio
async def test_a_refresh_that_finishes_after_a_later_one_does_not_publish_its_older_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two refreshes overlap; the first one's read is older, so the second one's answer stands."""
    first_reading = asyncio.Event()
    release_first = asyncio.Event()
    reads = [[], [_Row("openai-eu", "openai")]]

    class _Repository:
        def __init__(self, _uow: object) -> None:
            pass

        async def list_all(self) -> list[_Row]:
            rows = reads.pop(0)
            if not reads:
                return rows
            first_reading.set()
            await release_first.wait()
            return rows

    monkeypatch.setattr(index, "HostedProviderRepository", _Repository)

    first = asyncio.create_task(index.refresh_hosted_name_index(_Uow(), GatewayConfig()))  # type: ignore[arg-type]
    await first_reading.wait()
    await index.refresh_hosted_name_index(_Uow(), GatewayConfig())  # type: ignore[arg-type]
    release_first.set()
    await first

    assert index.hosted_implementation("openai-eu") == "openai"
