"""Stored guardrails are built when the gateway starts, not when a request arrives.

The pass is a background task the lifespan starts and never awaits, so every
assertion here polls for it to settle rather than sleeping a fixed time.

``AnyGuardrail.create`` is stubbed throughout: a real one is a vendor client
wanting a real key, and what these cover is which rows reach it.
"""

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from any_guardrail import AnyGuardrail
from sqlalchemy.orm import Session

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.models.guardrails import GuardrailCredential
from gateway.services.guardrail_loader import apply_stored_guardrail
from gateway.services.guardrail_runner import get_guardrail_runner, reset_guardrail_runner
from gateway.services.secret_box import encrypt_secret, generate_secret_key

from .conftest import build_test_client

_SETTLE_S = 10.0


@pytest.fixture
def test_config(postgres_url: str) -> GatewayConfig:
    return GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        require_pricing=False,
    )


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    yield


@pytest.fixture(autouse=True)
def _no_runner_between_tests() -> Iterator[None]:
    """The runner is process-wide, and a profile left behind would fake a pass."""
    reset_guardrail_runner()
    yield
    reset_guardrail_runner()


@pytest.fixture(autouse=True)
def builds(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record every construction, and hand back something that is not a vendor client."""
    seen: list[dict[str, Any]] = []

    def _create(name: Any, **kwargs: Any) -> object:
        seen.append({"guardrail": name.value, **kwargs})
        return object()

    monkeypatch.setattr(AnyGuardrail, "create", _create)
    return seen


def _store(session: Session, name: str, *, enabled: bool = True, api_key: str = "lak-secret") -> None:
    session.add(
        GuardrailCredential(
            name=name,
            guardrail_name="lakera_guard",
            create_kwargs={"endpoint": "https://api.lakera.ai/v2/guard"},
            validate_kwargs={},
            encrypted_create_secrets=encrypt_secret(f'{{"api_key": "{api_key}"}}'),
            enabled=enabled,
        )
    )
    session.commit()


def _settled(predicate: Callable[[], bool]) -> bool:
    """Wait for the background pass, which nothing in the app hands back a handle to."""
    deadline = time.monotonic() + _SETTLE_S
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


@contextmanager
def _booted(config: GatewayConfig) -> Iterator[None]:
    """Run one gateway lifespan, which is all most of these need from a client."""
    clients = build_test_client(config)
    next(clients)
    try:
        yield
    finally:
        clients.close()


def test_a_stored_guardrail_is_built_before_any_request(
    test_config: GatewayConfig, clean_database: None, db_session: Session, builds: list[dict[str, Any]]
) -> None:
    _store(db_session, "prompt-injection")

    with _booted(test_config):
        assert _settled(lambda: get_guardrail_runner().knows("prompt-injection"))
        # The two halves of the row arrive as one constructor call.
        assert builds == [
            {
                "guardrail": "lakera_guard",
                "endpoint": "https://api.lakera.ai/v2/guard",
                "api_key": "lak-secret",
            }
        ]


def test_a_disabled_guardrail_is_not_built(
    test_config: GatewayConfig, clean_database: None, db_session: Session, builds: list[dict[str, Any]]
) -> None:
    """Turning one off without losing its configuration is what ``enabled`` is for."""
    _store(db_session, "off", enabled=False)
    _store(db_session, "on")

    with _booted(test_config):
        assert _settled(lambda: get_guardrail_runner().knows("on"))
        assert not get_guardrail_runner().knows("off")
        assert len(builds) == 1


def test_one_definition_that_will_not_build_does_not_stop_the_gateway(
    test_config: GatewayConfig, clean_database: None, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A vendor client that refuses to construct is one profile's problem, not the boot's."""
    _store(db_session, "broken", api_key="bad")
    _store(db_session, "fine")

    def _create(name: Any, **kwargs: Any) -> object:
        if kwargs.get("api_key") == "bad":
            raise RuntimeError("nope")
        return object()

    monkeypatch.setattr(AnyGuardrail, "create", _create)

    clients = build_test_client(test_config)
    client = next(clients)
    try:
        assert client.get(f"{API_ROOT}/health").status_code == 200
        assert _settled(lambda: get_guardrail_runner().knows("fine"))
        assert not get_guardrail_runner().knows("broken")
    finally:
        clients.close()


def test_a_row_whose_secrets_cannot_be_read_is_skipped_and_the_rest_are_built(
    test_config: GatewayConfig,
    clean_database: None,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    builds: list[dict[str, Any]],
) -> None:
    """A rotated key costs that row and nothing else; the store already flags it."""
    unreadable = encrypt_secret('{"api_key": "written-under-the-old-key"}')
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    db_session.add(
        GuardrailCredential(
            name="stale",
            guardrail_name="lakera_guard",
            create_kwargs={},
            validate_kwargs={},
            encrypted_create_secrets=unreadable,
        )
    )
    db_session.commit()
    _store(db_session, "fresh")

    with _booted(test_config):
        assert _settled(lambda: get_guardrail_runner().knows("fresh"))
        assert not get_guardrail_runner().knows("stale")
        assert len(builds) == 1


def test_a_deployment_with_no_definitions_builds_nothing(
    test_config: GatewayConfig, clean_database: None, builds: list[dict[str, Any]]
) -> None:
    with _booted(test_config):
        assert builds == []
        assert get_guardrail_runner().profiles() == frozenset()


@pytest.mark.asyncio
async def test_applying_a_disabled_row_forgets_it_rather_than_building_it(
    clean_database: None, db_session: Session, builds: list[dict[str, Any]]
) -> None:
    """One rule, read by the pass and by a write, so the two cannot disagree."""
    _store(db_session, "p")
    row = db_session.query(GuardrailCredential).filter_by(name="p").one()
    await apply_stored_guardrail(row)
    assert get_guardrail_runner().knows("p")

    row.enabled = False
    await apply_stored_guardrail(row)

    assert not get_guardrail_runner().knows("p")


@pytest.mark.asyncio
async def test_applying_a_row_whose_secrets_will_not_read_forgets_it(
    clean_database: None, db_session: Session, monkeypatch: pytest.MonkeyPatch, builds: list[dict[str, Any]]
) -> None:
    _store(db_session, "p")
    row = db_session.query(GuardrailCredential).filter_by(name="p").one()
    await apply_stored_guardrail(row)
    assert get_guardrail_runner().knows("p")

    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    await apply_stored_guardrail(row)

    assert not get_guardrail_runner().knows("p")


@pytest.mark.asyncio
async def test_applying_a_row_that_will_not_build_raises_nothing(
    clean_database: None, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The write is committed either way, so the worst outcome is a profile left cold."""
    _store(db_session, "p")
    row = db_session.query(GuardrailCredential).filter_by(name="p").one()

    def _refuse(name: Any, **kwargs: Any) -> object:
        raise RuntimeError("nope")

    monkeypatch.setattr(AnyGuardrail, "create", _refuse)
    await apply_stored_guardrail(row)

    assert not get_guardrail_runner().knows("p")
