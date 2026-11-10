"""Unit tests for first-run master-key bootstrap and the enriched 402 body."""

import hashlib
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from gateway.api import deps
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.main import create_app
from gateway.models.api_keys import APIKey
from gateway.models.platform import RuntimeSetting
from gateway.services import master_key_service
from gateway.services.master_key_service import (
    MASTER_KEY_HASH_KEY,
    MasterKeyRotationConflictError,
    ensure_master_key,
    generate_master_key,
    hash_master_key,
    setup_link,
    stage_generated_master_key_rotation,
)
from gateway.services.pricing_service import configure_default_pricing, no_pricing_error_detail


@pytest.fixture(autouse=True)
def _no_master_key_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # A master key in the environment would suppress generation and break these tests.
    monkeypatch.delenv("OTARI_MASTER_KEY", raising=False)
    monkeypatch.delenv("GATEWAY_MASTER_KEY", raising=False)
    yield


def _stored_hashes(database_url: str) -> list[str]:
    engine = create_engine(database_url)
    try:
        with Session(engine) as db:
            rows = db.query(RuntimeSetting).filter(RuntimeSetting.key == MASTER_KEY_HASH_KEY).all()
            return [row.value for row in rows]
    finally:
        engine.dispose()


def _api_key_names(database_url: str) -> list[str]:
    engine = create_engine(database_url)
    try:
        with Session(engine) as db:
            return [str(name) for (name,) in db.query(APIKey.key_name).all()]
    finally:
        engine.dispose()


def test_generate_master_key_format() -> None:
    token = generate_master_key()
    assert token.startswith("otari-mk-")
    assert len(token) > 30


def test_hash_master_key_is_plain_sha256() -> None:
    assert hash_master_key("otari-mk-abc") == hashlib.sha256(b"otari-mk-abc").hexdigest()


def test_setup_link_carries_the_key_in_the_fragment() -> None:
    link = setup_link(GatewayConfig(host="0.0.0.0", port=8000), "otari-mk-abc")

    assert link == "http://localhost:8000/#/setup?key=otari-mk-abc"
    # Everything from the "#" on stays in the browser, so the key is in no request.
    assert "otari-mk-abc" not in link.split("#", 1)[0]


def test_setup_link_uses_the_interface_address_when_one_is_set() -> None:
    config = GatewayConfig(public_base_url="https://otari.example.com/ui/")

    assert setup_link(config, "otari-mk-abc") == "https://otari.example.com/ui/#/setup?key=otari-mk-abc"


def _startup_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """What the start logged, apart from the note a checkout with no built dashboard gets.

    That note depends on whether ``make dashboard`` has run here, and it stays
    visible on purpose: without a bundle the setup link has no page to open.
    """
    return [record for record in caplog.records if not record.getMessage().startswith("No dashboard bundle found")]


def _start(config: GatewayConfig, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> list[str]:
    """Boot ``config`` at the CLI's default level and return the plain lines it printed."""
    printed: list[str] = []
    monkeypatch.setattr("gateway.main.log_plain", lambda *lines, level: printed.extend(lines))
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="gateway"), TestClient(create_app(config)):
        pass
    return printed


def test_a_first_start_prints_only_where_it_runs_and_the_setup_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Pins the whole of a zero-config first start's output, so startup chatter does not creep back."""
    monkeypatch.setattr("gateway.main.__version__", "1.2.3")
    config = GatewayConfig(database_url=f"sqlite:///{tmp_path / 'mk.db'}", port=8000)

    printed = _start(config, monkeypatch, caplog)

    link = printed[5]
    assert link.startswith("http://localhost:8000/#/setup?key=otari-mk-")
    assert printed == [
        "",
        "Otari 1.2.3 is running at http://localhost:8000",
        "",
        "Open this link to choose your password and finish setup:",
        "",
        link,
        "",
        "It is shown only once and carries your master key, so treat it like one.",
        "",
    ]
    key = link.split("key=", 1)[1]
    assert hash_master_key(key) == _stored_hashes(config.database_url)[0]
    # Nothing else at the default level: no log record, so no key in one either.
    assert [record.getMessage() for record in _startup_records(caplog)] == []


def test_a_later_start_says_where_to_sign_in_and_shows_no_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr("gateway.main.__version__", "1.2.3")
    database_url = f"sqlite:///{tmp_path / 'mk.db'}"
    _start(GatewayConfig(database_url=database_url, port=8000), monkeypatch, caplog)

    printed = _start(GatewayConfig(database_url=database_url, port=8000), monkeypatch, caplog)

    assert printed == ["Otari 1.2.3 is running at http://localhost:8000", "Sign in at http://localhost:8000/"]
    # The reminders the first start held back, and nothing below a warning.
    assert {record.levelno for record in _startup_records(caplog)} == {logging.WARNING}


def test_a_first_start_with_a_generated_master_key_creates_no_bootstrap_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    database_url = f"sqlite:///{tmp_path / 'mk.db'}"

    _start(GatewayConfig(database_url=database_url), monkeypatch, caplog)

    assert _api_key_names(database_url) == []


@pytest.mark.parametrize(
    "settings",
    [{"master_key": "operator-chosen-master-key"}, {"bootstrap_api_key": True}],
    ids=["operator master key", "bootstrap key asked for"],
)
def test_the_bootstrap_api_key_is_still_created_where_it_was_asked_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, settings: dict[str, Any]
) -> None:
    database_url = f"sqlite:///{tmp_path / 'mk.db'}"

    _start(GatewayConfig(database_url=database_url, **settings), monkeypatch, caplog)

    assert _api_key_names(database_url) == ["bootstrap"]


def test_first_run_generates_and_persists_a_hash(tmp_path: Path) -> None:
    config = GatewayConfig(database_url=f"sqlite:///{tmp_path / 'mk.db'}", require_pricing=False)
    with TestClient(create_app(config)):
        pass
    hashes = _stored_hashes(config.database_url)
    assert len(hashes) == 1
    assert hashes[0]


def test_generation_is_idempotent_across_restarts(tmp_path: Path) -> None:
    db_url = f"sqlite:///{tmp_path / 'mk.db'}"
    with TestClient(create_app(GatewayConfig(database_url=db_url, require_pricing=False))):
        pass
    first = _stored_hashes(db_url)
    # A fresh config against the same database stands in for a process restart.
    with TestClient(create_app(GatewayConfig(database_url=db_url, require_pricing=False))):
        pass
    second = _stored_hashes(db_url)
    assert first == second
    assert len(second) == 1


@pytest.mark.asyncio
async def test_first_run_race_adopts_winner_hash() -> None:
    # Simulate the losing worker: no hash on the pre-check, the INSERT collides
    # (IntegrityError on commit), and the winner's hash is present on re-read.
    config = GatewayConfig(require_pricing=False)
    assert config.master_key is None
    winner_hash = "winner-hash-value"

    session = AsyncMock()
    session.add = MagicMock()
    session.get.side_effect = [None, RuntimeSetting(key=MASTER_KEY_HASH_KEY, value=winner_hash)]
    session.commit.side_effect = IntegrityError("INSERT", {}, Exception("duplicate key"))

    await ensure_master_key(config, session)

    session.rollback.assert_awaited()
    assert config._master_key_hash == winner_hash


@pytest.mark.asyncio
async def test_rotation_rejects_a_stale_master_key_hash() -> None:
    session = AsyncMock()
    session.execute.return_value = MagicMock(rowcount=0)

    with pytest.raises(MasterKeyRotationConflictError, match="already rotated"):
        await stage_generated_master_key_rotation(session, "stale-hash")


def test_operator_key_skips_generation(tmp_path: Path) -> None:
    config = GatewayConfig(
        database_url=f"sqlite:///{tmp_path / 'mk.db'}", master_key="operator-set", require_pricing=False
    )
    with TestClient(create_app(config)):
        pass
    assert _stored_hashes(config.database_url) == []


def test_generated_key_authenticates_management_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixed = "otari-mk-" + "A" * 43
    monkeypatch.setattr(master_key_service, "generate_master_key", lambda: fixed)
    config = GatewayConfig(database_url=f"sqlite:///{tmp_path / 'mk.db'}", require_pricing=False)
    with TestClient(create_app(config)) as client:
        ok = client.get(f"{API_ROOT}/hosted-providers", headers={"Otari-Key": f"Bearer {fixed}"})
        assert ok.status_code == 200
        bad = client.get(f"{API_ROOT}/hosted-providers", headers={"Otari-Key": "Bearer wrong"})
        assert bad.status_code == 401


@pytest.mark.asyncio
async def test_is_valid_master_key_accepts_hash_and_plaintext() -> None:
    token = generate_master_key()
    generated_config = GatewayConfig()
    session = AsyncMock()
    session.get.return_value = RuntimeSetting(key=MASTER_KEY_HASH_KEY, value=hash_master_key(token))
    assert await deps.is_valid_master_key(token, generated_config, session) is True
    assert await deps.is_valid_master_key("otari-mk-wrong", generated_config, session) is False

    plain = GatewayConfig(master_key="literal-key")
    assert await deps.is_valid_master_key("literal-key", plain, session) is True
    assert await deps.is_valid_master_key("nope", plain, session) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("blank", ["", "   "])
async def test_blank_master_key_is_unset_and_accepts_no_empty_token(blank: str) -> None:
    config = GatewayConfig(master_key=blank)
    assert config.master_key is None
    session = AsyncMock()
    session.get.return_value = None
    assert await deps.is_valid_master_key("", config, session) is False


def test_402_message_states_cause_and_both_fixes() -> None:
    msg = no_pricing_error_detail("openai:gpt-5")
    assert "openai:gpt-5" in msg
    assert f"{API_ROOT}/pricing" in msg
    assert "default_pricing" in msg
    assert f"{API_ROOT}/settings" in msg


def test_402_message_with_default_pricing_on_names_a_typo_and_not_the_fallback() -> None:
    configure_default_pricing(True)
    msg = no_pricing_error_detail("openai:gpt5-mini")
    assert "openai:gpt5-mini" in msg
    assert "Default pricing is on" in msg
    assert "typo" in msg
    assert f"GET {API_ROOT}/models" in msg
    assert f"POST {API_ROOT}/pricing" in msg
    assert f"{API_ROOT}/settings" not in msg
    assert "enable the default-pricing fallback" not in msg
