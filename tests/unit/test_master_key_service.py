"""Unit tests for first-run master-key bootstrap and the enriched 402 body."""

import hashlib
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from gateway.api import deps
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.main import create_app
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


def test_first_run_banner_prints_the_setup_link_and_the_key_only_inside_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    secrets_logged: list[str] = []
    monkeypatch.setattr(master_key_service, "log_secret", lambda _message, secret: secrets_logged.append(secret))
    config = GatewayConfig(database_url=f"sqlite:///{tmp_path / 'mk.db'}", require_pricing=False, port=8000)

    with caplog.at_level("WARNING", logger="gateway"), TestClient(create_app(config)):
        pass

    assert len(secrets_logged) == 1
    link = secrets_logged[0]
    assert link.startswith("http://localhost:8000/#/setup?key=otari-mk-")
    key = link.split("key=", 1)[1]
    assert hash_master_key(key) == _stored_hashes(config.database_url)[0]
    assert key not in caplog.text


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
