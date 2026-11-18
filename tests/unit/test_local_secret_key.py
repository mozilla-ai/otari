"""The secret key a SQLite deployment keeps beside its database file."""

import logging
import os
import stat
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from gateway.core.config import GatewayConfig
from gateway.core.database import sqlite_database_file
from gateway.main import create_app
from gateway.services.secret_box import (
    SecretBoxUnavailableError,
    configured_secret_keys,
    decrypt_secret,
    encrypt_secret,
    forget_local_secret_key,
    generate_secret_key,
    secret_box_configured,
    secret_key_file_for,
    use_local_secret_key,
)


@pytest.fixture(autouse=True)
def _no_secret_key(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("OTARI_SECRET_KEY", raising=False)
    monkeypatch.delenv("OTARI_MASTER_KEY", raising=False)
    yield
    forget_local_secret_key()


def test_key_file_sits_beside_the_database(tmp_path: Path) -> None:
    assert secret_key_file_for(tmp_path / "otari.db") == tmp_path / "otari.db.secret-key"


def test_first_run_generates_a_private_key_file(tmp_path: Path) -> None:
    key_file = tmp_path / "otari.db.secret-key"

    use_local_secret_key(key_file, may_generate=True)

    assert stat.S_IMODE(key_file.stat().st_mode) == 0o600
    assert configured_secret_keys() == [key_file.read_text().strip()]
    assert decrypt_secret(encrypt_secret("sk-provider")) == "sk-provider"
    assert list(tmp_path.iterdir()) == [key_file]


def test_later_runs_load_the_same_key(tmp_path: Path) -> None:
    key_file = tmp_path / "otari.db.secret-key"
    use_local_secret_key(key_file, may_generate=True)
    sealed = encrypt_secret("sk-provider")
    forget_local_secret_key()
    assert secret_box_configured() is False

    use_local_secret_key(key_file, may_generate=False)

    assert decrypt_secret(sealed) == "sk-provider"


def test_environment_key_wins_and_no_file_is_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    environment_key = generate_secret_key()
    monkeypatch.setenv("OTARI_SECRET_KEY", environment_key)
    key_file = tmp_path / "otari.db.secret-key"

    use_local_secret_key(key_file, may_generate=True)

    assert not key_file.exists()
    assert configured_secret_keys() == [environment_key]


def test_environment_key_wins_over_an_existing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key_file = tmp_path / "otari.db.secret-key"
    use_local_secret_key(key_file, may_generate=True)
    environment_key = generate_secret_key()
    monkeypatch.setenv("OTARI_SECRET_KEY", environment_key)

    assert configured_secret_keys() == [environment_key]


def test_missing_file_is_never_replaced_when_generation_is_refused(tmp_path: Path) -> None:
    key_file = tmp_path / "otari.db.secret-key"

    with pytest.raises(SecretBoxUnavailableError, match="will not generate a new key"):
        use_local_secret_key(key_file, may_generate=False)

    assert not key_file.exists()
    assert secret_box_configured() is False


def test_a_file_that_holds_no_key_is_refused_without_echoing_it(tmp_path: Path) -> None:
    key_file = tmp_path / "otari.db.secret-key"
    key_file.write_text("not-a-fernet-key")

    with pytest.raises(SecretBoxUnavailableError) as raised:
        use_local_secret_key(key_file, may_generate=True)

    assert "not-a-fernet-key" not in str(raised.value)
    assert key_file.read_text() == "not-a-fernet-key"


def test_losing_the_race_to_create_adopts_the_winner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key_file = tmp_path / "otari.db.secret-key"
    winner = generate_secret_key()
    real_link = os.link

    def link_after_the_winner(source: str, target: str) -> None:
        key_file.write_text(winner)
        real_link(source, target)

    monkeypatch.setattr(os, "link", link_after_the_winner)

    use_local_secret_key(key_file, may_generate=True)

    assert configured_secret_keys() == [winner]
    assert list(tmp_path.iterdir()) == [key_file]


def test_unwritable_directory_leaves_the_deployment_without_a_key(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    key_file = tmp_path / "missing-directory" / "otari.db.secret-key"

    with caplog.at_level(logging.WARNING, logger="gateway"):
        use_local_secret_key(key_file, may_generate=True)

    assert secret_box_configured() is False


@pytest.mark.parametrize(
    ("database_url", "expected"),
    [
        ("sqlite:////data/otari.db", Path("/data/otari.db")),
        ("sqlite+aiosqlite:///./otari.db", Path("./otari.db")),
        ("sqlite://", None),
        ("sqlite:///:memory:", None),
        ("sqlite:///file:otari?mode=memory&uri=true", None),
        ("postgresql://otari:otari@postgres:5432/otari", None),
    ],
)
def test_only_a_sqlite_file_has_somewhere_to_keep_a_key(database_url: str, expected: Path | None) -> None:
    assert sqlite_database_file(database_url) == expected


def _boot(database_url: str) -> None:
    with TestClient(create_app(GatewayConfig(database_url=database_url, require_pricing=False))):
        assert secret_box_configured() is True


def test_a_sqlite_boot_with_no_key_generates_one_and_keeps_it(tmp_path: Path) -> None:
    database_url = f"sqlite:///{tmp_path / 'otari.db'}"
    key_file = tmp_path / "otari.db.secret-key"

    _boot(database_url)
    first = key_file.read_text()
    assert secret_box_configured() is False
    _boot(database_url)

    assert key_file.read_text() == first


def test_a_sqlite_boot_refuses_to_start_over_credentials_it_cannot_open(tmp_path: Path) -> None:
    database_url = f"sqlite:///{tmp_path / 'otari.db'}"
    key_file = tmp_path / "otari.db.secret-key"
    _boot(database_url)
    engine = create_engine(database_url)
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO sandbox_credentials (provider, encrypted_api_key, last4, updated_at) "
                "VALUES ('e2b', 'sealed', 'aled', CURRENT_TIMESTAMP)"
            )
        )
    engine.dispose()
    key_file.unlink()

    with pytest.raises(SecretBoxUnavailableError, match="holds encrypted credentials"):
        _boot(database_url)

    assert not key_file.exists()


def test_default_database_follows_the_image_default_and_yields_to_an_explicit_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OTARI_DATABASE_URL", raising=False)
    assert GatewayConfig().database_url == "sqlite:///./otari.db"

    monkeypatch.setenv("OTARI_DEFAULT_DATABASE_URL", "sqlite:////data/otari.db")

    assert GatewayConfig().database_url == "sqlite:////data/otari.db"
    assert GatewayConfig(database_url="postgresql://db/otari").database_url == "postgresql://db/otari"
