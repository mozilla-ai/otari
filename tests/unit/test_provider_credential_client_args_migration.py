"""The revision that seals ``provider_credentials.client_args``, on SQLite."""

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, text

from gateway.services.secret_box import (
    SecretBoxUnavailableError,
    decrypt_mapping,
    forget_local_secret_key,
    generate_secret_key,
)

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_REVISION = "b2d4f6a8c0e3"
_BEFORE = "a1c3e5b7d9f2"
_AT = datetime(2026, 11, 21, tzinfo=UTC).strftime("%Y-%m-%d %H:%M:%S.%f")
_SECRET = "wJalrXUtnFEMI-nobody-should-read"


@pytest.fixture(autouse=True)
def _no_key_left_loaded() -> Iterator[None]:
    """Keep a key one test loads from a file out of the next one."""
    yield
    forget_local_secret_key()


@pytest.fixture
def before(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Config, Engine]]:
    """A SQLite database at the revision before this one, with no key configured."""
    monkeypatch.delenv("OTARI_SECRET_KEY", raising=False)
    monkeypatch.delenv("GATEWAY_SECRET_KEY", raising=False)
    url = f"sqlite:///{tmp_path / 'providers.db'}"
    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", url)
    config.attributes["configure_logger"] = False
    command.upgrade(config, _BEFORE)
    engine = create_engine(url)
    try:
        yield config, engine
    finally:
        engine.dispose()


def _credential(engine: Engine, instance: str, client_args: dict[str, object]) -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO provider_credentials (instance, provider_type, client_args, created_at, updated_at) "
                "VALUES (:instance, 'bedrock', :client_args, :at, :at)"
            ),
            {"instance": instance, "client_args": json.dumps(client_args), "at": _AT},
        )


def _sealed(engine: Engine, instance: str) -> str | None:
    with engine.connect() as connection:
        sealed: str | None = connection.execute(
            text("SELECT encrypted_client_args FROM provider_credentials WHERE instance = :instance"),
            {"instance": instance},
        ).scalar_one()
    return sealed


def _columns(engine: Engine) -> set[str]:
    with engine.connect() as connection:
        return {row[1] for row in connection.execute(text("PRAGMA table_info(provider_credentials)"))}


def test_stored_settings_are_sealed_and_the_clear_column_is_gone(
    before: tuple[Config, Engine], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, engine = before
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    _credential(engine, "bedrock", {"region_name": "us-east-1", "aws_secret_access_key": _SECRET})

    command.upgrade(config, _REVISION)

    sealed = _sealed(engine, "bedrock")
    assert sealed is not None
    assert _SECRET not in sealed
    assert decrypt_mapping(sealed) == {"region_name": "us-east-1", "aws_secret_access_key": _SECRET}
    assert "client_args" not in _columns(engine)


def test_an_empty_settings_object_seals_nothing(before: tuple[Config, Engine]) -> None:
    """The column was NOT NULL with an empty object as its resting value, which is not a secret."""
    config, engine = before
    _credential(engine, "openai", {})

    command.upgrade(config, _REVISION)

    assert _sealed(engine, "openai") is None


def test_settings_with_no_key_configured_refuse_the_upgrade(before: tuple[Config, Engine]) -> None:
    config, engine = before
    _credential(engine, "bedrock", {"aws_secret_access_key": _SECRET})

    with pytest.raises((RuntimeError, SecretBoxUnavailableError), match="OTARI_SECRET_KEY"):
        command.upgrade(config, _REVISION)


def test_the_downgrade_puts_the_settings_back_in_a_not_null_column(
    before: tuple[Config, Engine], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, engine = before
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    _credential(engine, "bedrock", {"timeout": 9})
    _credential(engine, "openai", {})
    command.upgrade(config, _REVISION)

    command.downgrade(config, _BEFORE)

    with engine.connect() as connection:
        stored: dict[str, str] = dict(
            connection.execute(text("SELECT instance, client_args FROM provider_credentials")).all()
        )
    assert json.loads(stored["bedrock"]) == {"timeout": 9}
    assert json.loads(stored["openai"]) == {}
    assert "encrypted_client_args" not in _columns(engine)
