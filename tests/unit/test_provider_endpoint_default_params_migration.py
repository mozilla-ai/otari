"""The revision that seals ``provider_endpoints.default_params``, on SQLite."""

import json
import uuid
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
_REVISION = "c3e5a7b9d1f4"
_BEFORE = "b2d4f6a8c0e3"
_AT = datetime(2026, 11, 21, tzinfo=UTC).strftime("%Y-%m-%d %H:%M:%S.%f")
_SECRET = "edge-bearer-nobody-should-read"


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
    url = f"sqlite:///{tmp_path / 'endpoints.db'}"
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


def _endpoint(engine: Engine, name: str, default_params: dict[str, object] | None) -> uuid.UUID:
    organization, workspace, endpoint = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    with engine.begin() as connection:
        connection.execute(
            text("INSERT INTO organization (id, name, slug) VALUES (:id, 'Acme', :slug)"),
            {"id": organization.hex, "slug": f"acme-{organization.hex[:6]}"},
        )
        connection.execute(
            text(
                "INSERT INTO workspace (id, organization_id, name, activation_classification, "
                "created_at, updated_at) VALUES (:id, :org, :name, 'eligible', :at, :at)"
            ),
            {"id": workspace.hex, "org": organization.hex, "name": f"W-{workspace.hex[:6]}", "at": _AT},
        )
        connection.execute(
            text(
                "INSERT INTO provider_endpoints (id, workspace_id, name, provider, api_base, default_params, "
                "created_at, updated_at) "
                "VALUES (:id, :workspace, :name, 'openai', 'https://vllm.example/v1', :defaults, :at, :at)"
            ),
            {
                "id": endpoint.hex,
                "workspace": workspace.hex,
                "name": name,
                "defaults": None if default_params is None else json.dumps(default_params),
                "at": _AT,
            },
        )
    return endpoint


def _sealed(engine: Engine, endpoint: uuid.UUID) -> str | None:
    with engine.connect() as connection:
        sealed: str | None = connection.execute(
            text("SELECT encrypted_default_params FROM provider_endpoints WHERE id = :id"), {"id": endpoint.hex}
        ).scalar_one()
    return sealed


def _columns(engine: Engine) -> set[str]:
    with engine.connect() as connection:
        return {row[1] for row in connection.execute(text("PRAGMA table_info(provider_endpoints)"))}


def test_stored_defaults_are_sealed_and_the_clear_column_is_gone(
    before: tuple[Config, Engine], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, engine = before
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    endpoint = _endpoint(engine, "edge", {"x_edge": _SECRET, "top_k": 20})

    command.upgrade(config, _REVISION)

    sealed = _sealed(engine, endpoint)
    assert sealed is not None
    assert _SECRET not in sealed
    assert decrypt_mapping(sealed) == {"x_edge": _SECRET, "top_k": 20}
    assert "default_params" not in _columns(engine)


def test_an_endpoint_with_no_defaults_seals_nothing(
    before: tuple[Config, Engine], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, engine = before
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    missing = _endpoint(engine, "plain", None)
    empty = _endpoint(engine, "empty", {})

    command.upgrade(config, _REVISION)

    assert _sealed(engine, missing) is None
    assert _sealed(engine, empty) is None


def test_defaults_with_no_key_configured_refuse_the_upgrade(before: tuple[Config, Engine]) -> None:
    config, engine = before
    _endpoint(engine, "edge", {"x_edge": _SECRET})

    with pytest.raises((RuntimeError, SecretBoxUnavailableError), match="OTARI_SECRET_KEY"):
        command.upgrade(config, _REVISION)


def test_the_downgrade_puts_the_defaults_back_in_clear(
    before: tuple[Config, Engine], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, engine = before
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    endpoint = _endpoint(engine, "edge", {"top_k": 20})
    command.upgrade(config, _REVISION)

    command.downgrade(config, _BEFORE)

    with engine.connect() as connection:
        stored = connection.execute(
            text("SELECT default_params FROM provider_endpoints WHERE id = :id"), {"id": endpoint.hex}
        ).scalar_one()
    assert json.loads(stored) == {"top_k": 20}
    assert "encrypted_default_params" not in _columns(engine)
