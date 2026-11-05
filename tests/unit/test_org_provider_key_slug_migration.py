"""The revision that names each organization key with a custom endpoint by a slug, on SQLite."""

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, text

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_REVISION = "b3e7d1a9c5f2"
_BEFORE = "9e4b7a2c6d1f"
_AT = datetime(2026, 11, 1, tzinfo=UTC).strftime("%Y-%m-%d %H:%M:%S.%f")


def _alembic_config(database_url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", database_url)
    config.attributes["configure_logger"] = False
    return config


@pytest.fixture
def before(tmp_path: Path) -> Iterator[tuple[Config, Engine]]:
    """A SQLite database at the revision before this one."""
    url = f"sqlite:///{tmp_path / 'keys.db'}"
    config = _alembic_config(url)
    command.upgrade(config, _BEFORE)
    engine = create_engine(url)
    try:
        yield config, engine
    finally:
        engine.dispose()


def _organization(engine: Engine) -> uuid.UUID:
    organization = uuid.uuid4()
    with engine.begin() as connection:
        connection.execute(
            text("INSERT INTO organization (id, name, slug) VALUES (:id, 'Acme', :slug)"),
            {"id": organization.hex, "slug": f"acme-{organization.hex[:6]}"},
        )
    return organization


def _key(
    engine: Engine,
    organization: uuid.UUID,
    name: str,
    *,
    api_base: str | None = None,
    default: bool = False,
    offers: tuple[str, ...] = (),
) -> uuid.UUID:
    key_id = uuid.uuid4()
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO org_provider_keys (id, organization_id, provider, name, api_base, is_org_default, "
                "zero_data_retention, created_at, updated_at) "
                "VALUES (:id, :org, 'openai', :name, :api_base, :default, 0, :at, :at)"
            ),
            {
                "id": key_id.hex,
                "org": organization.hex,
                "name": name,
                "api_base": api_base,
                "default": default,
                "at": _AT,
            },
        )
        for model in offers:
            connection.execute(
                text(
                    "INSERT INTO org_provider_key_models (id, organization_id, org_provider_key_id, model, enabled, "
                    "created_at, updated_at) VALUES (:id, :org, :key, :model, 1, :at, :at)"
                ),
                {"id": uuid.uuid4().hex, "org": organization.hex, "key": key_id.hex, "model": model, "at": _AT},
            )
    return key_id


def _rate(engine: Engine, organization: uuid.UUID, model_key: str) -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO organization_model_pricing (id, organization_id, kind, model_key, "
                "input_price_per_million, output_price_per_million, pricing_tiers, unit, rates, effective_from, "
                "created_at, updated_at) VALUES (:id, :org, 'model', :key, 1, 2, '[]', 'token', "
                '\'{"input": "1", "output": "2"}\', :at, :at, :at)'
            ),
            {"id": uuid.uuid4().hex, "org": organization.hex, "key": model_key, "at": _AT},
        )


def _slugs(engine: Engine) -> dict[str, tuple[str | None, bool]]:
    with engine.connect() as connection:
        rows = connection.execute(text("SELECT name, slug, is_org_default FROM org_provider_keys")).all()
    return {name: (slug, bool(default)) for name, slug, default in rows}


def _rate_keys(engine: Engine, organization: uuid.UUID) -> list[str]:
    with engine.connect() as connection:
        return sorted(
            connection.execute(
                text("SELECT model_key FROM organization_model_pricing WHERE organization_id = :org"),
                {"org": organization.hex},
            ).scalars()
        )


def test_a_key_with_a_custom_endpoint_gets_a_slug_and_its_rates(before: tuple[Config, Engine]) -> None:
    config, engine = before
    organization = _organization(engine)
    _key(engine, organization, "Synthetic", api_base="https://api.synthetic.new/v1", default=True, offers=("hf:a",))
    _key(engine, organization, "OpenAI", default=False, offers=("gpt-4o",))
    _rate(engine, organization, "openai:hf:a")

    command.upgrade(config, _REVISION)

    assert _slugs(engine) == {"Synthetic": ("synthetic", False), "OpenAI": (None, False)}
    assert _rate_keys(engine, organization) == ["synthetic:hf:a"]


def test_a_rate_a_bare_key_still_needs_is_copied(before: tuple[Config, Engine]) -> None:
    config, engine = before
    organization = _organization(engine)
    _key(engine, organization, "Local", api_base="http://llm.internal/v1", offers=("gpt-4o",))
    _key(engine, organization, "OpenAI", offers=("gpt-4o",))
    _rate(engine, organization, "openai:gpt-4o")

    command.upgrade(config, _REVISION)

    assert _rate_keys(engine, organization) == ["local:gpt-4o", "openai:gpt-4o"]


def test_a_name_that_names_a_provider_or_repeats_gets_a_free_slug(before: tuple[Config, Engine]) -> None:
    config, engine = before
    organization = _organization(engine)
    _key(engine, organization, "OpenAI", api_base="https://a.example/v1")
    _key(engine, organization, "OpenAI ", api_base="https://b.example/v1")

    command.upgrade(config, _REVISION)

    assert sorted(slug for slug, _ in _slugs(engine).values() if slug is not None) == [
        "openai-custom",
        "openai-custom-2",
    ]


def test_every_key_offering_the_model_gets_the_rate(before: tuple[Config, Engine]) -> None:
    config, engine = before
    organization = _organization(engine)
    _key(engine, organization, "Synthetic", api_base="https://api.synthetic.new/v1", offers=("hf:a",))
    _key(engine, organization, "Backup", api_base="https://backup.example/v1", offers=("hf:a",))
    _rate(engine, organization, "openai:hf:a")

    command.upgrade(config, _REVISION)

    assert _rate_keys(engine, organization) == ["backup:hf:a", "synthetic:hf:a"]


def test_a_bare_key_offering_nothing_keeps_the_rate(before: tuple[Config, Engine]) -> None:
    """A key with no offered models serves every model of its provider."""
    config, engine = before
    organization = _organization(engine)
    _key(engine, organization, "Synthetic", api_base="https://api.synthetic.new/v1", offers=("hf:a",))
    _key(engine, organization, "OpenAI")
    _rate(engine, organization, "openai:hf:a")

    command.upgrade(config, _REVISION)

    assert _rate_keys(engine, organization) == ["openai:hf:a", "synthetic:hf:a"]


def test_a_rate_under_the_legacy_spelling_moves_too(before: tuple[Config, Engine]) -> None:
    config, engine = before
    organization = _organization(engine)
    _key(engine, organization, "Synthetic", api_base="https://api.synthetic.new/v1", offers=("hf:a",))
    _rate(engine, organization, "openai/hf:a")

    command.upgrade(config, _REVISION)

    assert _rate_keys(engine, organization) == ["synthetic:hf:a"]


def test_a_slug_steps_around_an_instance_the_deployment_stored(before: tuple[Config, Engine]) -> None:
    config, engine = before
    organization = _organization(engine)
    with engine.begin() as connection:
        connection.execute(
            text("INSERT INTO configured_provider_imports (name, imported_at) VALUES ('synthetic', :at)"),
            {"at": _AT},
        )
    _key(engine, organization, "Synthetic", api_base="https://api.synthetic.new/v1")

    command.upgrade(config, _REVISION)

    assert _slugs(engine)["Synthetic"][0] == "synthetic-custom"


def test_downgrade_returns_the_rates_to_the_provider(before: tuple[Config, Engine]) -> None:
    config, engine = before
    organization = _organization(engine)
    _key(engine, organization, "Synthetic", api_base="https://api.synthetic.new/v1", offers=("hf:a",))
    _key(engine, organization, "OpenAI", offers=("hf:a",))
    _rate(engine, organization, "openai:hf:a")
    command.upgrade(config, _REVISION)

    command.downgrade(config, _BEFORE)

    assert _rate_keys(engine, organization) == ["openai:hf:a"]
