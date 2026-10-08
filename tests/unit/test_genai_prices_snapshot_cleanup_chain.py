"""The revision that deletes the stored genai-prices snapshots, exercised on SQLite."""

import uuid
from datetime import UTC, datetime
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_REVISION = "c4e8a2d6f0b3"
_BEFORE = "b5d1f3a7c9e2"


def test_genai_prices_rows_are_deleted_and_models_dev_rows_kept(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'cleanup.db'}"
    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", url)
    config.attributes["configure_logger"] = False
    command.upgrade(config, _BEFORE)
    engine = create_engine(url)
    now = datetime(2026, 10, 8, tzinfo=UTC)
    try:
        with engine.begin() as conn:
            for source in ("genai-prices", "genai-prices-pending", "genai-prices-poll-claim", "models.dev"):
                conn.execute(
                    text("INSERT INTO pricing_snapshots (source, snapshot, updated_at) VALUES (:s, '{}', :t)"),
                    {"s": source, "t": now},
                )
            for source in ("genai-prices", "models.dev"):
                conn.execute(
                    text(
                        "INSERT INTO pricing_snapshot_history (id, source, accepted_at, accepted_by, model_count, snapshot)"
                        " VALUES (:i, :s, :t, 'schedule', 1, '{}')"
                    ),
                    {"i": uuid.uuid4().hex, "s": source, "t": now},
                )

        command.upgrade(config, _REVISION)

        with engine.connect() as conn:
            assert [r[0] for r in conn.execute(text("SELECT source FROM pricing_snapshots"))] == ["models.dev"]
            assert [r[0] for r in conn.execute(text("SELECT source FROM pricing_snapshot_history"))] == ["models.dev"]

        command.downgrade(config, _BEFORE)
        with engine.connect() as conn:
            assert conn.execute(text("SELECT count(*) FROM pricing_snapshots")).scalar_one() == 1
    finally:
        engine.dispose()
