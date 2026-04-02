"""Compatibility symbols for legacy `tests.gateway.conftest` imports."""

from dataclasses import dataclass
from pathlib import Path

from alembic import command
from alembic.config import Config

MODEL_NAME = "gemini:gemini-2.5-flash"


def _run_alembic_migrations(database_url: str) -> None:
    """Run Alembic migrations for test databases."""
    alembic_cfg = Config()
    alembic_dir = Path(__file__).resolve().parents[3] / "alembic"
    alembic_cfg.set_main_option("script_location", str(alembic_dir))
    alembic_cfg.set_main_option("sqlalchemy.url", database_url)
    alembic_cfg.attributes["configure_logger"] = False
    command.upgrade(alembic_cfg, "head")


@dataclass
class LiveServer:
    """Holds information about a running test server."""

    url: str
    api_key: str
