"""The stored recommenders, held in memory for the adapter's quote.

The decisions scaffold asks the port for its price before every call,
synchronously and with no session, so the stored row cannot be read per
request. This keeps a copy per worker: loaded when the app starts, reloaded
every ``REFRESH_SECONDS``, and replaced at once by the worker that served a
write. Another worker sees a write within ``REFRESH_SECONDS``.

Which recommender answers follows the rows: the one switched on answers at
its rates. With none stored, or every row off, nothing does, and nothing
answers until the first load lands either.
"""

import asyncio
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field

from gateway.core.database import DATABASE_ERRORS, describe_database_error
from gateway.core.unit_of_work import UnitOfWork, create_unit_of_work
from gateway.log_config import logger
from gateway.models.routing import AgentRecommender
from gateway.repositories.routing import AgentRecommenderRepository
from gateway.services.routing._agent_recommender_price import RecommenderPrice
from gateway.services.secret_box import SecretBoxUnavailableError, SecretDecryptionError, decrypt_secret

REFRESH_SECONDS = 30.0


@dataclass(frozen=True)
class StoredRecommender:
    """A stored recommender that can answer: where to ask, on what key, and what one answer costs."""

    id: uuid.UUID
    provider: str
    model: str
    api_base: str | None
    price: RecommenderPrice
    api_key: str | None = field(default=None, repr=False)


_active: StoredRecommender | None = None
# Counts replacements, so a reload that read before a write cannot undo it.
_version = 0


def active_agent_recommender() -> StoredRecommender | None:
    """The recommender to answer with now, or None where none can answer."""
    return _active


def _from_row(row: AgentRecommender) -> StoredRecommender | None:
    api_key = None
    if row.encrypted_api_key is not None:
        try:
            api_key = decrypt_secret(row.encrypted_api_key)
        except (SecretDecryptionError, SecretBoxUnavailableError):
            logger.error(
                "The subagent router's key cannot be decrypted with OTARI_SECRET_KEY; no recommendation is served"
            )
            return None
    return StoredRecommender(
        id=row.id,
        provider=row.provider,
        model=row.model,
        api_base=row.api_base,
        price=RecommenderPrice(
            input_price_per_million=row.input_price_per_million, output_price_per_million=row.output_price_per_million
        ),
        api_key=api_key,
    )


def apply_agent_recommenders(rows: Sequence[AgentRecommender]) -> None:
    """Take ``rows`` as every recommender stored, as the worker that wrote them does."""
    global _active, _version  # noqa: PLW0603

    _version += 1
    enabled = next((row for row in rows if row.enabled), None)
    _active = _from_row(enabled) if enabled is not None else None


async def refresh_agent_recommenders(uow: UnitOfWork) -> None:
    """Read the rows again, unless a write replaced them while they were read."""
    version = _version
    async with uow:
        rows = await AgentRecommenderRepository(uow).list_all()
    if _version == version:
        apply_agent_recommenders(rows)


def reset_agent_recommenders() -> None:
    """Forget the rows, so nothing answers until the next load (startup and tests)."""
    global _active  # noqa: PLW0603

    _active = None


async def _refresh_on_a_session_of_its_own() -> None:
    async with create_unit_of_work() as uow:
        await refresh_agent_recommenders(uow)


async def load_agent_recommenders_at_startup() -> None:
    """Read the rows before the first request. Until a read lands, nothing answers."""
    global _active  # noqa: PLW0603

    _active = None
    try:
        await _refresh_on_a_session_of_its_own()
    except DATABASE_ERRORS as exc:
        # By class only: a database error's message can carry the row it read.
        logger.error(
            "Failed to load the subagent routers (%s); no recommendation is served until the next refresh",
            describe_database_error(exc),
        )
    except Exception:
        logger.exception("Failed to load the subagent routers; no recommendation is served until the next refresh")


async def run_agent_recommender_refresher(interval: float = REFRESH_SECONDS) -> None:
    """Reload the rows forever, so another worker's writes arrive. Every error is retried next tick."""
    while True:
        await asyncio.sleep(interval)
        try:
            await _refresh_on_a_session_of_its_own()
        except asyncio.CancelledError:
            raise
        except DATABASE_ERRORS as exc:
            logger.warning(
                "Subagent router refresh failed (%s); retrying in %ss", describe_database_error(exc), interval
            )
        except Exception:
            logger.warning("Subagent router refresh failed; retrying in %ss", interval, exc_info=True)
