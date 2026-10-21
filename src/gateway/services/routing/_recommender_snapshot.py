"""The subagent recommenders an operator added, held in memory for the recommender's quote.

The decisions scaffold asks the recommender for its price before every call,
synchronously and with no session, so the stored rows cannot be read per
request. This keeps a copy: loaded at startup, reloaded every
``REFRESH_SECONDS``, and replaced at once when the dashboard writes in this
process. Another replica sees a write within ``REFRESH_SECONDS``.

While no row has been added, ``agent_recommender_model`` in the configuration
answers. Once one has, the row switched on answers, and with every row off
nothing does. Once a load has started, nothing answers until one succeeds,
since the rows may have switched the configuration's model off. A deployment
that never loads, such as a hybrid gateway with no database, keeps the
configuration's answer.
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
from gateway.services.routing._recommender_pricing import RecommenderPrice
from gateway.services.secret_box import SecretBoxUnavailableError, SecretDecryptionError, decrypt_secret

REFRESH_SECONDS = 30.0


@dataclass(frozen=True)
class RecommenderConfig:
    """A recommender that can answer: which row it is, where to ask, on what key, and its price."""

    id: uuid.UUID
    provider: str
    model: str
    api_base: str | None
    price: RecommenderPrice
    api_key: str | None = field(default=None, repr=False)


def config_from_row(row: AgentRecommender) -> RecommenderConfig | None:
    """The row as a recommender, or None when it is switched off or its key cannot be read."""
    if not row.enabled:
        return None
    api_key = None
    if row.encrypted_api_key is not None:
        try:
            api_key = decrypt_secret(row.encrypted_api_key)
        except (SecretDecryptionError, SecretBoxUnavailableError):
            logger.error("Subagent recommender %s has a key OTARI_SECRET_KEY cannot read; it answers nothing", row.id)
            return None
    return RecommenderConfig(
        id=row.id,
        provider=row.provider,
        model=row.model,
        api_base=row.api_base,
        price=RecommenderPrice(
            input_price_per_million=row.input_price_per_million,
            output_price_per_million=row.output_price_per_million,
        ),
        api_key=api_key,
    )


class RecommenderSnapshot:
    """The recommender rows as the quote reads them, without a session."""

    def __init__(self) -> None:
        self._row_present = False
        self._stored: RecommenderConfig | None = None
        self._awaiting_load = False
        # Counts replacements, so a reload that read before a write cannot undo it.
        self._version = 0

    def await_load(self) -> None:
        """Answer with no row until the next load or write lands."""
        self._awaiting_load = True

    @property
    def defers_to_config(self) -> bool:
        """Whether the configuration's ``agent_recommender_model`` answers, because no row has been added."""
        return not self._awaiting_load and not self._row_present

    def current(self) -> RecommenderConfig | None:
        """The row switched on, or None when no row can answer."""
        return None if self._awaiting_load else self._stored

    def replace(self, rows: Sequence[AgentRecommender]) -> None:
        """Take ``rows`` as every recommender the operator has added."""
        self._version += 1
        self._awaiting_load = False
        self._row_present = bool(rows)
        enabled = next((row for row in rows if row.enabled), None)
        self._stored = config_from_row(enabled) if enabled is not None else None

    async def reload(self, uow: UnitOfWork) -> None:
        """Read the rows again, unless a write replaces them while they are read."""
        version = self._version
        async with uow:
            rows = await AgentRecommenderRepository(uow).list_all()
            if self._version == version:
                self.replace(rows)

    def reset(self) -> None:
        """Forget every row, as a process that never loaded."""
        self._version += 1
        self._row_present = False
        self._stored = None
        self._awaiting_load = False


AGENT_RECOMMENDERS = RecommenderSnapshot()
"""This process's snapshot, which the recommender reads and the recommenders' service replaces."""


async def _reload_once() -> None:
    try:
        async with create_unit_of_work() as uow:
            await AGENT_RECOMMENDERS.reload(uow)
    except DATABASE_ERRORS as exc:
        # Keeps what it had: a database blip must not switch the recommender off.
        logger.warning("Could not reload the subagent recommenders: %s", describe_database_error(exc))


async def load_agent_recommenders_at_startup() -> None:
    """Load the rows before the first request. Until a load succeeds, no row answers."""
    AGENT_RECOMMENDERS.await_load()
    await _reload_once()


async def run_agent_recommender_refresher(interval: float = REFRESH_SECONDS) -> None:
    """Reload the rows forever, so another replica's writes arrive here within ``interval``. Cancelled at shutdown."""
    while True:
        await asyncio.sleep(interval)
        await _reload_once()


def reset_agent_recommenders() -> None:
    """Drop the loaded rows, so the next app in this process starts from none."""
    AGENT_RECOMMENDERS.reset()
