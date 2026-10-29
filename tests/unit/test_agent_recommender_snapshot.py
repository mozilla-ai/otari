"""What the in-memory copy of the stored subagent routers answers with, before and after it reads the rows."""

import uuid
from collections.abc import Iterator
from contextlib import asynccontextmanager
from decimal import Decimal
from typing import Any

import pytest

from gateway.models.routing import AgentRecommender
from gateway.services.routing import (
    RecommenderPrice,
    RecommenderSource,
    active_agent_recommender,
    apply_agent_recommenders,
    load_agent_recommenders_at_startup,
    refresh_agent_recommenders,
    reset_agent_recommenders,
)
from gateway.services.routing import _agent_recommenders as snapshot_module
from gateway.services.secret_box import encrypt_secret, generate_secret_key


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    reset_agent_recommenders()
    yield
    reset_agent_recommenders()


def _row(*, enabled: bool, key: str | None = "sk-decider-wxyz", model: str = "decider") -> AgentRecommender:
    return AgentRecommender(
        id=uuid.uuid4(),
        provider="typesafe",
        model=model,
        api_base=None,
        encrypted_api_key=encrypt_secret(key) if key is not None else None,
        api_key_last4="wxyz",
        input_price_per_million=Decimal("3"),
        output_price_per_million=Decimal("0"),
        enabled=enabled,
    )


def test_with_no_rows_the_configured_model_answers() -> None:
    apply_agent_recommenders([])

    assert active_agent_recommender().source is RecommenderSource.CONFIGURED


def test_the_row_switched_on_answers_with_its_key_read_back() -> None:
    on = _row(enabled=True)
    apply_agent_recommenders([_row(enabled=False, model="other"), on])

    active = active_agent_recommender()
    assert active.source is RecommenderSource.STORED
    assert active.stored is not None
    assert (active.stored.id, active.stored.provider, active.stored.model) == (on.id, "typesafe", "decider")
    assert active.stored.api_key == "sk-decider-wxyz"
    assert active.stored.price == RecommenderPrice(input_price_per_million=Decimal("3"))
    assert "sk-decider" not in repr(active.stored)


def test_with_every_row_off_nothing_answers() -> None:
    apply_agent_recommenders([_row(enabled=False), _row(enabled=False, model="other")])

    assert active_agent_recommender() == snapshot_module._NONE


def test_a_key_that_cannot_be_read_answers_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    on = _row(enabled=True)
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())

    apply_agent_recommenders([on])

    assert active_agent_recommender().source is RecommenderSource.NONE


def test_a_reset_hands_back_to_the_configured_model() -> None:
    apply_agent_recommenders([_row(enabled=False)])

    reset_agent_recommenders()

    assert active_agent_recommender().source is RecommenderSource.CONFIGURED


@pytest.mark.asyncio
async def test_a_failed_first_load_answers_nothing_until_a_load_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    @asynccontextmanager
    async def database_down() -> Any:
        raise ConnectionError("the database is down")
        yield

    monkeypatch.setattr(snapshot_module, "create_unit_of_work", database_down)

    await load_agent_recommenders_at_startup()

    # The rows may have switched the configured model off.
    assert active_agent_recommender().source is RecommenderSource.NONE
    apply_agent_recommenders([])
    assert active_agent_recommender().source is RecommenderSource.CONFIGURED


@pytest.mark.asyncio
async def test_a_refresh_that_read_before_a_write_does_not_undo_it(monkeypatch: pytest.MonkeyPatch) -> None:
    switched_off = _row(enabled=False)

    class ReadBeforeTheWrite:
        def __init__(self, _uow: object) -> None:
            pass

        async def list_all(self) -> list[AgentRecommender]:
            # The write commits and lands while this read is in flight.
            apply_agent_recommenders([switched_off])
            return []

    class Block:
        async def __aenter__(self) -> "Block":
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

    monkeypatch.setattr(snapshot_module, "AgentRecommenderRepository", ReadBeforeTheWrite)

    await refresh_agent_recommenders(Block())  # type: ignore[arg-type]

    assert active_agent_recommender().source is RecommenderSource.NONE
