"""``rates`` and the six columns stay one thing on a price row, whichever side a writer gives."""

import tempfile
from collections.abc import Iterator
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session, sessionmaker
from sqlmodel import SQLModel

import gateway.models  # noqa: F401  (registers every table on the shared metadata)
from gateway.models.pricing import ModelPricing, OrganizationModelPricing, rates_of
from gateway.models.tenancy import Organization
from gateway.types.price_rates import RateColumns, dump_rates, rates_from_columns

_AT = datetime(2026, 5, 20, tzinfo=UTC)
_TOKEN = {"text": {"input": {"price": 3, "unit": "token"}, "output": {"price": 15, "unit": "token"}}}
_IMAGE = {"image": {"output": {"price": 0.04, "unit": "image"}}}


@pytest.fixture
def sessions() -> Iterator[sessionmaker[Session]]:
    with tempfile.TemporaryDirectory() as directory:
        engine = create_engine(f"sqlite:///{directory}/prices.db")
        SQLModel.metadata.create_all(engine)
        try:
            yield sessionmaker(engine, expire_on_commit=False)
        finally:
            engine.dispose()


def _row(**given: Any) -> ModelPricing:
    return ModelPricing(model_key="openai:gpt-4o", effective_at=_AT, **given)


def _stored(session: Session) -> str:
    return str(session.execute(text("SELECT rates FROM model_pricing")).scalar_one())


def test_a_row_given_as_columns_gets_the_block_they_say(sessions: sessionmaker[Session]) -> None:
    with sessions() as session:
        session.add(_row(input_price_per_million=Decimal("2.5"), output_price_per_million=Decimal(15)))
        session.commit()

        expected = rates_from_columns(
            RateColumns(unit="token", input_price_per_million=Decimal("2.5"), output_price_per_million=Decimal(15))
        )
        assert _stored(session) == dump_rates(expected)
    with sessions() as session:
        row = session.execute(select(ModelPricing)).scalar_one()
        assert row.rates == expected
        assert isinstance(row.rates["text"]["input"]["price"], Decimal)


def test_a_row_given_as_a_block_gets_the_columns_it_says(sessions: sessionmaker[Session]) -> None:
    with sessions() as session:
        session.add(_row(rates=_IMAGE))
        session.commit()

    with sessions() as session:
        row = session.execute(select(ModelPricing)).scalar_one()
        assert (row.unit, row.input_price_per_million, row.output_price_per_million) == (
            "image",
            Decimal(40000),
            Decimal(0),
        )
        assert row.rates == {"image": {"output": {"price": Decimal("0.04"), "unit": "image"}}}


def test_a_writer_that_gives_both_sides_is_believed(sessions: sessionmaker[Session]) -> None:
    both = {**_TOKEN, **_IMAGE}
    with sessions() as session:
        session.add(_row(input_price_per_million=Decimal(3), output_price_per_million=Decimal(15), rates=both))
        session.commit()

    with sessions() as session:
        row = session.execute(select(ModelPricing)).scalar_one()
        assert (row.unit, row.input_price_per_million) == ("token", Decimal(3))
        assert set(row.rates) == {"text", "image"}


def test_the_side_a_writer_moves_rewrites_the_other(sessions: sessionmaker[Session]) -> None:
    with sessions() as session:
        session.add(_row(input_price_per_million=Decimal(3), output_price_per_million=Decimal(15)))
        session.commit()
        row = session.execute(select(ModelPricing)).scalar_one()

        row.input_price_per_million = Decimal(5)
        session.commit()
        assert row.rates["text"]["input"]["price"] == Decimal(5)

        row.rates = _IMAGE
        session.commit()
        assert (row.unit, row.input_price_per_million) == ("image", Decimal(40000))

        row.rates = _TOKEN
        row.unit = "request"
        row.input_price_per_million = Decimal(5000)
        session.commit()
        assert (row.unit, row.input_price_per_million, row.rates) == ("request", Decimal(5000), _TOKEN)


def test_a_block_no_column_can_hold_refuses_the_flush(sessions: sessionmaker[Session]) -> None:
    with sessions() as session:
        session.add(_row(rates={"image": {"input": {"price": 0.003, "unit": "image"}}}))
        with pytest.raises(ValueError, match="price nothing"):
            session.commit()


def test_an_organization_rate_is_kept_the_same_way(sessions: sessionmaker[Session]) -> None:
    with sessions() as session:
        organization = Organization(name="Acme", slug="acme")
        session.add(organization)
        session.flush()
        session.add(
            OrganizationModelPricing(
                organization_id=organization.id,
                model_key="openai:gpt-4o",
                input_price_per_million=Decimal(3),
                output_price_per_million=Decimal(15),
                effective_from=_AT,
            )
        )
        session.commit()

    with sessions() as session:
        row = session.execute(select(OrganizationModelPricing)).scalar_one()
        assert row.rates["text"]["output"]["price"] == Decimal(15)


def test_a_transient_row_reads_its_block_from_its_columns() -> None:
    row = _row(input_price_per_million=Decimal(3), output_price_per_million=Decimal(15))

    assert row.rates is None
    assert rates_of(row) == rates_from_columns(
        RateColumns(unit="token", input_price_per_million=Decimal(3), output_price_per_million=Decimal(15))
    )


def test_a_column_moved_alone_keeps_the_other_prices_of_the_block(sessions: sessionmaker[Session]) -> None:
    block = {
        "text": {"input": {"price": 3, "unit": "token"}, "output": {"price": 15, "unit": "token"}},
        "web_search": {"price": 10},
    }
    with sessions() as session:
        session.add(_row(rates=block))
        session.commit()
        row = session.execute(select(ModelPricing)).scalar_one()

        row.input_price_per_million = Decimal(5)
        session.commit()
        assert row.rates["text"]["input"]["price"] == Decimal(5)
        assert row.rates["web_search"] == {"price": Decimal(10)}

        row.unit = "image"
        row.input_price_per_million = Decimal(40000)
        row.output_price_per_million = Decimal(0)
        session.commit()
        assert set(row.rates) == {"image"}
