"""Alert rule request-body validation, and the one column default that must not exist.

These need no PostgreSQL, so they live here rather than in the integration
suite: the pydantic cases are pure, and the column-default case runs against an
in-memory SQLite table the way `test_tenancy_schema_chain.py` does. Both bugs
they cover reached CI as integration failures on a machine that could not run
that suite, which is the argument for having them here.

``Base`` here is the plugin's own metadata, not Otari's, which is why the
``create_all`` below builds exactly one table.
"""

import asyncio
import uuid

import pytest
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from otari_alerts.models import AlertRule, Base
from otari_alerts.service import AlertRuleCreate, AlertRuleUpdate

DESTINATION = "slack://xoxb-AAA/xoxb-BBB/xoxb-CCC/#alerts"


# --------------------------------------------------------------------------
# The create body
# --------------------------------------------------------------------------


def test_the_warning_threshold_defaults_to_eighty() -> None:
    """The default lives in the schema, not on the column. See the model comment."""
    assert AlertRuleCreate(name="R", destination=DESTINATION).warn_at_percent == 80


def test_an_explicit_null_threshold_is_kept_as_null() -> None:
    """``warn_at_percent: null`` means "alert only on refusal" and must survive.

    The schema's own default must not swallow an explicit null, or the API's
    documented way to decline early warnings would silently enable them.
    """
    body = AlertRuleCreate(name="R", destination=DESTINATION, warn_at_percent=None)
    assert body.warn_at_percent is None
    assert "warn_at_percent" in body.model_fields_set


def test_a_rule_that_could_never_fire_is_refused() -> None:
    """No warning and no exceeded alert is storable and meaningless."""
    with pytest.raises(ValidationError):
        AlertRuleCreate(
            name="Inert",
            destination=DESTINATION,
            warn_at_percent=None,
            notify_on_exceeded=False,
        )


@pytest.mark.parametrize("percent", [0, 100, 101, -1])
def test_the_warn_threshold_stays_inside_its_range(percent: int) -> None:
    """0 always fires and 100 is the refusal; neither is a warning."""
    with pytest.raises(ValidationError):
        AlertRuleCreate(name="Bad", destination=DESTINATION, warn_at_percent=percent)


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_destination_is_refused(blank: str) -> None:
    with pytest.raises(ValidationError):
        AlertRuleCreate(name="Blank", destination=blank)


def test_an_over_long_destination_is_refused() -> None:
    """The constraint is on the string, not on the field. See `AlertRuleUpdate`."""
    with pytest.raises(ValidationError):
        AlertRuleCreate(name="Long", destination="json://example.com/" + "x" * 5000)


# --------------------------------------------------------------------------
# The update body
# --------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["name", "destination", "notify_on_exceeded", "enabled"])
def test_an_explicit_null_is_refused_for_a_not_null_column(field: str) -> None:
    """A ``ValidationError``, not a ``TypeError``.

    ``max_length`` on an optional ``Field`` is applied to the whole union, and
    pydantic raises ``TypeError`` for None rather than refusing it, which
    surfaced as a 500 instead of a 422 on ``{"destination": null}``.
    """
    with pytest.raises(ValidationError):
        AlertRuleUpdate.model_validate({field: None})


def test_a_null_warn_threshold_is_a_value_on_the_update_body() -> None:
    """The one field where null means something rather than "not sent"."""
    body = AlertRuleUpdate.model_validate({"warn_at_percent": None})
    assert body.warn_at_percent is None
    assert "warn_at_percent" in body.model_fields_set


def test_an_omitted_field_is_absent_from_fields_set() -> None:
    """What the service branches on to leave a stored value alone."""
    body = AlertRuleUpdate.model_validate({"enabled": False})
    assert "warn_at_percent" not in body.model_fields_set
    assert body.destination is None


# --------------------------------------------------------------------------
# The column default that must not come back
# --------------------------------------------------------------------------


def test_an_explicit_none_threshold_is_stored_as_null() -> None:
    """A scalar column default on ``warn_at_percent`` would overwrite an explicit None.

    SQLAlchemy applies a scalar ``default=`` whenever the attribute is None at
    INSERT and cannot tell an explicit ``None`` from an omission, so
    ``default=80`` on that column stored 80 for a caller who asked for no early
    warning at all. This is the regression guard; it fails if the default
    returns.
    """

    async def _run() -> list[tuple[str, int | None]]:
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        try:
            async with engine.begin() as conn:
                await conn.run_sync(lambda c: Base.metadata.create_all(c, tables=[AlertRule.__table__]))  # type: ignore[list-item]
            factory = async_sessionmaker(engine, expire_on_commit=False)
            async with factory() as db:
                db.add(
                    AlertRule(
                        organization_id=uuid.uuid4(),
                        name="refusal only",
                        encrypted_destination="ciphertext",
                        redacted_destination="slack://***",
                        warn_at_percent=None,
                        notify_on_exceeded=True,
                        enabled=True,
                    )
                )
                await db.commit()
                rows = (await db.execute(select(AlertRule.name, AlertRule.warn_at_percent))).all()
                return [(name, warn) for name, warn in rows]
        finally:
            await engine.dispose()

    rows = asyncio.run(_run())
    assert rows == [("refusal only", None)]
