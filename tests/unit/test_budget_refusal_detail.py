"""The wording of a budget refusal, which tells a spent allowance from a hold that does not fit."""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from gateway.core.error_codes import RefusalException
from gateway.models.budgets import Budget
from gateway.models.users import User
from gateway.services.budgets._refusals import BlockedAxis, blocked_on, end_user_fields, refusal_detail, user_subject
from gateway.services.budgets._reservations import reserve_budget


def test_user_subject_names_an_end_user_by_external_id() -> None:
    assert user_subject("u") == "User 'u'"
    assert end_user_fields(None) == {}
    assert end_user_fields("alice") == {"end_user": "alice"}
    assert user_subject("eu_a886c78298c7419caf4764794945248e", external_id="alice") == "End user 'alice'"
    blocked = BlockedAxis("budget", remaining=Decimal("0.003747"), held=Decimal("0.008224"))
    assert refusal_detail(
        user_subject("eu_a886c78298c7419caf4764794945248e", external_id="alice"),
        blocked,
        new_request=True,
        reserved_output_tokens=8192,
    ) == (
        "End user 'alice' has $0.003747 of budget remaining, less than this request's estimated cost of $0.008224. "
        "The estimate assumes 8192 output tokens; a lower output token limit on the request may let it fit"
    )


@pytest.mark.asyncio
async def test_reserve_budget_names_end_user_external_id_on_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    end_user = User(
        user_id="eu_a886c78298c7419caf4764794945248e",
        external_id="alice",
        spend=Decimal("0.046253"),
        reserved=Decimal("0"),
        current_tokens=0,
        reserved_tokens=0,
        current_requests=0,
        reserved_requests=0,
        blocked=False,
    )
    budget = Budget(budget_id="b-1", max_budget=Decimal("0.05"))
    monkeypatch.setattr(
        "gateway.services.budgets._reservations._active_user_with_budget",
        AsyncMock(return_value=(end_user, budget)),
    )
    monkeypatch.setattr(
        "gateway.services.budgets._reservations.get_active_user",
        AsyncMock(return_value=end_user),
    )
    db = AsyncMock()
    update_result = MagicMock()
    update_result.rowcount = 0
    db.execute = AsyncMock(return_value=update_result)

    with pytest.raises(HTTPException) as exc_info:
        await reserve_budget(
            db,
            user_id="eu_a886c78298c7419caf4764794945248e",
            estimate=Decimal("0.008224"),
            reserved_output_tokens=8192,
        )

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == (
        "End user 'alice' has $0.003747 of budget remaining, less than this request's estimated cost of $0.008224. "
        "The estimate assumes 8192 output tokens; a lower output token limit on the request may let it fit"
    )
    assert isinstance(exc_info.value, RefusalException)
    assert exc_info.value.fields == {"end_user": "alice"}

    blocked_end_user = User(
        user_id="eu_a886c78298c7419caf4764794945248e",
        external_id="alice",
        blocked=True,
    )
    monkeypatch.setattr(
        "gateway.services.budgets._reservations._active_user_with_budget",
        AsyncMock(return_value=(blocked_end_user, budget)),
    )
    with pytest.raises(HTTPException) as blocked_exc:
        await reserve_budget(db, user_id="eu_a886c78298c7419caf4764794945248e", estimate=Decimal("0.001"))
    assert blocked_exc.value.status_code == 403
    assert blocked_exc.value.detail == "End user 'alice' is blocked"
    assert isinstance(blocked_exc.value, RefusalException)
    assert blocked_exc.value.fields == {"end_user": "alice"}


def test_a_spent_allowance_keeps_its_wording() -> None:
    blocked = blocked_on(
        "budget", committed=Decimal("1.0"), cap=Decimal("1.0"), wanted=Decimal("0.1"), new_request=True
    )

    assert blocked == BlockedAxis("budget")
    assert refusal_detail("User 'u'", blocked, new_request=True) == "User 'u' has exceeded budget limit"


def test_a_hold_that_does_not_fit_carries_both_amounts() -> None:
    blocked = blocked_on(
        "budget", committed=Decimal("0"), cap=Decimal("0.0015"), wanted=Decimal("0.0021"), new_request=True
    )

    assert blocked == BlockedAxis("budget", remaining=Decimal("0.0015"), held=Decimal("0.0021"), exceeds_cap=True)
    assert refusal_detail("Organization", blocked, new_request=True) == (
        "Organization has $0.0015 of budget remaining, less than this request's estimated cost of $0.0021"
    )


def test_the_amounts_never_round_to_equal() -> None:
    """The estimate rounds up and the remainder down, so the refusal never reads "$X, less than $X"."""
    blocked = BlockedAxis("budget", remaining=Decimal("0.0000014"), held=Decimal("0.0000016"))

    assert refusal_detail("User 'u'", blocked, new_request=True) == (
        "User 'u' has $0.000001 of budget remaining, less than this request's estimated cost of $0.000002"
    )


def test_the_output_allowance_is_named_only_when_known() -> None:
    blocked = BlockedAxis("budget", remaining=Decimal("1"), held=Decimal("2"))

    assert "output tokens" not in refusal_detail("User 'u'", blocked, new_request=True, reserved_output_tokens=0)
    assert "output tokens" not in refusal_detail("User 'u'", blocked, new_request=True)
    assert refusal_detail("User 'u'", blocked, new_request=True, reserved_output_tokens=512).endswith(
        ". The estimate assumes 512 output tokens; a lower output token limit on the request may let it fit"
    )


def test_a_token_axis_keeps_its_wording_even_when_the_hold_is_what_does_not_fit() -> None:
    blocked = blocked_on("token", committed=Decimal(900), cap=Decimal(1000), wanted=Decimal(200), new_request=True)

    assert blocked == BlockedAxis("token")
    assert refusal_detail("Workspace", blocked, new_request=True) == "Workspace has exceeded token limit"


def test_an_axis_with_room_admits() -> None:
    assert blocked_on("budget", committed=Decimal(0), cap=Decimal(1), wanted=Decimal(1), new_request=True) is None
