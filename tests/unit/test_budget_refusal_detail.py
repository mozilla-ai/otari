"""The wording of a budget refusal, which tells a spent allowance from a hold that does not fit."""

from decimal import Decimal

from gateway.services.budgets._refusals import BlockedAxis, blocked_on, refusal_detail


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
