"""The deployment settings that bound inviting, and the policy they build."""

from datetime import timedelta

import pytest

from gateway.api.deps.organizations import invitation_policy_of
from gateway.core.config import GatewayConfig


def test_the_defaults_build_a_usable_policy() -> None:
    """A deployment that configures nothing still invites."""
    policy = invitation_policy_of(GatewayConfig())

    assert policy.expires_after == timedelta(hours=168)
    assert policy.per_address_per_day == 3
    assert policy.per_day == 100
    assert policy.resend_limit == 5
    assert policy.resend_wait == timedelta(minutes=15)


def test_each_setting_reaches_the_field_it_bounds() -> None:
    """One mapping per field, so a crossed pair cannot pass by both being defaults."""
    policy = invitation_policy_of(
        GatewayConfig(
            invitation_expiry_hours=1,
            invitation_per_address_per_day=2,
            invitation_per_day=3,
            invitation_resend_limit=4,
            invitation_resend_wait_minutes=5,
        )
    )

    assert policy.expires_after == timedelta(hours=1)
    assert policy.per_address_per_day == 2
    assert policy.per_day == 3
    assert policy.resend_limit == 4
    assert policy.resend_wait == timedelta(minutes=5)


@pytest.mark.parametrize(
    "setting",
    [
        "invitation_expiry_hours",
        "invitation_per_address_per_day",
        "invitation_per_day",
        "invitation_resend_limit",
        "invitation_resend_wait_minutes",
    ],
)
def test_a_number_below_one_is_refused_at_load(setting: str) -> None:
    """``InvitationPolicy`` raises on a non-positive field, which a config load must not reach."""
    with pytest.raises(ValueError, match=setting):
        GatewayConfig(**{setting: 0})  # type: ignore[arg-type]
