"""The step an invitation sends an invitee to, and the ways in that step names."""

from gateway.models.tenancy import InvitationNextStep, SignInMeans
from gateway.services.tenancy.organization_service import InvitationClaimSettings, _invitation_outcome

_DEPLOYMENTS = [
    InvitationClaimSettings(terms_url=None, sign_in_providers=providers, can_deliver_links=can_deliver_links)
    for providers in ((), ("Google",))
    for can_deliver_links in (False, True)
]

_OUTCOMES = [
    _invitation_outcome(means, may_set_password=may_set_password, settings=settings)
    for means in SignInMeans
    # Only an identity carrying nothing is claimable, so the impossible pairs cost one
    # redundant case each and save the sweep from encoding that rule twice.
    for may_set_password in (False, True)
    for settings in _DEPLOYMENTS
]


def test_the_sweep_reaches_every_step() -> None:
    """Without this the rules below pass by never meeting the step they are about."""
    assert {step for step, _ in _OUTCOMES} == set(InvitationNextStep)


def test_verifying_an_address_is_only_offered_where_the_link_can_be_delivered() -> None:
    """The page names the link outright, so a step that reached no mailbox would strand the invitee."""
    assert all("password_reset" in ways_in for step, ways_in in _OUTCOMES if step is InvitationNextStep.VERIFY_EMAIL)


def test_a_step_that_names_no_ways_in_carries_none() -> None:
    assert all(ways_in == [] for step, ways_in in _OUTCOMES if not step.carries_ways_in)
