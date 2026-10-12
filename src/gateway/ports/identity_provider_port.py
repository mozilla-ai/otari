"""How an external OAuth identity becomes an identity on this deployment.

The seam between the sign-in flow and whichever build decides who an external
identity is allowed to be here. ``ARCHITECTURE.md``'s port table names this one
"authenticating users/sign-in", and the split is narrower than that reads:
authenticating is not behind the port. Proving that the person at the browser
controls a Google or GitHub account is protocol work that does not vary by
edition, so it stays a plain service (``gateway.services.oauth_service``, on
apron-auth). What varies is the *policy* applied to the proven identity, and
that is the whole of this port.

An overlay can bind its own policy to this port without editing any file in this tree.
For example, an enterprise edition can map an OIDC directory group to an organization.

Scope is OAuth and OpenID Connect. Other federation shapes are separate
surfaces rather than this method widened: a SAML assertion is keyed on a subject
with an optional email and no email-verified signal, and SCIM is out-of-band
provisioning rather than interactive sign-in. Each arrives as its own port or
method when its adapter does.

**Stability: this interface is not frozen while Otari is pre-1.0.** The
anticipated change is named and already scheduled elsewhere:
mozilla-ai/otari-ai#1551 moves identity resolution onto apron-auth's own model,
keying on ``(provider, subject)`` rather than on an address and reading
``email_verified`` as the tri-state it really is. That work lands once, on the
platform, and arrives here with the shape it settles on; this port deliberately
does not anticipate it. Overlay authors should pin a released tag and expect
``resolve`` to gain a subject and lose the boolean.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from gateway.models.tenancy import User


class EmailAssertion(StrEnum):
    """What a provider says about the email address it returns.

    NOTE: A provider that sends no claim is ``UNASSERTED``, not ``UNVERIFIED``.
    """

    UNASSERTED = "unasserted"
    UNVERIFIED = "unverified"
    VERIFIED = "verified"

    @classmethod
    def from_claim(cls, email_verified: bool | None) -> "EmailAssertion":
        """Return the assertion for a provider's tri-state ``email_verified`` claim."""
        if email_verified is None:
            return cls.UNASSERTED
        return cls.VERIFIED if email_verified else cls.UNVERIFIED


@dataclass(frozen=True, slots=True)
class IdentityResolution:
    """The identity an OAuth sign-in resolved to.

    ``new_account`` is true when this sign-in created the account, or is the
    first way in to one nobody had claimed. The caller reports it as a signup.
    """

    identity: User
    new_account: bool


class IdentityProviderPort(Protocol):
    """Which identity here, if any, an external OAuth identity signs in as."""

    async def resolve(
        self,
        *,
        provider: str,
        email: str | None,
        full_name: str | None,
        email_verified: bool,
    ) -> IdentityResolution:
        """Return the identity this external OAuth identity signs in as, as an ``IdentityResolution``.

        ``IdentityResolution.new_account`` says whether this sign-in created the
        account or is the first way in to one nobody had claimed, which the
        caller reports as a signup.

        ``provider`` names the provider or connection and is an open string
        rather than a closed enum, because a customer's own OIDC connections are
        not enumerable by this tree.

        ``email_verified`` is the provider's assertion about ``email``, already
        collapsed to a boolean by the caller. apron-auth reports it as
        tri-state (true, false, or unasserted), and an unasserted value arrives
        here as ``False``: an adapter must be able to read this argument as "the
        provider affirmatively vouched for this address", so silence has to
        collapse to the unverified side rather than be laundered into a verified
        identity. See the module docstring for where the tri-state model lands.

        Each adapter decides whether to refuse an unknown identity or create an account for it.
        An adapter does not have to write anything.

        Raises:
            TenancyError: If this identity may not sign in to this deployment.
                The concrete subclass carries the status and the message the
                caller sees, so the sign-in route stays thin.

        """
        ...


__all__ = ["EmailAssertion", "IdentityProviderPort", "IdentityResolution"]
