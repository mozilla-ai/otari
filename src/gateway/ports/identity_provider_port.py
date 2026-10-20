"""How an external OAuth identity becomes an identity on this deployment.

The seam between the sign-in flow and whichever build decides who an external
identity is allowed to be here. ``ARCHITECTURE.md``'s port table names this one
"authenticating users/sign-in", and the split is narrower than that reads:
authenticating is not behind the port. Proving that the person at the browser
controls a Google or GitHub account is protocol work that does not vary by
edition, so it stays a plain service (``gateway.services.oauth_service``, on
apron-auth). What varies is the *policy* applied to the proven identity, and
that is the whole of this port.

Every edition keys a sign-in on the provider's ``(provider, subject)`` pair,
and that rule is not behind the port either.
An identity already linked to an account signs in as that account without asking the port.
The port decides only which account an identity with no link signs in as, and that account gains the link.

An overlay can bind its own policy to this port without editing any file in this tree.
For example, an enterprise edition can map an OIDC directory group to an organization.

Scope is OAuth and OpenID Connect. Other federation shapes are separate
surfaces rather than this method widened: a SAML assertion is keyed on a subject
with an optional email and no email-verified signal, and SCIM is out-of-band
provisioning rather than interactive sign-in. Each arrives as its own port or
method when its adapter does.

**Stability: this interface is not frozen while Otari is pre-1.0.**
Overlay authors should pin a released tag.
"""

from dataclasses import dataclass
from typing import Protocol

from gateway.models.identity import ProviderIdentity
from gateway.models.tenancy import User


@dataclass(frozen=True, slots=True)
class IdentityResolution:
    """The identity an OAuth sign-in resolved to.

    ``new_account`` is true when this sign-in created the account, or is the
    first way in to one nobody had claimed. The caller reports it as a signup,
    and holds it to the terms the deployment publishes, so an adapter need not.
    ``password_removed`` is true when this sign-in removed a password the account
    held, so the caller can tell the person.
    """

    identity: User
    new_account: bool
    password_removed: bool = False


class IdentityProviderPort(Protocol):
    """Which account here, if any, an external OAuth identity with no link signs in as."""

    async def resolve(self, external: ProviderIdentity) -> IdentityResolution:
        """Return the account that an identity with no link here signs in as.

        ``IdentityResolution.new_account`` says whether this sign-in created the
        account or is the first way in to one nobody had claimed, which the
        caller reports as a signup.

        NOTE: No account is linked to the identity's ``(provider, subject)`` when this is called.
        The account this returns gains that link.
        An adapter must return an account only when this identity may sign in as it from then on.
        A deactivated account is refused whatever an adapter returns.
        An adapter must not pick an existing account by an email claim that a tenant can set.

        Each adapter decides whether to refuse an unknown identity or create an account for it.
        An adapter does not have to write anything.

        Raises:
            TenancyError: If this identity may not sign in to this deployment.
                The concrete subclass carries the status and the message the
                caller sees.

        """
        ...


__all__ = ["IdentityProviderPort", "IdentityResolution"]
