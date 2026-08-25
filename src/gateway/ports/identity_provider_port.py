"""How an external OAuth identity becomes an identity on this deployment.

The seam between the sign-in flow and whichever build decides who an external
identity is allowed to be here. ``ARCHITECTURE.md``'s port table names this one
"authenticating users/sign-in", and the split is narrower than that reads:
authenticating is not behind the port. Proving that the person at the browser
controls a Google or GitHub account is protocol work that does not vary by
edition, so it stays a plain service (``gateway.services.oauth_service``, on
apron-auth). What varies is the *policy* applied to the proven identity, and
that is the whole of this port.

The core adapter enforces the base build's policy, which is the roster:
an OAuth identity signs in as an account an operator already put here, and never
creates one. An overlay binds a different policy behind the same interface, a
hosted edition that provisions on first sight, or an enterprise OIDC connection
that maps a directory group onto an organization, without editing any file in
this tree.

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

from typing import Protocol

from gateway.models.tenancy import User


class IdentityProviderPort(Protocol):
    """Which identity here, if any, an external OAuth identity signs in as."""

    async def resolve(
        self,
        *,
        provider: str,
        email: str | None,
        full_name: str | None,
        email_verified: bool,
    ) -> User:
        """Return the identity this external OAuth identity signs in as.

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

        An adapter decides for itself what to do with an identity it does not
        recognize, and both answers are legitimate: refuse (the core), or
        provision (a hosted edition). Nothing here is required to write.

        Raises:
            TenancyError: If this identity may not sign in to this deployment.
                The concrete subclass carries the status and the message the
                caller sees, so the sign-in route stays thin.

        """
        ...


__all__ = ["IdentityProviderPort"]
