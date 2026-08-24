"""Which capabilities a deployment is entitled to.

The licensing axis of ``ARCHITECTURE.md``, "Deployment and entitlements": is
this capability enabled for this deployment at all? Scoped per deployment,
never per user, and composed with (never merged into) the surface axis that
``GET /v1/bootstrap`` answers.

The core adapter grants the base build's capability set and reports every
overlay-only capability as absent; an overlay binds a real resolver behind the
same port. The dashboard resolves the same axis in the browser, from
``BASE_CAPABILITIES`` in ``web/src/shared/hooks/useEntitlements.tsx``, and the
two answers are meant to agree: hiding a link is not authorization, so a router
an overlay contributes is gated on this port server-side as well
(``gateway.api.deps.require_capability``).
"""

from typing import Protocol


class EntitlementPort(Protocol):
    """The capability names this deployment is licensed for."""

    async def entitlements(self) -> set[str]:
        """Return the capability names the current deployment is entitled to.

        Takes no subject, so it answers at deployment grain; an adapter that
        needs to resolve more finely would extend the seam with the context it
        needs.

        Capability names are open-ended domain strings rather than a closed
        enum: an overlay ships capabilities the base does not, so the base
        cannot enumerate them.
        """
        ...
