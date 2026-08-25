"""Entitlement adapter granting the base build's capability set.

Satisfies :class:`gateway.ports.entitlement_port.EntitlementPort` with the
fixed set of capabilities Otari's base build ships. An overlay binds a real
resolver behind the same port.
"""

from sqlalchemy.ext.asyncio import AsyncSession

# The capabilities Otari's base build ships and therefore entitles.
#
# **Empty, because nothing in the base is gated on a capability yet.** That is
# not an oversight: the one candidate is routing, and ARCHITECTURE.md marks how
# far the core base extends before an overlay adapter takes over as provisional
# and not a contributor's to assume. So the base withholds nothing and declares
# nothing, and the axis waits for a real decision instead of anticipating one.
#
# This is the server-side half of ``BASE_CAPABILITIES`` in
# ``web/src/shared/hooks/useEntitlements.tsx``; the two are meant to agree, so a
# capability the base grows is added to both at once. Leave an overlay-only
# capability (billing, for example) out of both, which is what makes a gate on
# it refuse in this build.
BASE_CAPABILITIES: frozenset[str] = frozenset()


class BaseEntitlementAdapter:
    """Entitlement adapter granting the fixed base capability set.

    The set is per deployment, so the request's database session is unused.
    """

    def __init__(self, session: AsyncSession | None) -> None:
        # Accepted to match the container's per-request factory; unused, because
        # the base entitlement set is static per deployment.
        del session

    async def entitlements(self) -> set[str]:
        return set(BASE_CAPABILITIES)
