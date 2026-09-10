"""Entitlement adapter granting the base build's capability set plus what is installed.

Satisfies :class:`gateway.ports.entitlement_port.EntitlementPort` with the
capabilities Otari's base build ships and the ones a bootstrap installed into
this process by contributing a router for them. Whether a capability is
*installed* and whether the deployment is *entitled* to it are two axes; with no
real resolver bound, installed is the only evidence this build has, so it
grants what it mounts. An overlay that binds a real resolver replaces this
adapter entirely, so the entitlement axis is its own there.
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
    """Entitlement adapter granting the base capability set plus ``installed``.

    ``installed`` is what the composition root's contributions name. The set is
    per deployment, so the request's database session is unused.
    """

    def __init__(self, session: AsyncSession | None, installed: frozenset[str] = frozenset()) -> None:
        # The session is accepted to match the container's per-request factory
        # and unused, because the answer is static per deployment.
        del session
        self._installed = installed

    async def entitlements(self) -> set[str]:
        return set(BASE_CAPABILITIES | self._installed)
