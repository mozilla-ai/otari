"""The registry of core feature packages.

The one list that says which feature packages this build ships. It is a
literal tuple edited by hand: nothing is discovered, nothing registers itself
on import, and the set of loaded code is fixed at build time. Whether a
listed package runs is that package's own ``enabled`` setting.
"""

from gateway.core.package import CorePackage

CORE_PACKAGES: tuple[CorePackage, ...] = ()
