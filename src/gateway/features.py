"""The registry of core features.

The one list that says which features this build ships.
It is a literal tuple edited by hand: nothing is discovered and nothing registers itself on import,
so the set of loaded code is fixed at build time.
Each entry names the ``CoreFeature`` its own module declares, so nothing is built here.
Whether a listed feature runs is that feature's own ``enabled`` setting.
"""

from gateway.core.feature import CoreFeature

CORE_FEATURES: tuple[CoreFeature, ...] = ()
