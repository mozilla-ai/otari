"""The registry of core features.

The one list that says which features this build ships.
It is a literal tuple edited by hand: nothing is discovered and nothing registers itself on import,
so the set of loaded code is fixed at build time.
Each entry names the ``CoreFeature`` its own module declares, so nothing is built here.
Whether a listed feature runs is that feature's own ``enabled`` setting.
"""

from gateway.api.routes.feedback import FEATURE as FEEDBACK
from gateway.api.routes.health_checks import FEATURE as HEALTH_CHECKS
from gateway.api.routes.identity_apps import FEATURE as OAUTH_APPS
from gateway.api.routes.mcp_catalog import FEATURE as MCP_CATALOG
from gateway.api.routes.oauth import FEATURE as WORKLOAD_IDENTITY_EXCHANGE
from gateway.api.routes.platform_connect import FEATURE as PLATFORM_CONNECT
from gateway.api.routes.traces import FEATURE as TRACES
from gateway.api.routes.traces import RETENTION_FEATURE as TRACE_RETENTION
from gateway.api.routes.update_check import FEATURE as UPDATE_CHECK
from gateway.core.feature import CoreFeature

CORE_FEATURES: tuple[CoreFeature, ...] = (
    OAUTH_APPS,
    FEEDBACK,
    HEALTH_CHECKS,
    MCP_CATALOG,
    PLATFORM_CONNECT,
    TRACES,
    TRACE_RETENTION,
    UPDATE_CHECK,
    WORKLOAD_IDENTITY_EXCHANGE,
)
