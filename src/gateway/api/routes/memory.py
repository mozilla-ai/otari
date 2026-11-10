"""Agent memory, a core feature switched by ``memory_enabled``.

Listed in ``CORE_FEATURES`` as ``FEATURE`` below. Its MCP endpoint, where agents
search, propose, confirm, flag and forget, mounts on ``router``; until then the
feature has no routes, and switching it on changes nothing a caller can reach.
"""

from fastapi import APIRouter, Depends

from gateway.api.deps import verify_api_key
from gateway.core.feature import CoreFeature
from gateway.services.memory import MemoryService

# Agents reach memory with an API key, and its workspace and member are the ones that key names.
router = APIRouter(prefix="/memory", tags=["memory"], dependencies=[Depends(verify_api_key)])

# The service the routes on ``router`` are built on.
Service = MemoryService

FEATURE = CoreFeature(
    name="memory", surface=None, enabled=lambda config: config.memory_enabled, routers=lambda config: ()
)
