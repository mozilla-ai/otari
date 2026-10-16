"""The usage domain: the rows each request served or imported leaves behind.

``UsageService`` is its service. Most usage code is still in the old shape
(``api/routes/usage.py``, ``services/usage_admin_service.py``) and moves here
when the domain does.
"""

from gateway.repositories.usage_and_telemetry import ServedTraffic
from gateway.services.usage_and_telemetry._service import UsageService

__all__ = ["ServedTraffic", "UsageService"]
