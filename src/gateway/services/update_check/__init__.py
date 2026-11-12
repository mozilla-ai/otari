"""The update-check domain tells an operator about new otari releases and the security fixes in them."""

from gateway.services.update_check._service import UpdateCheckService

__all__ = ["UpdateCheckService"]
