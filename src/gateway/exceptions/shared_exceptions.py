"""Errors that several domains raise, each carrying the status it renders as.

A condition that belongs to the deployment rather than to one domain has no
domain module to sit in, and lives here.
"""

from gateway.exceptions import TenancyError


class SecretBoxUnavailableTenancyError(TenancyError):
    """`OTARI_SECRET_KEY` is not configured, so a secret cannot be stored.

    Wraps `services.secret_box.SecretBoxUnavailableError` in the status-carrying
    family so the route stays thin: the underlying error carries no key
    material, and neither does this one. A 500, not the 400 a
    `TenancyValidationError` would carry: the caller sent a well-formed
    request, and a missing secret key is a deployment configuration gap the
    caller cannot fix. Blaming the client here would also keep the condition
    out of 5xx error-rate alerting, which is exactly the audience that can.

    ``stored`` names what could not be stored, so the message points at the
    surface the caller was using. It defaults to the provider credentials this
    error was written for; workspace MCP servers pass their own.
    """

    def __init__(self, stored: str = "provider credentials") -> None:
        super().__init__(f"OTARI_SECRET_KEY is not set; it is required to store {stored}")


__all__ = [
    "SecretBoxUnavailableTenancyError",
]
