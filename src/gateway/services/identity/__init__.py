"""The identity domain owns who a person is and how they sign in."""

from gateway.models.identity import ProviderIdentity
from gateway.services.identity._connection_clients import (
    ConnectionCredentialCheck,
    check_connection_credentials,
    client_for,
    pending_flow_provider,
)
from gateway.services.identity._connection_service import OIDCConnectionService
from gateway.services.identity._service import EmailBlock, IdentityService, PasswordLink, SignIn, SignInListener

__all__ = [
    "ConnectionCredentialCheck",
    "EmailBlock",
    "IdentityService",
    "OIDCConnectionService",
    "PasswordLink",
    "ProviderIdentity",
    "SignIn",
    "SignInListener",
    "check_connection_credentials",
    "client_for",
    "pending_flow_provider",
]
