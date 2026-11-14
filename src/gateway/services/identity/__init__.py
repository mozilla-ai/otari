"""The identity domain owns who a person is and how they sign in."""

from gateway.models.identity import ProviderIdentity
from gateway.services.identity._blocklist import EmailBlocklist
from gateway.services.identity._connection_service import OIDCConnectionService
from gateway.services.identity._deactivation import SelfServiceKeyDeactivationListener
from gateway.services.identity._oauth_apps import OAuthAppService
from gateway.services.identity._service import EmailBlock, IdentityService, PasswordLink, SignIn, SignInListener
from gateway.services.identity._sso_exchange import exchange_sso_code

__all__ = [
    "EmailBlock",
    "EmailBlocklist",
    "SelfServiceKeyDeactivationListener",
    "OAuthAppService",
    "IdentityService",
    "OIDCConnectionService",
    "PasswordLink",
    "ProviderIdentity",
    "SignIn",
    "SignInListener",
    "exchange_sso_code",
]
