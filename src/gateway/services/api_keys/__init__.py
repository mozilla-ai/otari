"""The api-keys domain owns the deployment's and the members' API keys, and how a workload authenticates as one."""

from gateway.services.api_keys._access_tokens import (
    JWT_BEARER_GRANT,
    AccessTokenClaims,
    FederationGrantError,
    access_token_expiry,
    grant_names_organization,
    is_access_token,
    mint_access_token,
    read_access_token,
)
from gateway.services.api_keys._admission import key_refusal, operator_set_active
from gateway.services.api_keys._billed_user import (
    API_KEY_NO_USER_DETAIL,
    API_KEY_VALIDATION_FAILED_DETAIL,
    MASTER_KEY_USER_REQUIRED_DETAIL,
    USER_MISMATCH_DETAIL,
    names_end_user,
    resolve_user_id,
)
from gateway.services.api_keys._federation import (
    FederatedIdentity,
    FederatedTokenVerifier,
    FederationMatch,
    FederationRefusedError,
    FederationUnavailableError,
    match_federation_rules,
    rule_matches,
    unverified_issuer,
)
from gateway.services.api_keys._membership import ApiKeyMembershipListener
from gateway.services.api_keys._secrets import declared_secret_problem, refuse_unusable_declared_secrets
from gateway.services.api_keys._service import ApiKeyService, KeySummary
from gateway.services.api_keys._service_keys import (
    DEFAULT_ORGANIZATION_MAX_END_USERS,
    END_USER_SETTINGS,
    admit_organization_key,
    operator_max_end_users,
    organization_key_changes,
    refuse_service_key_flag_change,
)
from gateway.services.api_keys._token_exchange import FederationTokenExchange, OrganizationOfWorkspace

__all__ = [
    "API_KEY_NO_USER_DETAIL",
    "API_KEY_VALIDATION_FAILED_DETAIL",
    "MASTER_KEY_USER_REQUIRED_DETAIL",
    "USER_MISMATCH_DETAIL",
    "DEFAULT_ORGANIZATION_MAX_END_USERS",
    "END_USER_SETTINGS",
    "JWT_BEARER_GRANT",
    "AccessTokenClaims",
    "ApiKeyMembershipListener",
    "ApiKeyService",
    "FederatedIdentity",
    "FederatedTokenVerifier",
    "FederationGrantError",
    "FederationMatch",
    "FederationRefusedError",
    "FederationTokenExchange",
    "FederationUnavailableError",
    "KeySummary",
    "OrganizationOfWorkspace",
    "access_token_expiry",
    "admit_organization_key",
    "declared_secret_problem",
    "grant_names_organization",
    "is_access_token",
    "key_refusal",
    "match_federation_rules",
    "mint_access_token",
    "names_end_user",
    "operator_max_end_users",
    "operator_set_active",
    "organization_key_changes",
    "read_access_token",
    "refuse_service_key_flag_change",
    "refuse_unusable_declared_secrets",
    "resolve_user_id",
    "rule_matches",
    "unverified_issuer",
]
