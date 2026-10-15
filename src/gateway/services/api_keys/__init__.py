"""The api-keys domain owns the deployment's and the members' API keys, and how a workload authenticates as one."""

from gateway.services.api_keys._federation import (
    FederatedIdentity,
    FederatedTokenVerifier,
    FederationMatch,
    FederationRefusedError,
    FederationUnavailableError,
    match_federation_rules,
    unverified_issuer,
)
from gateway.services.api_keys._secrets import declared_secret_problem, refuse_unusable_declared_secrets
from gateway.services.api_keys._service import ApiKeyService

__all__ = [
    "ApiKeyService",
    "FederatedIdentity",
    "FederatedTokenVerifier",
    "FederationMatch",
    "FederationRefusedError",
    "FederationUnavailableError",
    "declared_secret_problem",
    "match_federation_rules",
    "refuse_unusable_declared_secrets",
    "unverified_issuer",
]
