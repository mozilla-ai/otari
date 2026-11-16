"""Errors that the api-keys domain may raise."""

from gateway.exceptions import (
    TenancyConflictError,
    TenancyForbiddenError,
    TenancyNotFoundError,
    TenancyValidationError,
)
from gateway.models.api_keys import DisabledReason
from gateway.types.key_scope import KeyScope


class DeclaredKeySecretInUseError(TenancyConflictError):
    """A key config.yml declares has the secret of a key it does not declare under that name.

    Names the declared key only: the other row's name could be anyone's, and the secret is never quoted.
    """

    def __init__(self, config_name: str):
        super().__init__(
            f"the secret of declared key '{config_name}' already belongs to another key; "
            "choose a new secret or delete that key"
        )


class KeyScopeMissingError(TenancyForbiddenError):
    """The request's API key lacks the scope something the request asked for requires."""

    def __init__(self, scope: KeyScope, what: str):
        super().__init__(f"This API key lacks the '{scope.value}' scope, which {what} requires.")
        self.scope = scope

    @property
    def challenge(self) -> dict[str, str]:
        """The ``WWW-Authenticate`` header naming the scope, as RFC 6750 section 3.1 shapes it.

        NOTE: the value is quoted, which is safe only while every scope is drawn from ``KeyScope``.
        """
        return {"WWW-Authenticate": f'Bearer error="insufficient_scope", scope="{self.scope.value}"'}


class FederationIssuerNotAllowedError(TenancyValidationError):
    """A federation rule names an issuer the deployment does not list in ``federation_issuers``."""

    def __init__(self, issuer: str):
        super().__init__(f"issuer '{issuer}' is not one of this deployment's federation_issuers")


class FederationIssuerInvalidError(TenancyValidationError):
    """A member's federation rule names an issuer that is not an https URL the gateway may fetch keys from."""

    def __init__(self, issuer: str, problem: str):
        super().__init__(f"issuer '{issuer}' {problem}")


class FederationRuleLimitError(TenancyValidationError):
    """A key already holds as many federation rules as it may."""

    def __init__(self, limit: int):
        super().__init__(f"a key holds at most {limit} federation rules")


class FederationRuleExistsError(TenancyConflictError):
    """A rule naming exactly this identity already exists, on this key or another.

    Two keys sharing one identity would leave its tokens matching both, which authenticates neither.
    """

    def __init__(self) -> None:
        super().__init__("a federation rule naming this identity already exists")


class FederationRuleNotFoundError(TenancyNotFoundError):
    """The key holds no federation rule with this id."""

    def __init__(self, rule_id: str):
        super().__init__(f"Federation rule '{rule_id}' not found")


class KeyEnableRefusedError(TenancyForbiddenError):
    """The caller cannot enable a key disabled for a reason they cannot clear.

    The message names who can clear the reason, not who set it, because an
    operator-clearable reason is also set by workspace and organization removal.
    """

    def __init__(self, reason: DisabledReason) -> None:
        super().__init__(
            "This key is disabled and only an operator can enable it."
            if reason.operator_may_clear
            else "This key is disabled and cannot be enabled again. Create a new key."
        )


class EndUserSettingsOnPlainKeyError(TenancyValidationError):
    """End-user budgets or a cap were sent for a key that is not a service key."""

    def __init__(self) -> None:
        super().__init__("end_user_budget_id, end_user_budget_ids and max_end_users apply only to a service key.")


class ServiceKeyFlagFixedError(TenancyValidationError):
    """A key a member or an organization made was asked to become, or stop being, a service key."""

    def __init__(self) -> None:
        super().__init__("Whether a key is a service key is fixed when it is created.")


class OrganizationServiceKeyUncappedError(TenancyValidationError):
    """An organization's service key was asked to drop its ``max_end_users``."""

    def __init__(self) -> None:
        super().__init__("An organization's service key always has a max_end_users.")


class AppKeyOwnerInactiveError(TenancyForbiddenError):
    """The approving identity no longer has an active spend identity."""

    def __init__(self) -> None:
        super().__init__("Your spend identity has been deactivated")
