"""Errors that the api-keys domain may raise."""

from gateway.exceptions import (
    TenancyConflictError,
    TenancyForbiddenError,
    TenancyNotFoundError,
    TenancyValidationError,
)


class DeclaredKeySecretInUseError(TenancyConflictError):
    """A key config.yml declares has the secret of a key it does not declare under that name.

    Names the declared key only: the other row's name could be anyone's, and the secret is never quoted.
    """

    def __init__(self, config_name: str):
        super().__init__(
            f"the secret of declared key '{config_name}' already belongs to another key; "
            "choose a new secret or delete that key"
        )


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


class OperatorDisabledKeyError(TenancyForbiddenError):
    """A key's owner tried to switch on a key an operator switched off, which only an operator may undo."""

    def __init__(self) -> None:
        super().__init__("An operator switched this key off; ask them to switch it back on.")


class EndUserSettingsOnPlainKeyError(TenancyValidationError):
    """End-user budgets or a cap were sent for a key that is not a service key."""

    def __init__(self) -> None:
        super().__init__("end_user_budget_id, end_user_budget_ids and max_end_users apply only to a service key.")


class ServiceKeyFlagFixedError(TenancyValidationError):
    """An organization's key was asked to become, or stop being, a service key."""

    def __init__(self) -> None:
        super().__init__("Whether a key is a service key is fixed when it is created.")


class MemberKeyNotServiceKeyError(TenancyValidationError):
    """An operator tried to make a key a member created into a service key.

    Its end users would belong to the member's own user, which every organization they belong to shares.
    """

    def __init__(self) -> None:
        super().__init__("A key a member created cannot become a service key; create a service key instead.")


class OrganizationServiceKeyUncappedError(TenancyValidationError):
    """An organization's service key was asked to drop its ``max_end_users``."""

    def __init__(self) -> None:
        super().__init__("An organization's service key always has a max_end_users.")
