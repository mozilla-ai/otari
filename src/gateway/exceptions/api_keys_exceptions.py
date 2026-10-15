"""Errors that the api-keys domain may raise."""

from fastapi import status

from gateway.exceptions import (
    TenancyConflictError,
    TenancyError,
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


class EndUserTokensDisabledError(TenancyForbiddenError):
    """Signed end-user tokens were asked for on a deployment that has them off."""

    def __init__(self) -> None:
        super().__init__("Signed end-user tokens are off on this deployment; an operator sets end_user_tokens_enabled")


class SigningKeysNotAllowedError(TenancyForbiddenError):
    """A key that may not mint keys for its end users was asked to register a signing key."""

    def __init__(self) -> None:
        super().__init__(
            "This service key may not sign end-user tokens; an operator sets its max_child_ttl_sec to allow it"
        )


class SigningKeyLimitError(TenancyConflictError):
    """A service key already holds as many signing keys as it may."""

    def __init__(self, limit: int) -> None:
        super().__init__(f"A service key holds at most {limit} signing keys; delete one first")


class SigningKeyInvalidError(TenancyValidationError):
    """A signing key was not an Ed25519 public key in PEM form."""

    def __init__(self) -> None:
        super().__init__("public_key must be an Ed25519 public key, PEM-encoded SubjectPublicKeyInfo")


class SigningKeyNotFoundError(TenancyNotFoundError):
    """No signing key of this service key has this kid."""

    def __init__(self, kid: str) -> None:
        super().__init__(f"Signing key '{kid}' not found")


class EndUserTokenRefusedError(TenancyError):
    """An end-user token did not authenticate.

    The message is the one an unknown API key gets, whatever the cause, so a holder cannot probe which check failed;
    ``reason`` names it for the auth-failure metric alone.
    """

    status_code = status.HTTP_401_UNAUTHORIZED

    def __init__(self, reason: str) -> None:
        super().__init__("Invalid API key")
        self.reason = reason
