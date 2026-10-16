"""Errors that the api-keys domain may raise."""

from gateway.exceptions import TenancyConflictError


class DeclaredKeySecretInUseError(TenancyConflictError):
    """A key config.yml declares has the secret of a key it does not declare under that name.

    Names the declared key only: the other row's name could be anyone's, and the secret is never quoted.
    """

    def __init__(self, config_name: str):
        super().__init__(
            f"the secret of declared key '{config_name}' already belongs to another key; "
            "choose a new secret or delete that key"
        )
