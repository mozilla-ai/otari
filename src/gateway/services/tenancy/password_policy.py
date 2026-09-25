"""The rule a new password has to meet, wherever one is set.

Its own module so ``organization_service`` (an invitation claimed with a
password) and ``user_service`` (signup, reset, change) apply the one rule:
``user_service`` imports ``organization_service``, so the rule cannot live in
either without a cycle.
"""

from gateway.exceptions.identity_exceptions import PasswordPolicyError
from gateway.services.password_service import MAX_PASSWORD_BYTES, MIN_PASSWORD_LENGTH


def validate_new_password(password: str) -> None:
    """Refuse a password bcrypt would reject or that is too short to be one."""
    if len(password) < MIN_PASSWORD_LENGTH:
        raise PasswordPolicyError(f"A password must be at least {MIN_PASSWORD_LENGTH} characters")
    if len(password.encode()) > MAX_PASSWORD_BYTES:
        raise PasswordPolicyError(
            f"A password must be at most {MAX_PASSWORD_BYTES} bytes; "
            "accented and non-Latin characters count for more than one each"
        )


__all__ = ["validate_new_password"]
