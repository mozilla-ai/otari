"""Which secrets a declared API key may carry."""

import re
from collections.abc import Mapping

from gateway.auth.models import API_KEY_PREFIX, MIN_API_KEY_LENGTH
from gateway.core.settings.api_keys import SECRET_RECIPE, ApiKeyConfig

_SECRET_BODY = re.compile(r"^[A-Za-z0-9_-]+$")

# Distinct characters a declared secret holds after its prefix. A random
# ``token_urlsafe(48)`` holds about forty of the 64 it draws from, so this only
# refuses a value somebody typed, such as a repeated character or a word.
_MIN_DISTINCT_SECRET_CHARACTERS = 16


def declared_secret_problem(secret: str) -> str | None:
    """Say what is wrong with a declared secret, or None when it is usable.

    The open-source key format: the ``tk-`` prefix, at least
    ``MIN_API_KEY_LENGTH`` characters, URL-safe characters only. A key in that
    format is checked locally by every key format a build binds, because a
    format refuses only a key that claims its own shape and fails it. The answer
    never quotes the secret.
    """
    if not secret:
        return "is empty"
    if secret != secret.strip():
        return "has leading or trailing whitespace"
    if not secret.startswith(API_KEY_PREFIX):
        return f"must start with '{API_KEY_PREFIX}'"
    if len(secret) < MIN_API_KEY_LENGTH:
        return f"must be at least {MIN_API_KEY_LENGTH} characters long"
    body = secret[len(API_KEY_PREFIX) :]
    if not _SECRET_BODY.match(body):
        return f"may hold only letters, digits, '-' and '_' after '{API_KEY_PREFIX}'"
    if len(set(body)) < _MIN_DISTINCT_SECRET_CHARACTERS:
        return "looks like a placeholder rather than a random value"
    return None


def refuse_unusable_declared_secrets(api_keys: Mapping[str, ApiKeyConfig], master_key: str | None) -> None:
    """Refuse declared keys whose secrets cannot be used, naming each key and never its secret.

    Raises:
        ValueError: every reason found, joined.
    """
    problems: list[str] = []
    owners: dict[str, str] = {}
    for name, key in api_keys.items():
        secret = key.secret.get_secret_value()
        if (problem := declared_secret_problem(secret)) is not None:
            problems.append(f"api_keys.{name}.secret {problem}; generate one with {SECRET_RECIPE}")
            continue
        if master_key is not None and secret == master_key:
            problems.append(f"api_keys.{name}.secret must differ from the master key")
        if (other := owners.setdefault(secret, name)) != name:
            problems.append(f"api_keys.{name}.secret is the same as api_keys.{other}.secret")
    if problems:
        raise ValueError("; ".join(problems))
