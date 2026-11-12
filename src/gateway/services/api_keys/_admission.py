"""Whether a key may authenticate now, however its holder proved who they are."""

from datetime import UTC, datetime

from gateway.models.api_keys import APIKey


def key_refusal(api_key: APIKey, now: datetime) -> tuple[str, str] | None:
    """Why ``api_key`` may not authenticate now, as a metric reason and a public detail, or None.

    A naive ``expires_at``, as SQLite returns one, is read as UTC, which is what is written.
    """
    if not api_key.is_active:
        return "inactive_key", "API key is inactive"
    expires_at = api_key.expires_at
    if expires_at is not None and expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    if expires_at is not None and expires_at < now:
        return "expired_key", "API key has expired"
    return None
