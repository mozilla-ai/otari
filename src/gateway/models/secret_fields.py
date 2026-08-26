"""Masking credential-shaped entries in a free-form settings dict.

Three tables carry arbitrary operator-supplied JSON that a provider SDK is
handed as keyword arguments: ``org_provider_keys.client_args``,
``provider_credentials.client_args``, and ``search_tool_credentials.options``.
All three are places a real credential legitimately lives (standalone Bedrock
needs ``aws_secret_access_key`` in ``client_args``; ``services/
bedrock_gateway_auth.py`` explains why), and none of them may echo one back
over the API. A leaf module rather than a helper on one of the three models, so
the second and third serializers share the first's rules instead of carrying a
copy that drifts.
"""

from typing import Any

# Substrings (matched case-insensitively against a key name) that a
# credential-bearing field is expected to contain.
_SECRET_LOOKING_KEY_SUBSTRINGS = ("key", "secret", "token", "password", "authorization", "credential")
REDACTED_VALUE = "***"


def redact_secret_like_values(values: dict[str, Any] | None) -> dict[str, Any] | None:
    """Mask values whose key name looks credential-shaped; pass the rest through.

    Substring match, not an exact-name allow-list: an operator can name a
    Bedrock/vertex/custom client kwarg however any-llm expects it, so a fixed
    set of exact names would miss a variant spelling and silently leak it.
    """
    if values is None:
        return None
    return {
        key: REDACTED_VALUE if any(marker in key.lower() for marker in _SECRET_LOOKING_KEY_SUBSTRINGS) else value
        for key, value in values.items()
    }


def restore_redacted_values(
    incoming: dict[str, Any] | None,
    stored: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Keep the stored value wherever the caller echoed the mask back.

    :func:`redact_secret_like_values` masks on read, so an editor that loads a
    row, changes one field and submits the whole object sends ``***`` for the
    entries it was never shown. Taking that literally would overwrite a real
    credential with the mask, which is how the dashboard's provider form saves
    (`web/src/features/providers/ProvidersPage.tsx` renders the stored args into
    its textarea), so an entry whose submitted value is exactly the mask keeps
    what is stored under that name.

    A caller that genuinely means to store the literal string ``***`` cannot say
    so, since the two are the same bytes on the wire. Clearing the entry and
    setting it again is the way out, and losing that beats overwriting a
    credential with a placeholder.
    """
    if incoming is None:
        return None
    if not stored:
        return dict(incoming)
    return {
        key: stored[key] if value == REDACTED_VALUE and key in stored else value for key, value in incoming.items()
    }
