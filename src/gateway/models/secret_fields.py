"""Masking credential-shaped entries in a free-form settings dict.

Four tables carry arbitrary operator-supplied JSON that something downstream is
handed as keyword arguments: ``org_provider_keys.client_args``,
``provider_credentials.client_args`` and ``search_tool_credentials.options``,
which reach a provider SDK, and ``organization_guardrails.validate_kwargs``,
which reaches the guardrails service. All four are places a real credential
legitimately lives (standalone Bedrock needs ``aws_secret_access_key`` in
``client_args``, ``services/bedrock_gateway_auth.py`` explains why; a guardrail
class can take its vendor key as a parameter), and none of them may echo one
back over the API. A leaf module rather than a helper on one of the models, so
the later serializers share the first's rules instead of carrying a copy that
drifts.

Masking keys off the entry's *name*, which is what lets it hold for a guardrail
parameter whose catalog entry cannot be read at all: the guardrails service is
what says which parameters are secret, and the mask has to apply while it is
down.
"""

from typing import Any, cast

from gateway.exceptions.shared_exceptions import UnresolvedRedactionError

# Substrings (matched case-insensitively against a key name) that a
# credential-bearing field is expected to contain.
_SECRET_LOOKING_KEY_SUBSTRINGS = ("key", "secret", "token", "password", "authorization", "credential")
REDACTED_VALUE = "***"
# Past this, a nested value is masked wholesale rather than walked. See _redact_node.
_MAX_NESTING_DEPTH = 16


def _looks_secret(key: str) -> bool:
    return any(marker in key.lower() for marker in _SECRET_LOOKING_KEY_SUBSTRINGS)


def _redact_node(node: Any, depth: int) -> Any:
    """Mask a value in place in the tree; containers recurse, leaves pass through."""
    if depth >= _MAX_NESTING_DEPTH:
        # Fail closed. Nothing legitimate nests this deep in a kwargs blob, and
        # the alternative to masking is either a RecursionError turning a read
        # into a 500 or a depth the masking never reaches.
        return REDACTED_VALUE
    if isinstance(node, dict):
        return {
            key: REDACTED_VALUE if _looks_secret(str(key)) else _redact_node(value, depth + 1)
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [_redact_node(item, depth + 1) for item in node]
    return node


def redact_secret_like_values(values: dict[str, Any] | None) -> dict[str, Any] | None:
    """Mask values whose key name looks credential-shaped; pass the rest through.

    Substring match, not an exact-name allow-list: an operator can name a
    Bedrock/vertex/custom client kwarg however any-llm expects it, so a fixed
    set of exact names would miss a variant spelling and silently leak it.

    Nested objects and lists are walked, because none of the four columns
    constrains its shape and a credential one level down was returned in clear
    (otari#1125). A matching key masks its value WHOLE, dict or list included:
    the same thing a matching top-level key has always done to a non-scalar, so
    depth 0 behaves exactly as before.

    Lists are walked but their bare elements are never masked: an element has no
    key name to match on, and masking one on its value would be a guess. Only a
    mapping inside a list can carry a masked entry. :func:`restore_redacted_values`
    depends on that; see its own note.
    """
    if values is None:
        return None
    # The walker returns whatever shape it was given, and it was given a dict.
    # The depth bound cannot fire at the root, so this is not a `dict | str`.
    return cast(dict[str, Any], _redact_node(values, 0))


def _restore_node(incoming: Any, stored: Any, depth: int) -> Any:
    """Prefer the stored value wherever the caller echoed the mask back, at any depth."""
    if depth >= _MAX_NESTING_DEPTH:
        # Masking never reaches this deep: a subtree at the bound was masked whole
        # as its parent's value, and the parent restores it by key or by list.
        return incoming
    if isinstance(incoming, dict):
        stored_map = stored if isinstance(stored, dict) else {}
        return {
            key: stored_map[key]
            if value == REDACTED_VALUE and key in stored_map
            else _restore_node(value, stored_map.get(key), depth + 1)
            for key, value in incoming.items()
        }
    if isinstance(incoming, list):
        return _restore_list(incoming, stored if isinstance(stored, list) else [], depth)
    return incoming


def _carries_mask(node: Any, depth: int) -> bool:
    """Whether a list entry still holds a mask that masking could have put there."""
    if depth > _MAX_NESTING_DEPTH:
        return False
    if isinstance(node, dict):
        return any(value == REDACTED_VALUE or _carries_mask(value, depth + 1) for value in node.values())
    if isinstance(node, list):
        return any(_carries_mask(item, depth + 1) for item in node)
    # Masking puts a bare element in a list only at the bound.
    return node == REDACTED_VALUE and depth >= _MAX_NESTING_DEPTH


def _restore_list(incoming: list[Any], stored: list[Any], depth: int) -> list[Any]:
    """Restore a list only where it is unambiguous which stored entry an element is.

    A list element has no key, so the one thing that identifies it is content:
    an entry the caller left alone equals its stored entry's masked form,
    wherever it moved. An unchanged list comes back as stored, so a plain
    load-and-save never loses a credential, however alike its entries look.

    An edited entry still carrying the mask cannot be paired with a stored
    entry without guessing, and a wrong guess hands one entry another's
    credential. It is refused instead, so the caller re-enters that entry's
    credential rather than storing ``***`` over it.
    """
    masked = [_redact_node(item, depth + 1) for item in stored]
    if masked == incoming:
        return list(stored)
    out: list[Any] = []
    for item in incoming:
        matches = [pos for pos, form in enumerate(masked) if form == item]
        if matches and all(stored[pos] == stored[matches[0]] for pos in matches):
            out.append(stored[matches[0]])
        elif _carries_mask(item, depth + 1):
            raise UnresolvedRedactionError
        else:
            out.append(item)
    return out


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

    Walks nested objects and lists IN STEP with the masking, and that pairing is
    the whole point rather than a detail: the moment the mask reaches a nested
    entry, a restore that still walks one level writes ``***`` into the database
    where a credential used to be on the next PATCH. Worse than the leak it was
    fixing (otari#1125).

    A list element has no key, so lists restore by content instead, and an
    edited list entry that still carries the mask raises
    :class:`UnresolvedRedactionError`; see :func:`_restore_list`.
    """
    if incoming is None:
        return None
    if not stored:
        return dict(incoming)
    return cast(dict[str, Any], _restore_node(incoming, stored, 0))
