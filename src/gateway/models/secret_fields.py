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
        # The bound is the one place a BARE list element gets masked: nothing
        # else masks an element, because an element has no key to match on. The
        # dict branch below restores a masked value by its key, and a list has
        # no key, so without this a list sitting exactly at the bound came back
        # as ``***`` and an unchanged PATCH wrote the mask over the credential.
        # With nothing stored underneath there is nothing to prefer, so the mask
        # is kept as itself rather than returned as the missing value; that is
        # the answer the dict branch already gives a key it cannot find.
        if incoming == REDACTED_VALUE and stored is not None:
            return stored
        return incoming
    if isinstance(incoming, dict):
        stored_map = stored if isinstance(stored, dict) else {}
        out: dict[Any, Any] = {}
        for key, value in incoming.items():
            if value == REDACTED_VALUE and key in stored_map:
                out[key] = stored_map[key]
            else:
                out[key] = _restore_node(value, stored_map.get(key), depth + 1)
        return out
    if isinstance(incoming, list):
        return _restore_list(incoming, stored if isinstance(stored, list) else [], depth)
    return incoming


def _unchanged_fields(item: Any, form: Any) -> int:
    """Count the non-secret fields an entry still shares with a stored entry's masked form.

    Secret fields are left out: every entry echoes them as the same mask, so
    they would make any two entries look related.
    """
    if not isinstance(item, dict) or not isinstance(form, dict):
        return 0
    return sum(1 for key, value in item.items() if not _looks_secret(key) and key in form and form[key] == value)


def _only_best(scores: dict[int, int]) -> int | None:
    """Return the key with the highest score, or None when that score is 0 or shared."""
    if not scores:
        return None
    top = max(scores.values())
    leaders = [key for key, score in scores.items() if score == top]
    return leaders[0] if top > 0 and len(leaders) == 1 else None


def _restore_list(incoming: list[Any], stored: list[Any], depth: int) -> list[Any]:
    """Pair each element with the stored element it IS, then restore within it.

    An element has no key, so it is identified by content: it is the stored
    element whose masked form it equals, which is exactly what an entry the
    caller did not edit looks like, wherever it moved in the list. Pairing by
    index instead handed each entry whatever credential used to sit at its
    position, so reordering two headers swapped their tokens.

    An edited entry matches nothing. It still keeps its stored credential when
    it is the only unmatched entry, the only unclaimed stored entry sits at the
    same index, and the list kept its length: that is an in-place edit, and
    nothing else fits. With more than one edited entry, each is paired by the
    fields the caller left alone: with the stored entry it shares the most of
    them with, when each is the other's only best match. That holds whether or
    not the entries also moved, which their index cannot tell. Anything less
    certain keeps the mask as submitted rather than guess: a tie, an entry that
    shares no unchanged field, and entries that mask to the same thing but hold
    different values.

    A bare element is masked only at the depth bound, where it has no content to
    be identified by, so it keeps its position while the length is unchanged.
    """
    same_length = len(stored) == len(incoming)
    masked = [_redact_node(item, depth + 1) for item in stored]
    paired: dict[int, int] = {}
    claimed: set[int] = set()
    unmatched: list[int] = []
    for index, item in enumerate(incoming):
        if not isinstance(item, (dict, list)):
            continue
        candidates = [pos for pos, form in enumerate(masked) if form == item]
        if candidates and all(stored[pos] == stored[candidates[0]] for pos in candidates):
            paired[index] = candidates[0]
            claimed.update(candidates)
        else:
            unmatched.append(index)
    unclaimed = [pos for pos, item in enumerate(stored) if pos not in claimed and isinstance(item, (dict, list))]
    if same_length and len(unmatched) == 1 and unclaimed == unmatched:
        paired[unmatched[0]] = unmatched[0]
    else:
        shared = {
            (index, pos): _unchanged_fields(incoming[index], masked[pos]) for index in unmatched for pos in unclaimed
        }
        for index in unmatched:
            best = _only_best({pos: shared[index, pos] for pos in unclaimed})
            # Mutual, so one stored credential is never handed to two entries.
            if best is not None and _only_best({other: shared[other, best] for other in unmatched}) == index:
                paired[index] = best

    out: list[Any] = []
    for index, item in enumerate(incoming):
        if index in paired:
            out.append(_restore_node(item, stored[paired[index]], depth + 1))
        elif not isinstance(item, (dict, list)) and same_length:
            out.append(_restore_node(item, stored[index], depth + 1))
        else:
            out.append(_restore_node(item, None, depth + 1))
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

    A bare ``***`` inside a LIST is taken literally, because masking never puts
    one there (list elements have no key to match on), so an element that looks
    like the mask came from the caller and means itself.

    Elements of a list are paired by identity, not position; see
    :func:`_restore_list`.
    """
    if incoming is None:
        return None
    if not stored:
        return dict(incoming)
    return cast(dict[str, Any], _restore_node(incoming, stored, 0))
