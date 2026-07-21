"""Per-key model access control: allow-lists governing catalog and inference.

An allow-list is one of:
- ``None``  -> unrestricted (the key may use any model). Backward compatible: every
  key predating this feature has ``None``.
- ``[]``    -> deny all (the key may use no model). Deliberately distinct from
  ``None`` -- callers MUST branch on ``is None``, never on truthiness, or a
  locked-down key silently gets full access.
- a non-empty list of canonical ``instance:model`` entries, with two wildcard forms:
  ``instance:*`` (every model of that instance) and ``instance:prefix*`` (a single
  trailing glob).

The matcher (:func:`is_model_allowed`) is the one place membership is decided; both
the inference gates and the catalog filter feed it the *same* canonical
``instance:model`` key so visibility and execution can never disagree.
"""

from any_llm import LLMProvider
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import GatewayConfig
from gateway.models.entities import APIKey, User
from gateway.services.alias_service import resolve_effective_alias
from gateway.services.provider_kwargs import split_selector

# Wire code used in 403 bodies so clients can branch on it programmatically.
PERMISSION_CODE = "model_not_allowed"


def effective_allowlist(api_key: APIKey | None, user: User | None = None) -> list[str] | None:
    """Resolve the allow-list that governs a request. ``None`` = unrestricted.

    The key's own list wins when set; otherwise the user default (``User``'s own
    allow-list) is inherited; otherwise unrestricted. A key with ``None`` of its
    own therefore inherits its user's default rather than being unrestricted.
    ``user`` is read defensively via ``getattr`` so a user-like object without the
    column (e.g. a test stub) still works.
    """
    key_list: list[str] | None = api_key.allowed_models if api_key is not None else None
    if key_list is not None:
        return key_list
    if user is not None:
        user_list: list[str] | None = getattr(user, "allowed_models", None)
        if user_list is not None:
            return user_list
    return None


async def resolve_request_allowlist(db: AsyncSession, api_key: APIKey | None) -> list[str] | None:
    """The effective allow-list for a request, loading the user default if needed.

    Master-key callers (``api_key is None``) are unrestricted. A key with its own
    list uses it directly, no user lookup. Only a key that inherits (its own list
    is ``None``) costs one scalar query for its user's default.
    """
    if api_key is None or api_key.allowed_models is not None:
        return effective_allowlist(api_key)
    if api_key.user_id is None:
        return None
    user_default = (
        await db.execute(select(User.allowed_models).where(User.user_id == api_key.user_id))
    ).scalar_one_or_none()
    return user_default


def _covers(parent: list[str], entry: str) -> bool:
    """Whether a concrete parent allow-list permits everything ``entry`` could match.

    For a concrete ``entry`` (no wildcard) this is plain membership. For a wildcard
    ``entry`` (``inst:*`` or ``inst:pre*``) the parent must hold an entry at least as
    broad: ``inst:*`` covers anything on that instance, and ``inst:pp*`` covers
    ``inst:cc*`` only when ``cc`` extends ``pp`` (a concrete parent never covers a
    wildcard child).
    """
    inst, _, model = entry.partition(":")
    if not model.endswith("*"):
        return is_model_allowed(parent, entry)
    child_prefix = model[:-1]
    for candidate in parent:
        c_inst, _, c_model = candidate.partition(":")
        if c_inst != inst:
            continue
        if c_model == "*":
            return True
        if c_model.endswith("*") and child_prefix.startswith(c_model[:-1]):
            return True
    return False


def is_allowlist_subset(child: list[str] | None, parent: list[str] | None) -> bool:
    """Whether ``child`` grants no more than ``parent`` (a key vs its user default).

    ``child is None`` means the key inherits the parent, so it can never broaden ->
    always a subset. ``parent is None`` is unrestricted, so any child fits. Otherwise
    every child entry must be covered by the parent; an empty child grants nothing.
    """
    if child is None or parent is None:
        return True
    return all(_covers(parent, entry) for entry in child)


def _entry_matches(entry: str, canonical_key: str) -> bool:
    """Whether one allow-list entry matches a canonical ``instance:model`` key."""
    entry_instance, _, entry_model = entry.partition(":")
    key_instance, _, key_model = canonical_key.partition(":")
    if not entry_instance or not key_instance:
        return False
    if entry_instance != key_instance:
        return False
    if entry_model == "*":
        return True
    if entry_model.endswith("*"):
        return key_model.startswith(entry_model[:-1])
    return entry_model == key_model


def is_model_allowed(allowlist: list[str] | None, canonical_key: str) -> bool:
    """Whether ``canonical_key`` (an ``instance:model`` string) is permitted.

    ``None`` allow-list -> always allowed (unrestricted). An empty list -> never
    allowed (deny all): ``any(...)`` over ``[]`` is ``False``. The ``is None``
    check is load-bearing -- do not collapse it into a truthiness test.
    """
    if allowlist is None:
        return True
    return any(_entry_matches(entry, canonical_key) for entry in allowlist)


def _known_prefix(config: GatewayConfig, prefix: str) -> bool:
    """Whether a prefix names a configured instance or a known any-llm provider."""
    if prefix in config.providers:
        return True
    try:
        LLMProvider(prefix)
    except ValueError:
        return False
    return True


def _validate_entry(config: GatewayConfig, entry: str) -> str:
    """Validate + canonicalize one allow-list entry, or raise ``ValueError``."""
    entry = entry.strip()
    if not entry:
        raise ValueError("an allowed_models entry cannot be empty")
    # An alias name is not a canonical key; storing it would match nothing at
    # dispatch (a fail-open-feeling footgun), so reject it explicitly.
    if resolve_effective_alias(config, entry) is not None:
        raise ValueError(f"'{entry}' is an alias name; use its canonical instance:model target")
    split = split_selector(entry)
    if split is None:
        raise ValueError(f"'{entry}' must be instance:model (e.g. openai:gpt-4o, openai:*, or openai:gpt-4*)")
    prefix, model = split
    if "*" in prefix:
        raise ValueError(f"'{entry}': wildcards are not allowed in the provider part")
    if "*" in model and (model.count("*") > 1 or not model.endswith("*")):
        raise ValueError(f"'{entry}': only a single trailing '*' is allowed (openai:* or openai:gpt-4*)")
    if not _known_prefix(config, prefix):
        raise ValueError(f"'{entry}': unknown provider or instance '{prefix}'")
    return f"{prefix}:{model}"


def validate_allowed_models(config: GatewayConfig, entries: list[str] | None) -> list[str] | None:
    """Validate + canonicalize a write. ``None`` and ``[]`` pass through as-is.

    Raises ``ValueError`` (surfaced by the caller as a 400) on any bad entry.
    Duplicates are collapsed while order is preserved.
    """
    if entries is None:
        return None
    if not isinstance(entries, list):
        raise ValueError("allowed_models must be a list of strings, [] to deny all, or null for unrestricted")
    canonical: list[str] = []
    for entry in entries:
        if not isinstance(entry, str):
            raise ValueError("allowed_models entries must be strings")
        normalized = _validate_entry(config, entry)
        if normalized not in canonical:
            canonical.append(normalized)
    return canonical


def model_not_allowed_detail(model: str) -> str:
    """Human-readable 403 detail for a model a key may not use."""
    return f"Model '{model}' is not permitted for this API key."
