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

from gateway.core.config import GatewayConfig
from gateway.models.entities import APIKey, User
from gateway.services.alias_service import resolve_effective_alias
from gateway.services.provider_kwargs import split_selector

# Wire code used in 403 bodies so clients can branch on it programmatically.
PERMISSION_CODE = "model_not_allowed"


def effective_allowlist(api_key: APIKey | None, user: User | None = None) -> list[str] | None:
    """Resolve the allow-list that governs a request. ``None`` = unrestricted.

    The key's own list wins when set; otherwise the user default (the per-user
    layer lands with the Users page, issue #300); otherwise unrestricted. ``user``
    is accepted now so the signature and every call site stay unchanged when the
    user column is added -- v1 reads it defensively via ``getattr`` so it works
    before that column exists.
    """
    key_list: list[str] | None = api_key.allowed_models if api_key is not None else None
    if key_list is not None:
        return key_list
    if user is not None:
        user_list: list[str] | None = getattr(user, "allowed_models", None)
        if user_list is not None:
            return user_list
    return None


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
