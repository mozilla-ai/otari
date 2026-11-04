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

A search tool has a list of its own, ``allowed_tools``, of tool names
(:func:`effective_tool_allowlist`), so no model wildcard can grant one.
"""

import uuid
from collections.abc import Container, Sequence
from dataclasses import dataclass
from http import HTTPStatus

from any_llm import LLMProvider
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.core.error_codes import INVALID_MODEL, MODEL_NOT_ALLOWED, MODEL_NOT_FOUND, MODEL_NOT_SERVING
from gateway.core.settings.tools import ALL_SEARCH_TOOLS
from gateway.models.api_keys import APIKey
from gateway.models.users import User
from gateway.services.alias_service import all_alias_names
from gateway.services.catalog import dated_builds
from gateway.services.provider_kwargs import reaches_organization_key, split_selector
from gateway.services.tenancy.org_provider_key_service import cached_org_key_offer, cached_org_model_restriction


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


# A key inherits its user's lists and may narrow them, never broaden them, so a
# key list that grants more than its user's is refused on write.
KEY_MODELS_EXCEED_USER_DETAIL = (
    "This key's allowed_models exceeds its user's default; a key can only narrow "
    "the user's model access, not broaden it."
)
KEY_TOOLS_EXCEED_USER_DETAIL = (
    "This key's allowed_tools exceeds its user's; a key can only narrow the user's search tool access, not broaden it."
)


def is_allowlist_subset(child: list[str] | None, parent: list[str] | None, *, tool_keys: Container[str]) -> bool:
    """Whether ``child`` grants no more than ``parent`` (a key vs its user default).

    ``child is None`` means the key inherits the parent, so it can never broaden ->
    always a subset. ``parent is None`` is unrestricted, so any child fits. Otherwise
    every child entry must be covered by the parent; an empty child grants nothing.
    An entry naming a search tool the deprecated way, one of ``tool_keys``, is
    covered only by the same entry, since no wildcard grants a tool.
    """
    if child is None or parent is None:
        return True
    return all(entry in parent if entry in tool_keys else _covers(parent, entry) for entry in child)


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


def legacy_tool_grants(config: GatewayConfig, allowed_models: list[str] | None) -> list[str] | None:
    """The search tools a model list grants by naming ``<provider>:<tool>``, the deprecated way.

    ``None`` stays unrestricted. Only an entry naming a configured tool's key
    whole counts, never a wildcard, which grants models alone.
    """
    if allowed_models is None:
        return None
    legacy = config.legacy_search_tool_keys()
    grants = (entry.partition(":")[2] for entry in dict.fromkeys(allowed_models) if entry in legacy)
    return [name for name in grants if name != ALL_SEARCH_TOOLS]


def effective_tool_allowlist(
    config: GatewayConfig,
    *,
    key_tools: list[str] | None,
    key_models: list[str] | None,
    user_tools: list[str] | None,
    user_models: list[str] | None,
) -> list[str] | None:
    """The search tools a key may use, by name. ``None`` = every one.

    The key's own ``allowed_tools`` wins, then its user's. With neither set, the
    tools its model list grants the deprecated way, so a key restricted to models
    stays refused every tool until one is granted, and never one its user's model
    list does not grant too: a user's wildcard can admit a key's exact entry at
    write time, and the wildcard grants no tool.
    """
    if key_tools is not None:
        return key_tools
    if user_tools is not None:
        return user_tools
    if key_models is None:
        return legacy_tool_grants(config, user_models)
    grants = legacy_tool_grants(config, key_models) or []
    user_grants = legacy_tool_grants(config, user_models)
    return grants if user_grants is None else [name for name in grants if name in user_grants]


def tools_within_user(
    config: GatewayConfig, tools: list[str] | None, *, user_tools: list[str] | None, user_models: list[str] | None
) -> bool:
    """Whether a key's ``allowed_tools`` fits its user's: the user's own list, else what its model list grants."""
    user_allowed = effective_tool_allowlist(
        config, key_tools=None, key_models=None, user_tools=user_tools, user_models=user_models
    )
    return is_tool_allowlist_subset(tools, user_allowed)


def is_tool_allowed(allowlist: list[str] | None, tool: str) -> bool:
    """Whether a search tool, by name, is permitted. ``None`` = every tool, ``[]`` = none."""
    if allowlist is None:
        return True
    return ALL_SEARCH_TOOLS in allowlist or tool in allowlist


def is_tool_allowlist_subset(child: list[str] | None, parent: list[str] | None) -> bool:
    """Whether a key's ``allowed_tools`` grants no more than its user's; ``*`` is covered only by ``*``."""
    if child is None or parent is None:
        return True
    return ALL_SEARCH_TOOLS in parent or all(entry != ALL_SEARCH_TOOLS and entry in parent for entry in child)


def validate_allowed_tools(config: GatewayConfig, entries: list[str] | None) -> list[str] | None:
    """Validate a write of ``allowed_tools``. ``None`` and ``[]`` pass through as-is.

    An entry is a search tool's name, configured or stored, or ``*``. Raises
    ``ValueError`` (a 400 at the caller) on any other; duplicates are collapsed.
    """
    if entries is None:
        return None
    if not isinstance(entries, list):
        raise ValueError("allowed_tools must be a list of strings, [] to deny all, or null to inherit")
    names: list[str] = []
    for entry in entries:
        if not isinstance(entry, str):
            raise ValueError("allowed_tools entries must be strings")
        name = entry.strip()
        if name != ALL_SEARCH_TOOLS and name not in config.search_tools:
            raise ValueError(f"'{name}' is not a search tool; name a configured one, or '*' for every search tool")
        if name not in names:
            names.append(name)
    return names


def tool_not_allowed_detail(tool: str, legacy_key: str, legacy_models: Sequence[list[str]] = ()) -> str:
    """Human-readable 403 detail for a search tool a key may not use.

    ``legacy_models`` are the model lists the refusal was read off the deprecated
    way, if it was; a wildcard there that matches ``legacy_key`` is named.
    """
    detail = f"Search tool '{tool}' is not permitted for this API key. Name it in allowed_tools to allow it."
    if any(legacy_key not in models and is_model_allowed(models, legacy_key) for models in legacy_models):
        detail += f" A wildcard in allowed_models allows models only, not '{legacy_key}'."
    return detail


def _known_prefix(config: GatewayConfig, prefix: str) -> bool:
    """Whether a prefix names something a model allow-list entry can legitimately gate.

    A configured provider instance or a known any-llm provider. The configured
    decision providers are accepted because ``/v1/decisions`` checks
    ``<provider>:<model>``.
    """
    if prefix in config.providers:
        return True
    if prefix in config.decision_providers:
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
    if entry in all_alias_names(config):
        raise ValueError(f"'{entry}' is an alias name; use its canonical instance:model target")
    split = split_selector(entry)
    if split is None:
        raise ValueError(f"'{entry}' must be instance:model (e.g. openai:gpt-4o, openai:*, or openai:gpt-4*)")
    prefix, model = split
    if "*" in prefix:
        raise ValueError(f"'{entry}': wildcards are not allowed in the provider part")
    if "*" in model and (model.count("*") > 1 or not model.endswith("*")):
        raise ValueError(f"'{entry}': only a single trailing '*' is allowed (openai:* or openai:gpt-4*)")
    canonical = f"{prefix}:{model}"
    if _known_prefix(config, prefix) or canonical in config.legacy_search_tool_keys():
        return canonical
    if prefix in config.search_tool_providers():
        raise ValueError(
            f"'{entry}' grants no model, and allowed_models grants a search tool only by its exact "
            "<provider>:<tool> key; name tools in allowed_tools"
        )
    raise ValueError(f"'{entry}': unknown provider or instance '{prefix}'")


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


def unresolvable_model_detail(model_selector: str) -> str:
    """Human-readable 400 detail for a selector the gateway cannot resolve."""
    return (
        f"Unknown or unsupported model {model_selector!r}. Send a 'provider:model' selector, an alias, "
        f"or a model id from the catalog (GET {API_ROOT}/models)."
    )


def _quoted(names: tuple[str, ...]) -> str:
    return ", ".join(f"'{name}'" for name in names)


def model_not_serving_detail(model: str, key_name: str, *, served_as: tuple[str, ...] = ()) -> str:
    """403 detail for a model an organization offers on a provider key with its serving switch off.

    ``served_as`` names the dated builds an undated ``model`` stands for, where those are what is switched off.
    """
    offered_as = f" as {_quoted(served_as)}" if served_as else ""
    return (
        f"Model '{model}' is offered on provider key '{key_name}'{offered_as} but not serving. "
        "Turn it on under Organization > Providers."
    )


def model_not_found_detail(model: str, key_name: str, *, dated: tuple[str, ...] = ()) -> str:
    """404 detail for a model no provider key of the organization offers.

    ``dated`` names the dated builds of an undated ``model`` the key does offer, which it cannot choose between.
    """
    if dated:
        return f"Provider key '{key_name}' does not offer '{model}'. It offers {_quoted(dated)}; send one of those."
    return f"Provider key '{key_name}' does not offer '{model}'. Refresh the key's models or add it."


@dataclass(frozen=True)
class ModelRefusal:
    """Why a request's model is turned away, in wire terms."""

    code: str
    status_code: int
    detail: str


def unresolved_model_refusal(resolved_instance: str | None, *, selector: str) -> ModelRefusal | None:
    """Why a selector that resolved to no instance is turned away, or ``None`` when it resolved.

    A selector no instance, known provider, alias, policy or catalog id resolves is
    an invalid model whatever the pricing settings: a name that names nothing is not
    unpriced. A ``provider:model`` pair whose model the gateway does not know locally
    still resolves, and is left to the pricing gate and the provider.
    """
    if resolved_instance is not None:
        return None
    return ModelRefusal(
        code=INVALID_MODEL, status_code=HTTPStatus.BAD_REQUEST, detail=unresolvable_model_detail(selector)
    )


def key_model_refusal(allowlist: list[str] | None, canonical_key: str | None, *, selector: str) -> ModelRefusal | None:
    """Why a key's allow-list turns a request away, or ``None`` when it does not.

    ``canonical_key`` is the ``instance:model`` the selector resolved to, or
    ``None`` when it resolved to nothing. Such a selector is refused as an invalid
    model rather than as one the key may not use: the allow-list cannot judge a
    name that names nothing, and a restricted key would otherwise be told
    something different from what an unrestricted one hears. ``selector`` is what
    the caller wrote and is what the detail names.
    """
    if allowlist is None:
        return None
    if canonical_key is None:
        return unresolved_model_refusal(None, selector=selector)
    if is_model_allowed(allowlist, canonical_key):
        return None
    return ModelRefusal(
        code=MODEL_NOT_ALLOWED, status_code=HTTPStatus.FORBIDDEN, detail=model_not_allowed_detail(selector)
    )


def organization_key_refusal(
    config: GatewayConfig,
    workspace_id: uuid.UUID | None,
    instance: str,
    provider: str,
    model: str,
    *,
    selector: str,
    through_alias: bool,
) -> ModelRefusal | None:
    """:func:`org_model_refusal` for a candidate that can be served on an organization key, else ``None``.

    Only a selector that names neither a configured instance nor a named
    hosted provider reaches an organization key (``reaches_organization_key``),
    so only it is subject to that key's allow-list; a request with no
    workspace has no organization key to ask about.
    """
    if workspace_id is None or not reaches_organization_key(config, instance):
        return None
    return org_model_refusal(workspace_id, provider, model, selector=selector, through_alias=through_alias)


def org_model_refusal(
    workspace_id: uuid.UUID, provider: str, model: str, *, selector: str, through_alias: bool
) -> ModelRefusal | None:
    """Why the active organization key's allow-list turns ``model`` away, or ``None`` when it does not.

    Reads the same list the catalog does (`cached_org_model_restriction`), so
    the gate and the catalog cannot disagree about whether the model serves;
    what it adds is the reason, which the list alone cannot carry. A model the
    organization offers with its switch off is refused as not serving, so the
    operator is sent to the switch rather than to a key's permissions. One no
    key offers at all is not found, which is what a misspelled or never-listed
    model is. A model the key serves that a workspace restriction excludes, and
    the legacy case of a restriction on a key offering no rows, keep the
    not-allowed refusal. ``selector`` is what the caller wrote (an alias, a
    ``provider:model`` pair) and is what the detail names.

    An undated name reaches here only where the catalog could not resolve it
    to one serving dated build. Where every dated build of it is switched off
    it is not serving, and otherwise the detail names the builds to choose from.
    It names them only to a caller who wrote the model: ``through_alias`` says
    the selector resolved through an alias, a policy or a catalog spelling,
    whose target the detail keeps off the wire.
    """
    allowed = cached_org_model_restriction(workspace_id, provider)
    if allowed is None or model in allowed:
        return None
    not_allowed = ModelRefusal(
        code=MODEL_NOT_ALLOWED, status_code=HTTPStatus.FORBIDDEN, detail=model_not_allowed_detail(selector)
    )
    offer = cached_org_key_offer(workspace_id, provider)
    if offer is None or offer.offered is None:
        return not_allowed
    enabled = offer.offered.get(model)
    if enabled is None:
        builds = dated_builds(model, offer.offered)
        shown = () if through_alias else tuple(f"{provider}:{build}" for build in builds)
        if builds and not any(offer.offered[build] for build in builds):
            return ModelRefusal(
                code=MODEL_NOT_SERVING,
                status_code=HTTPStatus.FORBIDDEN,
                detail=model_not_serving_detail(selector, offer.key_name, served_as=shown),
            )
        return ModelRefusal(
            code=MODEL_NOT_FOUND,
            status_code=HTTPStatus.NOT_FOUND,
            detail=model_not_found_detail(selector, offer.key_name, dated=shown),
        )
    if not enabled:
        return ModelRefusal(
            code=MODEL_NOT_SERVING,
            status_code=HTTPStatus.FORBIDDEN,
            detail=model_not_serving_detail(selector, offer.key_name),
        )
    return not_allowed
