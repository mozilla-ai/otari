"""Provider-instance resolution and kwargs building from gateway configuration.

A request's model selector (``instance:model``) is resolved here into the
underlying any-llm implementation plus the credentials configured for that
instance. The instance name is an otari-level routing key: it is what pricing,
budgeting, and usage logs are keyed on, while the *implementation* is what
any-llm is actually dispatched against. When an instance name is itself a real
any-llm provider (the common case, no ``provider_type`` declared), the two
coincide and behavior is identical to splitting the selector directly.

``workspace_id`` does two unrelated jobs here, and it is worth keeping them
apart. It selects **which workspace's aliases and routing policies** the
selector is resolved against (``alias_service``, ``policy_store``), which
applies to every selector shape. Separately, it selects **whose
organization-scoped provider keys** may supply credentials (otari-ai#1748,
otari#643): a second, disjoint credential source, consulted only for a *bare*
``provider:model``/``provider/model`` selector, i.e. one with no matching
``config.providers`` instance. An ``instance:model`` selector always draws its
credentials from ``config.providers`` exactly as before and never touches
organization-scoped keys, by construction: the two addressing schemes occupy
disjoint selector shapes rather than one layering over the other. See
``services/tenancy/org_provider_key_service.py`` for the overlay this reads.

``workspace_id`` is per-request, not per-deployment: which organization's keys
a master-key request's bare selector can reach is the default workspace's
organization, not every organization the gateway holds. See
``services/workspace_scope.py`` for why a master-key request resolves to that
one workspace at all.
"""

import os
import uuid
from dataclasses import dataclass
from typing import Any

from any_llm import AnyLLM, LLMProvider
from any_llm.exceptions import AnyLLMError

from gateway.auth.vertex_auth import setup_vertex_environment
from gateway.core.config import GatewayConfig, provider_credential_env_names
from gateway.services.alias_service import resolve_effective_alias
from gateway.services.policy_store import resolve_effective_policy
from gateway.services.tenancy.org_provider_key_service import cached_org_provider_kwargs

# Keys that describe an instance to otari but are not credentials any-llm
# understands, so they must be stripped before the provider call.
_INSTANCE_META_KEYS = ("provider_type", "models")

# any-llm rejects a keyless call to most providers (openai, anthropic, ...) with
# MissingApiKeyError, but a self-hosted OpenAI-/Anthropic-compatible backend
# (vLLM, llama.cpp, Ollama) usually needs no auth. When an instance points at a
# custom api_base and supplies no key anywhere, hand any-llm this harmless
# placeholder so the keyless endpoint is reachable, making good on the
# dashboard's "API key (optional)" promise for custom endpoints. A local server
# ignores the value; a real one that does need a key rejects it, which is the
# same failure the operator would already get. Mirrors any-llm's own keyless
# tolerance (mozilla-ai/any-llm#1198).
_KEYLESS_PLACEHOLDER_API_KEY = "otari-no-key-required"

# Two sets of providers any-llm calls without an otari-visible credential, even
# though each *declares* a credential environment variable.
# ``provider_credential_env_names`` sees only the declaration, so it cannot tell
# them from a keyed provider the way it can ollama/llamacpp/llamafile (which
# declare the literal ``"None"``) or Vertex AI (which declares an empty name).
# Both are written out by hand and both are drift-guarded in
# ``tests/unit/test_provider_instances.py``.
#
# Local and LAN backends that never require a key: each overrides
# ``_verify_and_set_api_key`` to return without raising and defaults to a
# localhost or LAN base URL, so a bare ``vllm:my-model`` reaches a self-hosted
# server today with nothing configured in otari at all.
_KEYLESS_SELF_HOSTED_PROVIDERS = frozenset({"vllm", "lmstudio", "cascadia", "otari"})
# Providers authenticating from cloud SDK credentials this gateway cannot see:
# an EC2 instance profile, an SSO session, or an ambient boto3 chain. They are
# the same category as Vertex AI's application default credentials, which
# ``docs/configuration.md`` already groups with Bedrock, and they only reach here
# with nothing configured, which is precisely the ambient case. Splitting the
# category on whether any-llm happens to declare a variable name would be an
# accident of upstream spelling rather than a difference in kind.
#
# Deliberately *not* here: gemini (raises ``MissingApiKeyError`` outright when no
# key resolves, so it genuinely needs one) and azure/azureopenai (an Entra ID
# deployment still needs an endpoint from config, so an empty ``kwargs`` means
# nothing is configured and nothing can serve it anyway).
_AMBIENT_CREDENTIAL_PROVIDERS = frozenset({"bedrock", "sagemaker"})

# Keys ``get_provider_kwargs`` can return that credential nothing on their own:
# transport tuning an operator attached to an instance. An instance carrying only
# these has been *described* and not credentialed, so it must not count as a rung
# that answered. ``api_base`` is deliberately absent: an instance with a base URL
# and no key takes the keyless placeholder, so it never reaches a caller alone.
_NON_CREDENTIAL_KWARGS = frozenset({"client_args"})

# The anthropic SDK's non-streaming pre-flight guard fires for a large max_tokens
# only while the client's timeout is untouched (otari#533); this is also the
# guard's own fallback value, so setting it explicitly disables the guard alone.
ANTHROPIC_DEFAULT_TIMEOUT_SECONDS = 600.0


def with_anthropic_default_timeout(
    provider: LLMProvider, client_args: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Ensure an Anthropic ``client_args`` carries an explicit ``timeout``.

    A no-op for every other provider. An operator's own ``timeout`` (set via
    ``client_args`` on the provider config or a platform attempt's
    ``extra_params``) is left untouched; this only fills the gap when nothing
    was configured.
    """
    if provider != LLMProvider.ANTHROPIC:
        return client_args
    merged = dict(client_args) if client_args else {}
    merged.setdefault("timeout", ANTHROPIC_DEFAULT_TIMEOUT_SECONDS)
    return merged


def _kwargs_carry_a_credential(kwargs: dict[str, Any]) -> bool:
    """Whether anything in a resolved provider's kwargs could authenticate a call.

    Emptiness is not the question: ``providers.openai: {client_args: {timeout:
    60}}`` resolves a non-empty dict with no credential in it, and a key written
    as ``api_key:`` with no value resolves ``None``. Both mean the ladder found a
    provider entry and no way to call it.
    """
    return any(value for key, value in kwargs.items() if key not in _NON_CREDENTIAL_KWARGS)


def _provider_env_key_present(provider: LLMProvider) -> bool:
    """Whether the provider's native API-key env var (e.g. OPENAI_API_KEY) is set.

    Read directly from the environment because these are any-llm's own SDK
    variables, not otari config: any-llm falls back to them before raising, so
    the placeholder must not shadow a key the operator supplied that way. The
    candidate names come from :func:`provider_credential_env_names`, shared with
    the config layer so both agree on which variables carry a credential.
    """
    return any(os.getenv(name) for name in provider_credential_env_names(provider.value) or ())


def keyless_placeholder_api_key(provider: LLMProvider, api_base: Any, api_key: Any) -> str | None:
    """Return a placeholder key for a keyless custom endpoint, else ``None``.

    A custom endpoint is one with an ``api_base`` set; it is keyless when no key
    is configured for it and the provider's native env var is unset. Only that
    case (which any-llm would otherwise reject) gets a placeholder, so this never
    overrides a real key or the documented env-var fallback.
    """
    if api_base and not api_key and not _provider_env_key_present(provider):
        return _KEYLESS_PLACEHOLDER_API_KEY
    return None


def credential_ladder_exhausted(provider: LLMProvider, kwargs: dict[str, Any]) -> bool:
    """Whether a credential was needed for ``provider`` and none was found.

    For a caller that has somewhere else to ask once every rung has missed; the
    ladder itself is unchanged and still the only thing consulted first.

    Both halves of that sentence are load-bearing. **None was found** means
    :func:`get_provider_kwargs` produced no credential for the candidate: no
    organization-scoped key, no stored instance, nothing usable from a
    ``config.yml`` entry (one declaring only ``provider_type``/``models`` leaves
    nothing behind once the meta keys are stripped, and one carrying only
    ``client_args`` leaves nothing that can authenticate a call, per
    :func:`_kwargs_carry_a_credential`), and so no keyless placeholder either,
    *and* the provider's own SDK environment variable is
    unset, which any-llm would otherwise fall back to before raising. That env
    check is the same one the keyless placeholder makes, so both agree on what
    counts as a credential already in hand.

    **Was needed** excludes the providers any-llm calls with no credential at
    all, because an empty ``kwargs`` for one of those is not a missing key, and
    reporting it as exhausted would hand a request that already works to a
    caller that might serve it from somewhere else. A deployment pointing at its
    own backends is served upstream of anything reading this. They come in two
    shapes: those declaring no credential variable at all (the keyless local
    backends ollama, llamacpp and llamafile, and Vertex AI, which authenticates
    through the cloud SDK), and those declaring one any-llm does not insist on
    (``_KEYLESS_SELF_HOSTED_PROVIDERS`` and ``_AMBIENT_CREDENTIAL_PROVIDERS``).

    ``provider_credential_env_names`` returns ``None`` rather than ``()`` for a
    provider it cannot inspect at all, and that stays exhausted: nothing is known
    to serve the candidate, which is the same position an uncredentialed provider
    with a declared variable is in.
    """
    if _kwargs_carry_a_credential(kwargs):
        return False
    if provider.value in _KEYLESS_SELF_HOSTED_PROVIDERS or provider.value in _AMBIENT_CREDENTIAL_PROVIDERS:
        return False
    if provider_credential_env_names(provider.value) == ():
        return False
    return not _provider_env_key_present(provider)


def get_provider_kwargs(
    config: GatewayConfig,
    provider: LLMProvider,
    instance: str | None = None,
    *,
    workspace_id: uuid.UUID | None = None,
) -> dict[str, Any]:
    """Get provider kwargs from config for any-llm calls.

    Args:
        config: Gateway configuration
        provider: Underlying any-llm implementation (drives provider-specific
            handling such as Vertex AI environment setup).
        instance: Configured instance name to read credentials from. Defaults to
            ``provider.value`` so existing call sites that key by implementation
            keep working unchanged.
        workspace_id: The requesting workspace, if any. Consulted only as a
            fallback, when no ``config.providers`` entry answers ``lookup``:
            organization-scoped keys never shadow an instance or config.yml
            provider, they only fill the gap when neither exists.

    Returns:
        Dictionary of provider kwargs (credentials, client_args, etc.) with the
        otari-only instance metadata stripped.

    """
    lookup = instance if instance is not None else provider.value
    kwargs: dict[str, Any] = {}
    raw_config = config.providers.get(lookup)
    if raw_config is None and workspace_id is not None:
        raw_config = cached_org_provider_kwargs(workspace_id, provider.value)
    if raw_config is not None:
        provider_config = {k: v for k, v in raw_config.items() if k not in _INSTANCE_META_KEYS}

        if provider == LLMProvider.VERTEXAI:
            vertex_creds = provider_config.get("credentials")
            vertex_project = provider_config.get("project")
            vertex_location = provider_config.get("location")

            kwargs.update(
                setup_vertex_environment(
                    credentials=vertex_creds,
                    project=vertex_project,
                    location=vertex_location,
                )
            )
            if "client_args" in provider_config:
                kwargs["client_args"] = provider_config["client_args"]
        else:
            kwargs = {k: v for k, v in provider_config.items() if k != "client_args"}
            if "client_args" in provider_config:
                kwargs["client_args"] = provider_config["client_args"]

    placeholder = keyless_placeholder_api_key(provider, kwargs.get("api_base"), kwargs.get("api_key"))
    if placeholder is not None:
        kwargs["api_key"] = placeholder

    if provider == LLMProvider.ANTHROPIC:
        kwargs["client_args"] = with_anthropic_default_timeout(provider, kwargs.get("client_args"))

    return kwargs


@dataclass(frozen=True)
class ResolvedProvider:
    """A model selector resolved against the configured provider instances."""

    instance: str
    """Otari-level routing key: pricing / budget / usage-log key prefix."""
    provider: LLMProvider
    """Underlying any-llm implementation to dispatch against."""
    model: str
    """Bare model name (no instance/provider prefix)."""
    kwargs: dict[str, Any]
    """Credentials / client args for the any-llm call."""
    alias: str | None = None
    """Display name the caller used when the selector was a configured alias.

    ``None`` for an ordinary selector. When set, response ``model`` fields are
    relabeled to this so the underlying provider/model stays hidden; pricing,
    budgets, and usage logs still key on the resolved target.
    """

    @property
    def dispatch_model(self) -> str:
        """The selector to hand to any-llm: ``<implementation>:<model>``."""
        return f"{self.provider.value}:{self.model}"


def provider_key(provider: str | LLMProvider) -> str:
    """The wire name of an any-llm provider.

    ``AnyLLM.split_model_provider`` returns an ``LLMProvider`` member where one
    exists and a bare name for registry-only gateways (any-llm >= 1.24), so
    callers that only need the name go through here instead of ``.value``.
    """
    return provider.value if isinstance(provider, LLMProvider) else provider


def split_selector(model_selector: str) -> tuple[str, str] | None:
    """Split a selector on its first ``:`` or ``/`` delimiter.

    Returns ``(prefix, remainder)`` or ``None`` when there is no usable
    delimiter (matching ``AnyLLM.split_model_provider``'s notion of a prefix).
    """
    colon = model_selector.find(":")
    slash = model_selector.find("/")
    if colon != -1 and (slash == -1 or colon < slash):
        prefix, remainder = model_selector.split(":", 1)
    elif slash != -1:
        prefix, remainder = model_selector.split("/", 1)
    else:
        return None
    if not prefix or not remainder:
        return None
    return prefix, remainder


def resolve_provider_selector(
    config: GatewayConfig,
    model_selector: str,
    user_id: str | None = None,
    *,
    workspace_id: uuid.UUID | None = None,
) -> ResolvedProvider:
    """Resolve a request model selector into instance, implementation, and kwargs.

    A selector whose prefix names a configured instance resolves to that
    instance's ``provider_type`` (defaulting to the instance name). Otherwise the
    selector is split by any-llm directly, so unconfigured selectors and the
    legacy ``provider/model`` form keep working exactly as before, and it is
    this bare-selector branch alone that hands ``workspace_id`` on to credential
    resolution: an instance-addressed selector never consults organization-scoped
    keys (see the module docstring). Alias and policy resolution reads the
    workspace on both branches.

    A selector that names an alias, whether from ``config.yml`` or the
    ``model_aliases`` table, is first substituted with the alias target, then
    resolved as usual; the resulting ``ResolvedProvider`` carries ``alias`` so
    response ``model`` fields can be relabeled.

    ``user_id`` is the billed user, so a user-scoped alias resolves to that
    user's target. Omit it only for a selector that is not caller input (the
    operator-configured vision describe model), which belongs to no user by
    definition; omitting it for a request selector would silently ignore the
    caller's own aliases and resolve the workspace-wide one instead.
    ``workspace_id`` is the same kind of omission: pass it for caller input,
    leave it unset for an operator-configured or otherwise workspace-less
    resolution, which then reads the deployment's default workspace's aliases
    and policies.

    Raises ``ValueError`` / ``AnyLLMError`` (from any-llm) for a selector that
    names neither a configured instance nor a known provider, mirroring the
    prior ``AnyLLM.split_model_provider`` behavior.
    """
    alias = resolve_effective_alias(config, model_selector, user_id, workspace_id=workspace_id)
    if alias is None:
        # A *static* routing policy is an alias in everything but spelling: one
        # name, one target. Resolving it here is what makes "an alias is a
        # one-target policy" true on every model-taking surface (pricing, the
        # catalog, embeddings, batches) and not only on the completion routes.
        #
        # A dynamic policy is deliberately left unresolved. Its candidate depends
        # on request state that this synchronous path cannot see, so there is no
        # honest answer to give; picking its default anyway would silently serve a
        # different model than the policy describes. The completion routes compile
        # it properly; everywhere else it surfaces as an unknown model.
        alias = resolve_static_policy_target(config, model_selector, user_id, workspace_id=workspace_id)
    selector = alias if alias is not None else model_selector

    split = split_selector(selector)
    if split is not None and split[0] in config.providers:
        instance, model = split
        provider = LLMProvider(config.provider_instance_type(instance))
        return ResolvedProvider(
            instance=instance,
            provider=provider,
            model=model,
            kwargs=get_provider_kwargs(config, provider, instance=instance),
            alias=model_selector if alias is not None else None,
        )

    # A registry-only gateway (a name with no ``LLMProvider`` member) is not
    # something otari can key pricing/budgets on, so it is rejected here exactly
    # as an unknown provider was before any-llm widened this return type.
    split_provider, model = AnyLLM.split_model_provider(selector)
    provider = LLMProvider(split_provider)
    return ResolvedProvider(
        instance=provider.value,
        provider=provider,
        model=model,
        kwargs=get_provider_kwargs(config, provider, instance=provider.value, workspace_id=workspace_id),
        alias=model_selector if alias is not None else None,
    )


def resolve_static_policy_target(
    config: GatewayConfig,
    model_selector: str,
    user_id: str | None = None,
    *,
    workspace_id: uuid.UUID | None = None,
) -> str | None:
    """The single target of a static routing policy, or ``None``.

    ``None`` for a name that is not a policy, for a dynamic policy (whose target
    depends on the request), and when routing is disabled. Scoped like an alias, so
    a user-scoped policy in this workspace resolves to that user's target.
    """
    spec = resolve_effective_policy(config, model_selector, user_id, workspace_id=workspace_id)
    if spec is None or spec.is_dynamic:
        return None
    return spec.default_target


def normalize_pricing_key(config: GatewayConfig, raw_key: str) -> str:
    """Normalize a pricing model key to its canonical ``instance:model`` form.

    A key whose prefix names a configured instance is kept as ``instance:model``;
    otherwise it is normalized through any-llm's provider split (so the legacy
    ``provider/model`` form collapses onto ``provider:model``). An unparseable
    key with no usable prefix is returned unchanged.
    """
    split = split_selector(raw_key)
    if split is not None and split[0] in config.providers:
        return f"{split[0]}:{split[1]}"
    try:
        provider, model = AnyLLM.split_model_provider(raw_key)
    # AnyLLMError (UnsupportedProviderError) fires when the prefix is an instance
    # name that is no longer configured, e.g. a pricing row left behind after its
    # stored provider was removed or could not be decrypted. Return it unchanged
    # rather than 500 the whole listing.
    except (ValueError, AnyLLMError):
        return raw_key
    return f"{provider_key(provider)}:{model}"
