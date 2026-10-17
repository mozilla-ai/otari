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
``provider:model`` selector, i.e. one with no matching ``config.providers``
instance, and which organization's own offerings the catalog spellings below
may resolve to. An ``instance:model`` selector always draws its
credentials from ``config.providers`` exactly as before and never touches
organization-scoped keys, by construction: the two addressing schemes occupy
disjoint selector shapes rather than one layering over the other. See
``services/tenancy/org_provider_key_service.py`` for the overlay this reads.

A third source sits ahead of both for the completion routes alone: an
endpoint the caller's workspace or the caller owns, reached as
``<name>:<model>`` (``services/providers``). It is consulted only when a caller
asks for it with ``owned_endpoints=True``, because what reaches one must skip
budgets and re-check its address before dispatch, and only those routes do.

``workspace_id`` is per-request, not per-deployment: which organization's keys
a master-key request's bare selector can reach is the default workspace's
organization, not every organization the gateway holds. See
``services/workspace_scope.py`` for why a master-key request resolves to that
one workspace at all.
"""

import hashlib
import hmac
import json
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar

from any_llm import AnyLLM, LLMProvider
from any_llm.exceptions import AnyLLMError, MissingApiKeyError

from gateway.auth.vertex_auth import setup_vertex_environment
from gateway.core.config import (
    KEYLESS_SELF_HOSTED_PROVIDERS,
    PROVIDER_TYPE_ALIASES,
    RESERVED_PROVIDER_INSTANCE_NAMES,
    GatewayConfig,
    provider_credential_env_names,
)
from gateway.core.error_codes import PROVIDER_NOT_CONFIGURED
from gateway.core.provider_params import FORBIDDEN_ENDPOINT_DEFAULTS
from gateway.log_config import logger
from gateway.services.alias_service import resolve_effective_alias
from gateway.services.bedrock_gateway_auth import with_bedrock_read_timeout
from gateway.services.catalog import resolve_catalog_offerings, resolve_catalog_selector, resolve_undated_selector
from gateway.services.policy_store import resolve_effective_policy
from gateway.services.tenancy.org_provider_key_service import cached_org_provider_kwargs
from gateway.types.provider_account import ProviderAccount, ResolvedCredential

if TYPE_CHECKING:
    from gateway.services.providers import OwnedEndpoint

# Keys that describe an instance to otari but are not credentials any-llm
# understands, so they must be stripped before the provider call.
_INSTANCE_META_KEYS = ("provider_type", "models", "max_retries", "messages_api")

# Providers whose any-llm client is built by an SDK that takes ``max_retries``
# in its constructor (OpenAI's, Anthropic's and the SDKs generated the same
# way). Every other SDK rejects the keyword or ignores it, so the setting is
# translated for Google's clients below and left out for the rest.
# ``tests/unit/test_provider_retries.py`` builds each one to keep this list true
# of the installed any-llm. ``otari`` builds its own client and is left out.
_MAX_RETRIES_CLIENT_PROVIDERS: frozenset[LLMProvider] = frozenset(
    {
        LLMProvider.ANTHROPIC,
        LLMProvider.AZUREANTHROPIC,
        LLMProvider.VERTEXAIANTHROPIC,
        LLMProvider.OPENAI,
        LLMProvider.AZUREOPENAI,
        LLMProvider.ATLASCLOUD,
        LLMProvider.CASCADIA,
        LLMProvider.CEREBRAS,
        LLMProvider.DASHSCOPE,
        LLMProvider.DATABRICKS,
        LLMProvider.DEEPINFRA,
        LLMProvider.DEEPSEEK,
        LLMProvider.EDENAI,
        LLMProvider.FIREWORKS,
        LLMProvider.GITHUB,
        LLMProvider.GMI,
        LLMProvider.GROQ,
        LLMProvider.INCEPTION,
        LLMProvider.KENARI,
        LLMProvider.LLAMA,
        LLMProvider.LLAMACPP,
        LLMProvider.LLAMAFILE,
        LLMProvider.META,
        LLMProvider.MINIMAX,
        LLMProvider.MOONSHOT,
        LLMProvider.MZAI,
        LLMProvider.NEBIUS,
        LLMProvider.NEOSANTARA,
        LLMProvider.OPENROUTER,
        LLMProvider.PERPLEXITY,
        LLMProvider.PORTKEY,
        LLMProvider.QINIU,
        LLMProvider.REQUESTY,
        LLMProvider.SAMBANOVA,
        LLMProvider.TELNYX,
        LLMProvider.TOGETHER,
        LLMProvider.VLLM,
        LLMProvider.ZAI,
    }
)

# google-genai counts attempts rather than retries, through
# ``http_options.retry_options``, and by default makes one attempt.
_GOOGLE_RETRY_PROVIDERS: frozenset[LLMProvider] = frozenset({LLMProvider.GEMINI, LLMProvider.VERTEXAI})

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

# Providers any-llm calls without an otari-visible credential, even though each
# *declares* a credential environment variable, so ``provider_credential_env_names``
# cannot tell them from a keyed provider the way it can ollama/llamafile (which
# declare the literal ``"None"``) or Vertex AI (which declares an empty name).
# The self-hosted ones are ``KEYLESS_SELF_HOSTED_PROVIDERS``; this set is drift-guarded
# alongside it in ``tests/unit/test_provider_instances.py``.
#
# Selector prefixes that name a provider whatever the deployment configures, so
# an owned endpoint never resolves under one.
_PROVIDER_NAMES: frozenset[str] = frozenset(
    {provider.value for provider in LLMProvider} | set(PROVIDER_TYPE_ALIASES) | RESERVED_PROVIDER_INSTANCE_NAMES
)


# Providers authenticating from cloud SDK credentials this gateway cannot see:
# an EC2 instance profile, an SSO session, or an ambient boto3 chain. They are
# the same category as Vertex AI's application default credentials, which
# ``docs/configuration.md`` already groups with Bedrock, and an entry naming one
# with nothing configured is precisely the ambient case. Splitting the
# category on whether any-llm happens to declare a variable name would be an
# accident of upstream spelling rather than a difference in kind.
#
# Deliberately *not* here: gemini (raises ``MissingApiKeyError`` outright when no
# key resolves, so it genuinely needs one) and azure/azureopenai (an Entra ID
# deployment still needs an endpoint from config, so an empty ``kwargs`` means
# nothing is configured and nothing can serve it anyway).
AMBIENT_CREDENTIAL_PROVIDERS = frozenset({"bedrock", "sagemaker"})

# Keys ``get_provider_kwargs`` can return that credential nothing on their own:
# transport tuning an operator attached to an instance. An instance carrying only
# these has been *described* and not credentialed, so it must not count as a rung
# that answered. ``api_base`` is deliberately absent: an instance with a base URL
# and no key takes the keyless placeholder, so it never reaches a caller alone.
_NON_CREDENTIAL_KWARGS = frozenset({"client_args"})


def _kwargs_carry_a_credential(kwargs: dict[str, Any]) -> bool:
    """Whether anything in a resolved provider's kwargs could authenticate a call.

    Emptiness is not the question: ``providers.openai: {client_args: {timeout:
    60}}`` resolves a non-empty dict with no credential in it, and a key written
    as ``api_key:`` with no value resolves ``None``. Both mean the ladder found a
    provider entry and no way to call it.
    """
    return any(value for key, value in kwargs.items() if key not in _NON_CREDENTIAL_KWARGS)


def keyless_placeholder_api_key(api_base: Any, api_key: Any) -> str | None:
    """Return a placeholder key for a keyless custom endpoint, else ``None``.

    A custom endpoint is one with an ``api_base`` set; it is keyless when no key
    is configured for it. Only that case (which any-llm would otherwise reject,
    or answer with the provider's own environment variable) gets a placeholder,
    so this never overrides a real key and the deployment's variable is never
    sent to an endpoint someone configured.
    """
    if api_base and not api_key:
        return _KEYLESS_PLACEHOLDER_API_KEY
    return None


def credential_ladder_exhausted(provider: LLMProvider, kwargs: dict[str, Any], *, declared: bool) -> bool:
    """Whether a call to ``provider`` with ``kwargs`` has nothing to authenticate with.

    For a caller that has somewhere else to ask once every rung has missed; the
    ladder itself is unchanged and still the only thing consulted first.

    A call is served on what :func:`get_provider_kwargs` produced and nothing
    else: an organization-scoped key, a stored instance's key, a ``config.yml``
    entry's own ``api_key`` (``${VAR}`` interpolation included), or the keyless
    placeholder for an ``api_base``. An entry carrying only ``client_args`` or
    instance metadata carries no credential, per :func:`_kwargs_carry_a_credential`.
    The provider's own SDK environment variable never counts, because any-llm
    reads it whenever no key is passed, and a variable is not a statement that
    the deployment serves anyone on it.

    Without a credential, only a provider a ``providers:`` entry names
    (``declared``) may still be called, and only one any-llm calls with no key at
    all: the keyless local backends (those declaring no credential variable, and
    ``KEYLESS_SELF_HOSTED_PROVIDERS``) and the providers authenticating from
    ambient cloud credentials (Vertex AI and ``AMBIENT_CREDENTIAL_PROVIDERS``).
    For those the entry is the whole declaration. Undeclared, they are exhausted
    like any other, because the local server or the instance role is the
    deployment's own.

    ``provider_credential_env_names`` returns ``None`` rather than ``()`` for a
    provider it cannot inspect at all, and that stays exhausted: nothing is known
    to serve the candidate.
    """
    if _kwargs_carry_a_credential(kwargs):
        return False
    return not (declared and _needs_no_key(provider))


def _needs_no_key(provider: LLMProvider) -> bool:
    """Whether any-llm calls ``provider`` with no key: a keyless local backend or an ambient-credential provider."""
    if provider.value in KEYLESS_SELF_HOSTED_PROVIDERS or provider.value in AMBIENT_CREDENTIAL_PROVIDERS:
        return True
    return provider_credential_env_names(provider.value) == ()


def missing_declared_credential(
    config: GatewayConfig, instance: str, provider: LLMProvider, kwargs: dict[str, Any]
) -> "MissingCredential | None":
    """The credential a call to ``instance`` lacks once every rung has missed, or ``None``.

    For a dispatch path with nowhere else to ask: ``None`` means any-llm may be
    called with ``kwargs``. Otherwise the call must be refused rather than sent,
    because any-llm would read the provider's own environment variable, or call
    the deployment's local backend or cloud role, on nobody's say-so.
    """
    if not credential_ladder_exhausted(provider, kwargs, declared=instance in config.providers):
        return None
    env_names = provider_credential_env_names(provider.value)
    return MissingCredential(provider=instance, env_var="/".join(env_names) if env_names else None)


def require_declared_credential(
    config: GatewayConfig, instance: str, provider: LLMProvider, kwargs: dict[str, Any]
) -> None:
    """Raise what any-llm raises for a missing key when :func:`missing_declared_credential` finds one.

    Raised before the call rather than left to any-llm, so every caller that
    already handles a missing key handles this one the same way.

    Raises:
        MissingApiKeyError: nothing declared serves ``instance``.
    """
    missing = missing_declared_credential(config, instance, provider, kwargs)
    if missing is not None:
        raise MissingApiKeyError(missing.provider, missing.env_var or "")


@dataclass(frozen=True, slots=True)
class HostedLookup:
    """How a candidate may be served by the deployment's hosted providers.

    ``own_name_only`` marks a provider any-llm calls with no key, which only
    the hosted provider of its own name may serve: anything else would answer
    a request for the caller's named backend with a different one.
    """

    own_name_only: bool

    def accepts(self, response_provider: str, instance: str) -> bool:
        """Whether a hosted answer for ``response_provider`` may serve the ``instance`` candidate."""
        return not self.own_name_only or response_provider == instance


def hosted_lookup(provider: LLMProvider) -> HostedLookup:
    """How the hosted providers may serve a candidate :func:`require_declared_credential` refused.

    Asked only for such a candidate, so BYO precedence holds, and a declared
    keyless provider keeps its own connection even when all its entry sets is
    an SDK option such as Bedrock's region.
    """
    return HostedLookup(own_name_only=_needs_no_key(provider))


# Instances already reported as ignored, so the warning is logged once per
# instance per process rather than on every overlay refresh.
_uncredentialed_warned: set[str] = set()


def uncredentialed_env_names(config: GatewayConfig, instance: str, entry: Mapping[str, Any]) -> tuple[str, ...] | None:
    """The variables a ``providers:`` entry could have taken its credential from, or ``None``.

    ``None`` means the entry can be called as it stands: it carries a credential
    or an ``api_base`` (which takes the keyless placeholder), its provider needs
    no key, or the provider cannot be inspected at all. A tuple means the entry
    declares nothing a call could authenticate with: a bare entry for a keyed
    provider, or ``api_key: ${VAR}`` with ``VAR`` unset.
    """
    try:
        provider = LLMProvider(config.provider_instance_type(instance))
    except ValueError:
        return None
    env_names = provider_credential_env_names(provider.value)
    if not env_names:
        return None
    if _entry_declares_a_credential(entry):
        return None
    # Asked with no kwargs: the entry contributes nothing that authenticates, so
    # whether the provider needs a key at all decides.
    if not credential_ladder_exhausted(provider, {}, declared=True):
        return None
    return tuple(env_names)


# Field names that hold something a call can authenticate with. An entry's other
# fields are call options (``temperature``, ``timeout``), which a request would
# carry upstream without ever being let in.
_CREDENTIAL_FIELD = re.compile(r"key|token|secret|credential|password|auth", re.IGNORECASE)


def _entry_declares_a_credential(entry: Mapping[str, Any]) -> bool:
    """Whether a ``providers:`` entry holds anything a call could authenticate with.

    A non-empty credential-named field does, and so does an ``api_base`` (which
    takes the keyless placeholder) or ``client_args`` carrying a credential-named
    field, such as an ``Authorization`` default header.
    """
    for key, value in entry.items():
        if key in _INSTANCE_META_KEYS or not value:
            continue
        if key == "api_base" or _CREDENTIAL_FIELD.search(key):
            return True
        if key == "client_args" and isinstance(value, Mapping) and _mapping_declares_a_credential(value):
            return True
    return False


def _mapping_declares_a_credential(mapping: Mapping[str, Any]) -> bool:
    for key, value in mapping.items():
        if not value:
            continue
        if _CREDENTIAL_FIELD.search(str(key)):
            return True
        if isinstance(value, Mapping) and _mapping_declares_a_credential(value):
            return True
    return False


def prune_uncredentialed_providers(config: GatewayConfig) -> dict[str, tuple[str, ...]]:
    """Drop every ``providers:`` entry that declares no credential, and say so once.

    A hollow entry is worse than no entry: an ``instance:model`` selector that
    matches it never consults an organization's own key for the same provider,
    and a bare ``provider:model`` selector is gated and priced as if the
    deployment served it, so the credential an operator stored on the dashboard
    goes unused while every request fails upstream. Returns what was dropped,
    keyed by instance, with the variables that would have filled each.
    """
    pruned = {
        instance: env_names
        for instance, entry in config.providers.items()
        if isinstance(entry, Mapping) and (env_names := uncredentialed_env_names(config, instance, entry)) is not None
    }
    for instance, env_names in pruned.items():
        del config.providers[instance]
        if instance in _uncredentialed_warned:
            continue
        _uncredentialed_warned.add(instance)
        logger.warning(
            "providers.%s declares no credential, so the entry is ignored. Name the key it reads "
            "(api_key: ${%s}), or store one for the provider through the dashboard.",
            instance,
            # codeql[py/clear-text-logging-sensitive-data]
            env_names[0],
        )
    return pruned


def warn_about_ignored_provider_variables(config: GatewayConfig) -> None:
    """Warn once that a provider's own key variable is set and no ``providers:`` entry reads it.

    Such a variable serves no request, which an operator upgrading from a release
    that fell back to it would otherwise learn only from refused requests. An
    entry reads it when its ``api_key`` holds the variable's value, which is what
    ``${VAR}`` interpolation leaves behind. It is a match on values, because the
    loader keeps no record of which variable filled an entry, so the same key
    pasted in literally also counts as read. Only the ``<PROVIDER>_API_KEY``
    spelling is checked, because confirming any other name means importing every
    provider's SDK. Names only, never values.
    """
    keys_in_use = {
        entry.get("api_key")
        for entry in config.providers.values()
        if isinstance(entry, Mapping) and entry.get("api_key")
    }
    ignored: list[tuple[str, str]] = []
    for provider in LLMProvider:
        name = provider.value
        variable = f"{name.upper()}_API_KEY"
        value = os.getenv(variable)
        if value and value not in keys_in_use and variable in (provider_credential_env_names(name) or ()):
            ignored.append((name, variable))
    if not ignored:
        return
    name, variable = ignored[0]
    logger.warning(
        "Ignoring %s: no providers: entry reads it, and a provider's own environment variable serves nothing "
        "on its own. Name it in the entry that should use it (providers: {%s: {api_key: ${%s}}}) or store the "
        "key on the dashboard's Providers page.",
        # codeql[py/clear-text-logging-sensitive-data]
        ", ".join(variable for _, variable in ignored),
        name,
        # codeql[py/clear-text-logging-sensitive-data]
        variable,
    )


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

            # Credentials, project and location configure the genai client, so they
            # go in client_args: any-llm passes every other kwarg to the completion
            # call, where GenerateContentConfig rejects them.
            kwargs["client_args"] = {
                **setup_vertex_environment(
                    credentials=vertex_creds,
                    project=vertex_project,
                    location=vertex_location,
                ),
                **provider_config.get("client_args", {}),
            }
        else:
            kwargs = {k: v for k, v in provider_config.items() if k != "client_args"}
            if "client_args" in provider_config:
                kwargs["client_args"] = provider_config["client_args"]

    # boto3 reads AWS_DEFAULT_REGION, while some AWS runtimes expose only AWS_REGION.
    if provider == LLMProvider.BEDROCK:
        region = (os.getenv("AWS_REGION") or "").strip()
        default_region = (os.getenv("AWS_DEFAULT_REGION") or "").strip()
        client_args = kwargs.get("client_args")
        if region and not default_region:
            if client_args is None:
                kwargs["client_args"] = {"region_name": region}
            elif isinstance(client_args, dict) and "region_name" not in client_args:
                kwargs["client_args"] = {**client_args, "region_name": region}
        client_args = kwargs.get("client_args")
        if client_args is None or isinstance(client_args, Mapping):
            kwargs["client_args"] = with_bedrock_read_timeout(client_args)

    max_retries = raw_config.get("max_retries") if raw_config is not None else None
    if max_retries is None:
        max_retries = config.provider_max_retries
    if max_retries is not None:
        kwargs = with_provider_retries(provider, kwargs, max_retries)

    placeholder = keyless_placeholder_api_key(kwargs.get("api_base"), kwargs.get("api_key"))
    if placeholder is not None:
        kwargs["api_key"] = placeholder

    return kwargs


def _client_retries_set(provider: LLMProvider, client_args: Mapping[str, Any]) -> bool:
    """Whether ``client_args`` already says how often ``provider``'s client retries."""
    if provider in _GOOGLE_RETRY_PROVIDERS:
        http_options = client_args.get("http_options")
        if isinstance(http_options, Mapping):
            return http_options.get("retry_options") is not None
        return getattr(http_options, "retry_options", None) is not None
    return "max_retries" in client_args


def with_provider_retries(provider: LLMProvider, kwargs: dict[str, Any], max_retries: int) -> dict[str, Any]:
    """``kwargs`` with a client that retries a failed call ``max_retries`` times.

    Returns ``kwargs`` unchanged when the provider's SDK has no retry setting
    Otari knows how to pass, or when the instance's own ``client_args`` already
    set one, which is the most specific statement an operator can make. The
    ``client_args`` dict is copied, never mutated, because it can be the one
    held by the deployment's config.
    """
    if provider not in _MAX_RETRIES_CLIENT_PROVIDERS and provider not in _GOOGLE_RETRY_PROVIDERS:
        return kwargs
    client_args = kwargs.get("client_args")
    if client_args is None:
        client_args = {}
    elif not isinstance(client_args, Mapping):
        return kwargs
    if _client_retries_set(provider, client_args):
        return kwargs
    if provider in _MAX_RETRIES_CLIENT_PROVIDERS:
        return {**kwargs, "client_args": {**client_args, "max_retries": max_retries}}
    http_options = client_args.get("http_options")
    if http_options is not None and not isinstance(http_options, Mapping):
        # An HttpOptions object came from code rather than config; it is left as built.
        return kwargs
    retry_options = {"attempts": max_retries + 1}
    return {
        **kwargs,
        "client_args": {**client_args, "http_options": {**(http_options or {}), "retry_options": retry_options}},
    }


def with_failover_retries(provider: LLMProvider, kwargs: dict[str, Any]) -> dict[str, Any]:
    """``kwargs`` for one candidate of a plan that can fall over, with no client retries by default.

    A provider SDK that retries a 429 and honors its ``Retry-After`` spends seconds
    before the walk may try the next candidate, which is the remedy the policy
    already names. A retry count the instance or the deployment configured wins.
    """
    return with_provider_retries(provider, kwargs, 0)


def effective_credential(provider: LLMProvider, kwargs: Mapping[str, Any]) -> ResolvedCredential:
    """The credential a call made with ``kwargs`` authenticates with: the key they carry.

    Raises:
        LookupError: the kwargs hold no key.
    """
    api_key = kwargs.get("api_key")
    if not api_key:
        raise LookupError(f"no credential configured for provider '{provider.value}'")
    return ResolvedCredential(
        api_key=str(api_key), api_base=kwargs.get("api_base"), client_args=dict(kwargs.get("client_args") or {})
    )


class ProviderAccounts:
    """Names the provider accounts one workspace's dispatch credentials reach.

    The name is a keyed digest, HMAC-SHA256 under a pepper kept for this one
    purpose, so a copy of the database cannot confirm a guessed credential.
    It is derived rather than stored, so a changed credential names a different
    account at once and nothing has to notice the change.
    """

    def __init__(self, *, pepper: str, workspace_id: uuid.UUID) -> None:
        self._pepper = pepper.encode()
        self._workspace_id = workspace_id

    def name(self, provider: LLMProvider, instance: str, credential: ResolvedCredential) -> ProviderAccount:
        """The account ``credential`` reaches.

        The instance is left out of the digest, because renaming one does not
        move its account.
        """
        canonical = json.dumps(
            {"provider": provider.value, "api_base": credential.api_base, "api_key": credential.api_key},
            sort_keys=True,
            separators=(",", ":"),
        )
        return ProviderAccount(
            provider=provider,
            instance=instance,
            workspace_id=self._workspace_id,
            # codeql[py/weak-sensitive-data-hashing]
            identity=hmac.new(self._pepper, canonical.encode(), hashlib.sha256).hexdigest(),
        )


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
    relabeled to this rather than naming the target; pricing, budgets, and usage
    logs still key on the resolved target.
    """
    owned_endpoint: "OwnedEndpoint | None" = None
    """The caller's own endpoint when the selector named one, else ``None``.

    Its owner pays the upstream, so such a request never counts toward a budget.
    """
    deployment_paid: bool = False
    """Whether ``kwargs`` carry a credential ``ModelProviderPort`` served.

    The deployment owes the upstream bill for such a request, so it is gated on
    and charged to its organization's funds through ``BillingPort``. Only the
    hosted rung of the credential ladder sets it.
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


def owned_endpoint_kwargs(endpoint: "OwnedEndpoint") -> dict[str, Any]:
    """The any-llm kwargs for an owned endpoint: its base URL and its key.

    With no key of its own it gets the keyless placeholder unconditionally. The
    provider's environment variable is never a fallback here, because that is the
    operator's key and the base URL is somebody else's server.
    """
    return {"api_base": endpoint.api_base, "api_key": endpoint.api_key or _KEYLESS_PLACEHOLDER_API_KEY}


def apply_endpoint_defaults(call_kwargs: dict[str, Any], resolved: ResolvedProvider) -> dict[str, Any]:
    """Add an owned endpoint's default request fields beneath the ones the call already sets.

    They travel as ``extra_body``, which the provider SDKs merge into the JSON
    body, so a field any-llm does not model (vLLM's ``chat_template_kwargs``)
    reaches the server too. A default is dropped when the call already carries
    that field, because the body merge would otherwise put it over the caller's.
    """
    endpoint = resolved.owned_endpoint
    if endpoint is None or not endpoint.default_params:
        return call_kwargs
    defaults = {
        key: value
        for key, value in endpoint.default_params.items()
        if key not in call_kwargs and key not in FORBIDDEN_ENDPOINT_DEFAULTS
    }
    if not defaults:
        return call_kwargs
    return {**call_kwargs, "extra_body": {**defaults, **(call_kwargs.get("extra_body") or {})}}


def _resolve_owned_endpoint(
    config: GatewayConfig,
    split: tuple[str, str] | None,
    user_id: str | None,
    workspace_id: uuid.UUID | None,
) -> tuple[str, "OwnedEndpoint", str] | None:
    """``(name, endpoint, model)`` when the selector's prefix names one of the caller's endpoints.

    A prefix that already names an instance or a provider keeps that meaning.
    Saving an endpoint refuses such names too, but an instance added at runtime
    or a provider a later any-llm adds would otherwise be taken over by an
    endpoint saved before it.
    """
    if split is None or workspace_id is None or not config.provider_endpoints_enabled:
        return None
    name, model = split
    if name in config.providers or name in _PROVIDER_NAMES:
        return None
    # Imported here: the providers package imports this module through model discovery.
    from gateway.services.providers import cached_owned_endpoint

    endpoint = cached_owned_endpoint(name, workspace_id=workspace_id, user_id=user_id)
    return (name, endpoint, model) if endpoint is not None else None


def resolve_catalog_fallback(
    config: GatewayConfig,
    model_selector: str,
    user_id: str | None = None,
    *,
    workspace_id: uuid.UUID | None = None,
) -> tuple[str, ...]:
    """Every offering a catalog ID stands for, best first, or an empty tuple.

    Empty when an alias or a static policy claims the name, because both come before a catalog spelling.
    """
    if _claimed_target(config, model_selector, user_id, workspace_id) is not None:
        return ()
    return resolve_catalog_offerings(model_selector, workspace_id=workspace_id)


def _claimed_target(
    config: GatewayConfig, model_selector: str, user_id: str | None, workspace_id: uuid.UUID | None
) -> str | None:
    """The target an alias or a static routing policy gives a name, or ``None``."""
    alias = resolve_effective_alias(config, model_selector, user_id, workspace_id=workspace_id)
    if alias is not None:
        return alias
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
    return resolve_static_policy_target(config, model_selector, user_id, workspace_id=workspace_id)


def resolve_provider_selector(
    config: GatewayConfig,
    model_selector: str,
    user_id: str | None = None,
    *,
    workspace_id: uuid.UUID | None = None,
    owned_endpoints: bool = False,
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

    ``owned_endpoints`` lets a ``<name>:<model>`` selector reach an endpoint the
    caller or its workspace owns, ahead of every other source. Only a caller that
    exempts the request from budgets and re-checks the endpoint's address before
    dispatch may pass it; everywhere else such a name is an unknown provider.

    Raises ``ValueError`` / ``AnyLLMError`` (from any-llm) for a selector that
    names neither a configured instance nor a known provider, mirroring the
    prior ``AnyLLM.split_model_provider`` behavior.
    """
    alias = _claimed_target(config, model_selector, user_id, workspace_id)
    if alias is None:
        # The catalog's own spellings: a catalog id for the first of the model's
        # offerings, or ``instance:<catalog id>`` to pin the instance. Consulted last and only
        # for a selector that names no offering already, so nothing a caller
        # sends verbatim is ever rewritten; the workspace picks which
        # organization's own offerings may answer. Relabeled like an alias: the
        # caller sees the spelling they sent, and pricing keys on the target.
        alias = resolve_catalog_selector(model_selector, workspace_id=workspace_id)
    if alias is None and (written := split_selector(model_selector)) is not None and written[0] not in config.providers:
        # An organization key serves only the ids it offers, so a provider's
        # undated name reaches its one dated build there. A ``providers:``
        # instance sends any id as written and is never rewritten.
        alias = resolve_undated_selector(model_selector, workspace_id=workspace_id)
    selector = alias if alias is not None else model_selector

    split = split_selector(selector)
    owned = _resolve_owned_endpoint(config, split, user_id, workspace_id) if owned_endpoints else None
    if owned is not None:
        name, endpoint, model = owned
        return ResolvedProvider(
            instance=name,
            provider=LLMProvider(endpoint.provider),
            model=model,
            kwargs=owned_endpoint_kwargs(endpoint),
            alias=model_selector if alias is not None else None,
            owned_endpoint=endpoint,
        )
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


def is_deployment_instance_key(config: GatewayConfig, model_key: str) -> bool:
    """Whether this pricing key names an instance the deployment holds the credential for.

    The line between the two credential mechanisms, stated as a predicate. An
    ``instance:model`` key whose prefix is in ``config.providers`` (config.yml
    overlaid with the stored ``provider_credentials``) dispatches on the
    deployment's own credential and never consults the organization-scoped tables;
    a bare ``provider:model`` key resolves against an organization's BYO key
    instead. :func:`resolve_provider_selector` above branches on exactly this
    test, and `models.provider_keys` states the rule it comes from.

    So it answers "who pays the upstream bill for this model", which is what
    decides whether an organization may set its own rate for it.
    """
    split = split_selector(model_key)
    return split is not None and split[0] in config.providers


@dataclass(frozen=True)
class MissingCredential:
    """A call any-llm refused locally because no credential resolved for its provider.

    Nothing reached the provider, so the failure is the deployment's
    configuration rather than the provider's, and it is named in Otari's terms.
    The variable is a name, never a value, so it is safe to show. A request that
    named an alias is told the ``alias`` instead of the provider and its
    variable, both of which give away the target the alias hides.

    Answered with a 424 rather than a 502: clients retry a 5xx, and no retry can
    supply a key. ``code`` lets a caller tell it from a provider's own 424.
    """

    provider: str
    env_var: str | None
    alias: str | None = None
    status_code: ClassVar[int] = 424
    code: ClassVar[str] = PROVIDER_NOT_CONFIGURED

    @classmethod
    def from_error(cls, exc: MissingApiKeyError) -> "MissingCredential":
        return cls(provider=exc.provider_name or "unknown", env_var=exc.env_var_name or None)

    def behind_alias(self, alias: str | None) -> "MissingCredential":
        """This credential as a caller who named ``alias`` may be told about it."""
        return self if alias is None else replace(self, alias=alias)

    @property
    def detail(self) -> str:
        if self.alias is not None:
            return (
                f"No credential is configured for the provider behind model '{self.alias}'. "
                "Declare it under providers: in config.yml, store it on the Providers page, "
                "or add an organization provider key."
            )
        # The variable is offered as what a declaration reads its key from: set
        # alone, it serves nothing.
        variable = self.env_var.split("/")[0] if self.env_var else None
        declare = f" (api_key: ${{{variable}}})" if variable else ""
        return (
            f"No credential is configured for provider '{self.provider}'. "
            f"Declare it under providers: in config.yml{declare}, store it on the Providers page, "
            "or add an organization provider key."
        )


def missing_credential(exc: BaseException) -> MissingCredential | None:
    """The missing credential behind ``exc``, read through its ``original_exception`` chain."""
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, MissingApiKeyError):
            return MissingCredential.from_error(current)
        current = getattr(current, "original_exception", None)
    return None


def no_candidate_configured_detail(missing: Sequence[MissingCredential]) -> str:
    """Why a request failed when every candidate it could use lacked a credential."""
    names = ", ".join(sorted({entry.provider for entry in missing})) or "unknown"
    return (
        f"No credential is configured for any provider this request could use ({len(missing)} attempts: {names}). "
        "Add one in config.yml or through the dashboard."
    )
