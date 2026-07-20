"""Provider-instance resolution and kwargs building from gateway configuration.

A request's model selector (``instance:model``) is resolved here into the
underlying any-llm implementation plus the credentials configured for that
instance. The instance name is an otari-level routing key: it is what pricing,
budgeting, and usage logs are keyed on, while the *implementation* is what
any-llm is actually dispatched against. When an instance name is itself a real
any-llm provider (the common case, no ``provider_type`` declared), the two
coincide and behavior is identical to splitting the selector directly.
"""

from dataclasses import dataclass
from typing import Any

from any_llm import AnyLLM, LLMProvider
from any_llm.exceptions import AnyLLMError

from gateway.auth.vertex_auth import setup_vertex_environment
from gateway.core.config import GatewayConfig
from gateway.services.alias_service import resolve_effective_alias

# Keys that describe an instance to otari but are not credentials any-llm
# understands, so they must be stripped before the provider call.
_INSTANCE_META_KEYS = ("provider_type", "models")


def get_provider_kwargs(
    config: GatewayConfig,
    provider: LLMProvider,
    instance: str | None = None,
) -> dict[str, Any]:
    """Get provider kwargs from config for any-llm calls.

    Args:
        config: Gateway configuration
        provider: Underlying any-llm implementation (drives provider-specific
            handling such as Vertex AI environment setup).
        instance: Configured instance name to read credentials from. Defaults to
            ``provider.value`` so existing call sites that key by implementation
            keep working unchanged.

    Returns:
        Dictionary of provider kwargs (credentials, client_args, etc.) with the
        otari-only instance metadata stripped.

    """
    lookup = instance if instance is not None else provider.value
    kwargs: dict[str, Any] = {}
    raw_config = config.providers.get(lookup)
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


def resolve_provider_selector(config: GatewayConfig, model_selector: str) -> ResolvedProvider:
    """Resolve a request model selector into instance, implementation, and kwargs.

    A selector whose prefix names a configured instance resolves to that
    instance's ``provider_type`` (defaulting to the instance name). Otherwise the
    selector is split by any-llm directly, so unconfigured selectors and the
    legacy ``provider/model`` form keep working exactly as before.

    A selector that names an alias, whether from ``config.yml`` or the
    ``model_aliases`` table, is first substituted with the alias target, then
    resolved as usual; the resulting ``ResolvedProvider`` carries ``alias`` so
    response ``model`` fields can be relabeled.

    Raises ``ValueError`` / ``AnyLLMError`` (from any-llm) for a selector that
    names neither a configured instance nor a known provider, mirroring the
    prior ``AnyLLM.split_model_provider`` behavior.
    """
    alias = resolve_effective_alias(config, model_selector)
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

    provider, model = AnyLLM.split_model_provider(selector)
    return ResolvedProvider(
        instance=provider.value,
        provider=provider,
        model=model,
        kwargs=get_provider_kwargs(config, provider, instance=provider.value),
        alias=model_selector if alias is not None else None,
    )


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
    return f"{provider.value}:{model}"
