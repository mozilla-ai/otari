"""The catalog of providers a search or fetch tool may name, read from the libraries.

What a provider is comes from the metadata any-search and any-fetch publish, so
a provider a library adds is listed with no change to the gateway. What the
deployment does with it, its tools on each provider and an endpoint a tool
inherits from its settings, is shown only to a caller who operates the
deployment.
"""

from collections.abc import Mapping

import any_fetch
import any_search
from gateway.core.config import GatewayConfig
from gateway.core.settings.tools import (
    SEARCH_PROVIDERS_REQUIRING_API_BASE,
    SEARCH_PROVIDERS_REQUIRING_API_KEY,
    SEARCH_PROVIDERS_WITHOUT_ADAPTER,
    ToolInstance,
    ToolKind,
    default_api_base,
    effective_fetch_instances,
    effective_search_instances,
    supported_fetch_providers,
)
from gateway.schemas.tools import SearchProviderOptionSchema, SearchProviderSchema


def search_provider_catalog(config: GatewayConfig, kind: ToolKind, *, operates: bool) -> list[SearchProviderSchema]:
    """The providers a ``kind`` tool may name, sorted by id.

    Providers that exist only for tests are left out, and so is the fetch
    provider ``builtin``, which only the implicit ``builtin_fetch`` tool uses.
    ``operates`` is whether the caller operates the deployment: anyone else
    gets no tool names, and the library's endpoint where the deployment's
    settings supply another.
    """
    if kind == "fetch":
        return _fetch_providers(config, operates=operates)
    return _search_providers(config, operates=operates)


def _instance_names(instances: Mapping[str, ToolInstance]) -> dict[str, list[str]]:
    """The instances' names, sorted, by the provider each runs on."""
    names: dict[str, list[str]] = {}
    for name, instance in sorted(instances.items()):
        names.setdefault(instance.provider, []).append(name)
    return names


def _option_schemas(
    options: list[any_search.OptionSpec] | list[any_fetch.OptionSpec],
) -> list[SearchProviderOptionSchema]:
    """A library's option specs, as the catalog serves them."""
    return [SearchProviderOptionSchema(**option.model_dump()) for option in options]


def _search_providers(config: GatewayConfig, *, operates: bool) -> list[SearchProviderSchema]:
    """any-search's providers, and those it has no adapter for yet."""
    instances = _instance_names(effective_search_instances(config)) if operates else {}
    served = {
        provider: any_search.AnySearch.get_provider_metadata(provider)
        for provider in any_search.AnySearch.get_supported_providers()
    }
    served = {provider: metadata for provider, metadata in served.items() if metadata.tier != "test"}
    entries = []
    for provider in sorted({*served, *SEARCH_PROVIDERS_WITHOUT_ADAPTER}):
        metadata = served.get(provider)
        library_base = metadata.default_api_base if metadata is not None else None
        # An endpoint other than the library's comes from this deployment's
        # settings, such as the web_search_url a searxng tool inherits, so only
        # an operator is shown the one a tool would really use.
        api_base = default_api_base(config, provider) if operates else library_base
        if metadata is None:
            # No adapter, so no metadata and no option schema: the gateway's own
            # tables say what a tool on it needs, so the form can offer it.
            entries.append(
                SearchProviderSchema(
                    id=provider,
                    kind="search",
                    requires_api_key=provider in SEARCH_PROVIDERS_REQUIRING_API_KEY,
                    requires_api_base=provider in SEARCH_PROVIDERS_REQUIRING_API_BASE,
                    default_api_base=api_base,
                    instances=instances.get(provider, []),
                )
            )
            continue
        entries.append(
            SearchProviderSchema(
                id=provider,
                kind="search",
                requires_api_key=metadata.requires_api_key,
                requires_api_base=metadata.requires_api_base,
                default_api_base=api_base,
                doc_url=metadata.doc_url,
                tier=metadata.tier,
                options=_option_schemas(metadata.options),
                instances=instances.get(provider, []),
                max_results=metadata.max_results,
                query_in_url=metadata.query_in_url,
                key_in_url=metadata.key_in_url,
            )
        )
    return entries


def _fetch_providers(config: GatewayConfig, *, operates: bool) -> list[SearchProviderSchema]:
    """any-fetch's providers, except ``builtin``, which only the implicit builtin_fetch instance runs on."""
    instances = _instance_names(effective_fetch_instances(config)) if operates else {}
    entries = []
    for provider in supported_fetch_providers():
        metadata = any_fetch.AnyFetch.get_provider_metadata(provider)
        if metadata.tier == "test":
            continue
        entries.append(
            SearchProviderSchema(
                id=provider,
                kind="fetch",
                requires_api_key=metadata.requires_api_key,
                requires_api_base=metadata.requires_api_base,
                default_api_base=metadata.default_api_base,
                doc_url=metadata.doc_url,
                tier=metadata.tier,
                options=_option_schemas(metadata.options),
                instances=instances.get(provider, []),
                max_urls_per_call=metadata.max_urls_per_call,
                renders_javascript=metadata.renders_javascript,
                formats=list(metadata.formats),
            )
        )
    return sorted(entries, key=lambda entry: entry.id)
