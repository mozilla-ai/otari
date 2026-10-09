"""Connection tests for search and fetch instances: one search or one fetch, reported without its content.

An instance is tested as the resolver will call it: a provider any-search or
any-fetch serves through the library, with a client of its own, and a
``searxng`` instance, until any-search has its adapter, through the direct
endpoint's own client. The answer says whether the call worked, the error's tag
when it did not, and how many hits or characters came back.
"""

from collections.abc import Mapping
from typing import Any

import any_fetch
import any_search
from gateway.core.config import GatewayConfig
from gateway.core.settings.tools import (
    BUILTIN_FETCH,
    SEARCH_PROVIDERS_WITHOUT_ADAPTER,
    ToolInstance,
    ToolKind,
    effective_fetch_instances,
    effective_search_instances,
    instance_from_entry,
)
from gateway.exceptions.tools_exceptions import (
    SearchToolNotFoundError,
    SearchToolRefusedError,
    SearchToolUntestableError,
)
from gateway.log_config import logger
from gateway.schemas.tools import SearchToolTestResponse
from gateway.services.search_backend import (
    SearchProviderError,
    SearchQuery,
    SearchToolError,
    resolve_search_tool,
    run_search,
)
from gateway.services.tools._search_tool_writes import check_tool_entry, check_tool_name_is_free, check_tool_write


def instance_to_test(config: GatewayConfig, name: str) -> tuple[ToolInstance, Mapping[str, Any]]:
    """The configured or stored instance called ``name``, and its entry.

    A search instance wins, as it does in the maps; ``builtin_fetch`` has no
    test yet.
    """
    if (search := effective_search_instances(config).get(name)) is not None:
        return search, config.search_tools[name]
    if name == BUILTIN_FETCH:
        raise SearchToolUntestableError(name)
    if (fetch := effective_fetch_instances(config).get(name)) is not None:
        return fetch, config.fetch_tools[name]
    raise SearchToolNotFoundError(name)


def unsaved_instance_to_test(
    config: GatewayConfig, name: str, entry: Mapping[str, Any], *, kind: ToolKind
) -> ToolInstance:
    """An instance not saved yet, held to the checks its create would meet.

    So a test never passes what cannot be saved. Checked against the instances
    this worker has loaded.
    """
    check_tool_entry(name, entry, kind=kind, inherited_api_base=config.web_search_url)
    check_tool_write(
        config,
        name,
        entry,
        kind=kind,
        sets_name=True,
        sets_options=entry.get("options") is not None,
        sets_fetch_tool=True,
    )
    check_tool_name_is_free(config, name, kind)
    return instance_from_entry(kind, name, entry)


def check_test_input(instance: ToolInstance, *, query: str | None, url: str | None) -> None:
    """A search instance is tested with a query, a fetch instance with a URL."""
    if instance.kind == "search" and not query:
        raise SearchToolRefusedError("A search instance is tested with a 'query'.")
    if instance.kind == "fetch" and not url:
        raise SearchToolRefusedError("A fetch instance is tested with a 'url'.")


def _library_call(instance: ToolInstance) -> tuple[dict[str, Any], dict[str, Any]]:
    """What a library provider is built and called with: the instance's settings and options as the resolver sends them.

    The instance's own key and base URL, never the environment's: the libraries
    read a provider's variable when none is passed, and an empty one counts as
    passed. Options carry otari's defaults below them, and leave out the ones
    the rules drop.
    """
    default_timeout = any_search.DEFAULT_TIMEOUT if instance.kind == "search" else any_fetch.DEFAULT_TIMEOUT
    settings = {
        "api_key": instance.api_key or "",
        "api_base": instance.api_base or "",
        "timeout": instance.timeout or default_timeout,
    }
    kept = {key: value for key, value in instance.options.items() if key not in instance.dropped_options}
    return settings, {**instance.provider_defaults, **kept}


async def _test_with_library(instance: ToolInstance, *, query: str | None, url: str | None) -> SearchToolTestResponse:
    """One search or fetch through any-search or any-fetch, the provider opening a client of its own."""
    settings, options = _library_call(instance)
    try:
        if instance.kind == "search":
            async with any_search.AnySearch.create(instance.provider, **settings) as engine:
                result = await engine.search(str(query), **options)
            if result.error is not None:
                return SearchToolTestResponse(ok=False, error=result.error.tag)
            return SearchToolTestResponse(ok=True, hits=len(result.hits))
        async with any_fetch.AnyFetch.create(instance.provider, **settings) as fetcher:
            page = await fetcher.fetch(str(url), **options)
        if page.error is not None:
            return SearchToolTestResponse(ok=False, error=page.error.tag)
        return SearchToolTestResponse(ok=True, characters=len(page.text))
    except (any_search.ProviderError, any_fetch.ProviderError) as exc:
        return SearchToolTestResponse(ok=False, error=exc.tag)
    except (any_search.AnySearchError, any_fetch.AnyFetchError) as exc:
        # A key or option the provider cannot run with: the instance's to fix.
        raise SearchToolRefusedError(str(exc)) from None


async def _test_with_old_client(
    config: GatewayConfig, name: str, entry: Mapping[str, Any], query: str
) -> SearchToolTestResponse:
    """One search through the direct endpoint's own client, which serves SearXNG until any-search does."""
    try:
        # Resolved as the direct endpoint resolves it, against this entry alone,
        # so an unsaved one inherits web_search_url and the engines the same way.
        tool = resolve_search_tool(config.model_copy(update={"search_tools": {name: dict(entry)}}), name)
    except SearchToolError as exc:
        raise SearchToolRefusedError(str(exc)) from None
    try:
        outcome = await run_search(tool, SearchQuery(query=query))
    except SearchProviderError as exc:
        return SearchToolTestResponse(ok=False, error=exc.tag)
    return SearchToolTestResponse(ok=True, hits=len(outcome.results))


async def run_connection_test(
    config: GatewayConfig,
    instance: ToolInstance,
    entry: Mapping[str, Any],
    *,
    query: str | None,
    url: str | None,
) -> SearchToolTestResponse:
    """Run one search or one fetch on ``instance`` and say how it went, never what came back."""
    if instance.kind == "search" and instance.provider in SEARCH_PROVIDERS_WITHOUT_ADAPTER:
        response = await _test_with_old_client(config, instance.name, entry, str(query))
    else:
        response = await _test_with_library(instance, query=query, url=url)
    logger.info(
        "Connection test of %s instance '%s': %s",
        instance.kind,
        instance.name,
        "ok" if response.ok else f"failed ({response.error})",
    )
    return response
