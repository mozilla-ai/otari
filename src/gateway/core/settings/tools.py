"""The tools domain's settings.

Only the web search and fetch settings have moved here so far: the fields, the
named instances and the defaults. The rest of the domain's settings, the code
execution sandbox's among them, are still on ``GatewayConfig`` in
``core/config.py``.

A deployment names its search and fetch backends as instances: ``search_tools``
and ``fetch_tools`` in the configuration file, and search instances stored
through ``/api/v1/search-tools``. The legacy in-loop settings, the provider pair
and ``web_search_url``, describe one more search backend, the synthesized
instance, which serves the in-loop tool only and is not part of either map.

This module answers settings questions only: which instances exist, which one
the in-loop tool defaults to, which fetch instance enriches a search, and how
many calls a request may make.

The checks run in two places. :meth:`ToolSettings.validate_search_tools` and
:meth:`ToolSettings.validate_fetch_tools` refuse at load what no deployment can
have relied on. :func:`warn_about_tool_instances` runs once stored rows and
runtime settings have loaded, and warns about what loads anyway: a name or an
option that breaks the rules, a default or ``fetch_tool`` that names nothing,
and several search instances with no default between them.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, PrivateAttr, field_validator

import any_fetch
import any_search
from gateway.core.env import otari_env
from gateway.core.settings_view import OMITTED, SECRET, SettingsGroup, Shown
from gateway.log_config import logger

# Search providers the standalone POST /api/v1/search endpoint's own clients can
# dispatch to, and the ones its provider catalog lists. Declared here rather than
# in the adapter module so the config layer never imports the service layer.
# Which providers a ``search_tools`` entry may name is wider: see
# :func:`supported_search_providers`.
SEARCH_PROVIDERS = ("exa", "searxng")
# Licensed search APIs the in-loop ``otari_web_search`` backend can call
# directly, as an alternative to pointing ``web_search_url`` at a SearXNG-shaped
# service. Declared here for the same reason as SEARCH_PROVIDERS above: startup
# validation rejects an unknown ``web_search_provider`` without the config layer
# importing `gateway.services.web_search_providers`, which imports this name.
WEB_SEARCH_PROVIDERS = ("tavily", "brave")
# Providers that authenticate with an API key, for the direct endpoint's own
# clients and its catalog. Validation asks the library's metadata instead, so a
# provider any-search adds is held to its own requirement.
SEARCH_PROVIDERS_REQUIRING_API_KEY = ("exa",)
# Providers with no endpoint of their own to default to, so the tool has to say
# where the backend is. The only one today is ``searxng``, which speaks the same
# wire contract as the in-loop otari_web_search backend and therefore inherits
# ``web_search_url`` when the tool declares no ``api_base``.
SEARCH_PROVIDERS_REQUIRING_API_BASE = ("searxng",)
# Accepted in ``search_tools`` although any-search has no adapter for it: the
# direct endpoint serves it through its own client, its options passed unchecked
# since there is no schema to check them against. It needs no key.
SEARCH_PROVIDERS_WITHOUT_ADAPTER = ("searxng",)

# The fetch instance that always exists: the built-in fetcher, on any-fetch's
# ``builtin`` provider, which no declared instance may use.
BUILTIN_FETCH = "builtin_fetch"
BUILTIN_FETCH_PROVIDER = "builtin"
# The value of ``web_search_default_tool`` that turns in-loop search off.
NO_SEARCH_DEFAULT = "none"
# Names no instance may take: each already means something to a default setting.
RESERVED_INSTANCE_NAMES = frozenset({BUILTIN_FETCH, NO_SEARCH_DEFAULT})

DEFAULT_WEB_SEARCH_MAX_CALLS = 10
# The engines the old in-loop backend asks a SearXNG service for when
# ``web_search_engines`` is unset; why these four is said where it uses them.
DEFAULT_SEARXNG_ENGINES = ("duckduckgo", "mojeek", "qwant", "wikipedia")
# What otari has always sent these providers, applied below every other option
# layer so that moving a provider onto its library adapter changes nothing: GET
# for SearXNG, which some SearXNG-shaped backends require, and page text from
# Tavily, so its hits need no enrichment. An instance that sets the key wins.
PROVIDER_DEFAULT_OPTIONS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        "searxng": MappingProxyType({"method": "get"}),
        "tavily": MappingProxyType({"include_raw_content": True}),
    }
)
_NO_OPTIONS: Mapping[str, Any] = MappingProxyType({})

ToolKind = Literal["search", "fetch"]


def supported_search_providers() -> tuple[str, ...]:
    """The providers a ``search_tools`` entry may name."""
    return (*any_search.AnySearch.get_supported_providers(), *SEARCH_PROVIDERS_WITHOUT_ADAPTER)


def supported_fetch_providers() -> tuple[str, ...]:
    """The providers a ``fetch_tools`` entry may name: any-fetch's, except ``builtin``."""
    return tuple(
        provider for provider in any_fetch.AnyFetch.get_supported_providers() if provider != BUILTIN_FETCH_PROVIDER
    )


def _is_test_tier(kind: ToolKind, provider: str) -> bool:
    """Whether the provider exists for tests: accepted in configuration, never offered."""
    if kind == "fetch":
        return any_fetch.AnyFetch.get_provider_metadata(provider).tier == "test"
    if provider in SEARCH_PROVIDERS_WITHOUT_ADAPTER:
        return False
    return any_search.AnySearch.get_provider_metadata(provider).tier == "test"


def _requires_api_key(kind: ToolKind, provider: str) -> bool:
    if kind == "fetch":
        return any_fetch.AnyFetch.get_provider_metadata(provider).requires_api_key
    if provider in SEARCH_PROVIDERS_WITHOUT_ADAPTER:
        return False
    return any_search.AnySearch.get_provider_metadata(provider).requires_api_key


def _option_enums(kind: ToolKind, provider: str) -> dict[str, list[str] | None] | None:
    """The provider's option names, each with the values it is limited to, if any.

    ``None`` when the provider has no schema to check against: one any-search
    has no adapter for yet.
    """
    if kind == "fetch":
        if provider not in any_fetch.AnyFetch.get_supported_providers():
            return None
        return {spec.name: spec.enum for spec in any_fetch.AnyFetch.get_provider_metadata(provider).options}
    if provider not in any_search.AnySearch.get_supported_providers():
        return None
    return {spec.name: spec.enum for spec in any_search.AnySearch.get_provider_metadata(provider).options}


def option_problems(kind: ToolKind, provider: str, options: Mapping[str, Any]) -> dict[str, str]:
    """Each option the provider's schema does not know or refuses the value of, with why.

    Only names and the schema's fixed lists of values are checked, not types:
    a schema's type says what a provider usually takes, and some take more (an
    Exa ``contents`` is an object, or ``false`` for none).
    """
    enums = _option_enums(kind, provider)
    if enums is None:
        return {}
    problems: dict[str, str] = {}
    for key, value in options.items():
        if key not in enums:
            problems[key] = f"option '{key}' is not one the provider knows"
        elif (allowed := enums[key]) is not None and value not in allowed:
            problems[key] = f"option '{key}' has a value the provider refuses"
    return problems


def instance_name_problems(name: str) -> list[str]:
    """What breaks the name rules beyond the empty name and ``/``, which are always refused.

    ``<provider>:<instance>`` is a pricing key, so a name carries no colon; a
    name is a path segment of ``/api/v1/search/{tool}``, so it carries no slash.
    """
    problems: list[str] = []
    if ":" in name:
        problems.append("its name contains ':'")
    if name in RESERVED_INSTANCE_NAMES:
        problems.append(f"its name is reserved ({', '.join(sorted(RESERVED_INSTANCE_NAMES))})")
    return problems


def _validate_transport(section: str, name: str, api_base: Any, api_key: Any) -> None:
    if not api_key or not api_base:
        return
    try:
        scheme = urlsplit(str(api_base).strip()).scheme.lower()
    except ValueError:
        scheme = ""
    if scheme != "https":
        msg = f"{section}.{name}.api_base must use https when api_key is set."
        raise ValueError(msg)


def validate_search_tool_transport(name: str, api_base: Any, api_key: Any) -> None:
    """Require encrypted transport when a search tool carries a credential."""
    _validate_transport("search_tools", name, api_base, api_key)


def _validate_tool_entry(kind: ToolKind, name: str, entry: Any) -> str:
    """The checks a search and a fetch entry share; returns the entry's provider."""
    section = "search_tools" if kind == "search" else "fetch_tools"
    if not name:
        msg = f"{kind} tool name must not be empty."
        raise ValueError(msg)
    if "/" in name:
        msg = f"{kind} tool name '{name}' must not contain '/' (it is used as a URL path segment)."
        raise ValueError(msg)
    if not isinstance(entry, dict):
        msg = f"{section}.{name} must be a mapping."
        raise ValueError(msg)
    provider = str(entry.get("provider") or name)
    supported = supported_search_providers() if kind == "search" else supported_fetch_providers()
    if provider not in supported:
        listed = sorted(candidate for candidate in supported if not _is_test_tier(kind, candidate))
        msg = (
            f"{section}.{name}.provider '{provider}' is not a supported {kind} provider (one of: {', '.join(listed)})."
        )
        raise ValueError(msg)
    if _requires_api_key(kind, provider) and not entry.get("api_key"):
        msg = f"{section}.{name}.api_key is required for provider '{provider}'."
        raise ValueError(msg)
    _validate_transport(section, name, entry.get("api_base"), entry.get("api_key"))
    timeout = entry.get("timeout")
    if timeout is not None:
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            msg = f"{section}.{name}.timeout must be a number of seconds."
            raise ValueError(msg)
        # A negative timeout would reach httpx and fail at request time, and a
        # zero is silently swapped for the default when the tool is resolved.
        # Both are misconfigurations worth failing on here.
        if timeout <= 0:
            msg = f"{section}.{name}.timeout must be greater than 0 seconds, got {timeout}."
            raise ValueError(msg)
    options = entry.get("options")
    if options is not None and not isinstance(options, dict):
        msg = f"{section}.{name}.options must be a mapping."
        raise ValueError(msg)
    return provider


def validate_search_tool_entry(name: str, entry: Any) -> None:
    """Validate one ``search_tools`` entry, raising ``ValueError`` on any problem.

    Module-level rather than a method so the runtime CRUD path
    (``/api/v1/search-tools``) can hold a dashboard-written tool to the same rules
    the config file is held to at startup, instead of restating them.

    The provider is one any-search serves, or ``searxng`` until it has an
    adapter. Whether it needs a key is the library's metadata's answer; a keyless
    provider (a self-hosted SearXNG or an adapter fronting one) is allowed to
    declare none. The tool name doubles as a ``/api/v1/search/{tool}`` path
    segment, so it must not contain a slash.

    The rest of the name rules and the option checks are not here: an entry that
    breaks them still loads, with a warning (:func:`warn_about_tool_instances`).
    A missing backend URL is deliberately not fatal here either; see
    :meth:`ToolSettings.search_tools_without_backend_url`.
    """
    _validate_tool_entry("search", name, entry)
    fetch_tool = entry.get("fetch_tool")
    if fetch_tool is not None and not (isinstance(fetch_tool, str) and fetch_tool.strip()):
        msg = f"search_tools.{name}.fetch_tool must name a fetch instance."
        raise ValueError(msg)


def validate_fetch_tool_entry(name: str, entry: Any) -> None:
    """Validate one ``fetch_tools`` entry, raising ``ValueError`` on any problem.

    The provider is one any-fetch serves, other than ``builtin``, which only the
    implicit ``builtin_fetch`` instance uses. The map is new, so nothing in it
    predates the name rules: unlike a search entry's, they are enforced here.
    """
    _validate_tool_entry("fetch", name, entry)
    if problems := instance_name_problems(name):
        msg = f"fetch_tools.{name} is refused: {'; '.join(problems)}."
        raise ValueError(msg)


@dataclass(frozen=True)
class ToolInstance:
    """One search or fetch instance, as the resolver will read it.

    ``library_backed`` is whether any-search or any-fetch serves the provider;
    ``dropped_options`` are the option keys the provider's schema does not know
    or refuses the value of, which load with a warning, for the resolver to
    leave out of a call. ``provider_defaults`` are :data:`PROVIDER_DEFAULT_OPTIONS`' entry for
    the provider, below every other option layer. Both option maps are read-only
    copies, so a reader cannot change the configuration they came from, and are
    left out of the hash.
    """

    name: str
    kind: ToolKind
    provider: str
    library_backed: bool
    api_key: str | None = field(default=None, repr=False)
    api_base: str | None = field(default=None, repr=False)
    timeout: float | None = None
    options: Mapping[str, Any] = field(default=_NO_OPTIONS, repr=False, hash=False)
    dropped_options: frozenset[str] = frozenset()
    provider_defaults: Mapping[str, Any] = field(default=_NO_OPTIONS, hash=False)
    fetch_tool: str | None = None


@dataclass(frozen=True)
class SynthesizedSearchInstance:
    """The in-loop search backend the legacy settings describe.

    It is not in either map, so it cannot be named by a default or by the direct
    endpoint's path, and it is priced under ``otari:web_search``. Its name is
    its provider's. Whether a ``searxng`` one at the platform's origin is the
    hosted hop is the resolver's question, not this one's.
    """

    provider: str
    api_key: str | None = field(default=None, repr=False)
    api_base: str | None = field(default=None, repr=False)
    engines: tuple[str, ...] = ()
    provider_defaults: Mapping[str, Any] = field(default=_NO_OPTIONS, hash=False)

    @property
    def name(self) -> str:
        return self.provider


class ToolSettings(BaseModel):
    """The tools domain's settings, mixed into ``GatewayConfig``. So far only the web tools' have moved here.

    For web search and fetch: the named instances, the legacy in-loop settings,
    the defaults and the call limit.
    """

    search_tools: Annotated[dict[str, dict[str, Any]], OMITTED] = Field(
        default_factory=dict,
        description=(
            "Named search instances, for both POST /api/v1/search (by name, in 'search_tool_name' or "
            "the /api/v1/search/{tool} path) and the in-loop otari_web_search tool (see "
            "web_search_default_tool). Each entry may declare a 'provider' (defaults to the name; "
            "GET /api/v1/search-tools/providers lists them), an 'api_key' where the provider needs "
            "one, an 'api_base' (a searxng entry without one inherits web_search_url), a 'timeout' "
            "in seconds, an 'options' mapping of provider-native defaults, and a 'fetch_tool' naming "
            "the fetch instance that enriches its results."
        ),
    )
    fetch_tools: Annotated[dict[str, dict[str, Any]], OMITTED] = Field(
        default_factory=dict,
        description=(
            "Named fetch instances, for the otari_web_fetch tool and for enriching search results. "
            "Each entry may declare a 'provider' (defaults to the name), an 'api_key' where the "
            "provider needs one, an 'api_base', a 'timeout' in seconds and an 'options' mapping. "
            "'builtin_fetch', the built-in fetcher, always exists and cannot be declared. A name is "
            "unique across search_tools and fetch_tools."
        ),
    )
    web_fetch_enabled: Annotated[bool, Shown(SettingsGroup.TOOLS)] = Field(
        default=False,
        description=(
            "Whether Otari may execute the managed otari_web_fetch tool. Off by default because "
            "enabling it permits model-directed outbound requests to public web destinations."
        ),
    )
    web_search_url: Annotated[str | None, Shown(SettingsGroup.TOOLS)] = Field(
        default=None,
        description=(
            "Base URL of the web-search backend (SearXNG instance or a search adapter) for "
            "otari_web_search tools. When unset, otari_web_search requests are rejected with 400. "
            "docker-compose sets this to the bundled SearXNG container."
        ),
    )
    web_search_provider: Annotated[str | None, Shown(SettingsGroup.TOOLS)] = Field(
        default=None,
        description=(
            "Licensed search API the web-search backend calls directly ('tavily' or 'brave'), "
            "instead of the SearXNG-shaped service web_search_url names. Requires "
            "web_search_provider_api_key. When both are set, web_search_url is not needed."
        ),
    )
    web_search_provider_api_key: Annotated[str | None, SECRET] = Field(
        default=None,
        description=(
            "Credential for web_search_provider. Held by whichever process runs the search: on a "
            "hosted deployment that is the control plane, never the data plane."
        ),
    )
    web_search_backend_token: Annotated[str | None, SECRET] = Field(
        default=None,
        description=(
            "Shared secret GET /api/v1/web-search/search requires as X-Gateway-Token. Set on a hosted "
            "control plane so its data-plane gateway can search through it; without it the route "
            "is not served, because it spends the deployment's own search quota. The gateway "
            "presents its platform token (OTARI_AI_TOKEN) and nothing else, so this must be that "
            "token, and rotating it stops web search for that data plane until both are updated."
        ),
    )
    web_search_purpose_hint: Annotated[str | None, Shown(SettingsGroup.TOOLS)] = Field(
        default=None,
        description=(
            "Default purpose hint for the web-search backend when an otari_web_search tool entry "
            "does not supply its own."
        ),
    )
    web_search_engines: Annotated[str | None, Shown(SettingsGroup.TOOLS)] = Field(
        default=None,
        description=(
            "Comma-separated SearXNG engine list for the web-search backend (e.g. 'google,bing'). "
            "When unset, the backend default engines are used."
        ),
    )
    web_search_max_results: Annotated[int | None, Shown(SettingsGroup.TOOLS)] = Field(
        default=None,
        ge=1,
        description=(
            "Default cap on the number of hits returned by the web-search backend (a per-tool "
            "max_results still overrides it)."
        ),
    )
    web_search_extract: Annotated[bool | None, Shown(SettingsGroup.TOOLS)] = Field(
        default=None,
        description=(
            "Whether the web-search backend extracts page content in-process (True) or returns "
            "snippet-only results (False). When unset, the backend default (extraction on) applies."
        ),
    )
    web_search_intercept: Annotated[bool | None, Shown(SettingsGroup.TOOLS)] = Field(
        default=None,
        description=(
            "Whether a provider-named web-search declaration (bare 'web_search', Anthropic-native "
            "'web_search_<date>') is run against the gateway's own backend instead of being forwarded "
            "to the provider. Off when unset: the explicit otari_web_search type is always run by the "
            "gateway, and every other keyword reaches the provider untouched. Requires a search backend."
        ),
    )
    web_search_allow_private_hosts: Annotated[bool, Shown(SettingsGroup.TOOLS)] = Field(
        default=False,
        description=(
            "SSRF gate: allow the web-search backend to fetch private/loopback/reserved hosts. "
            "Off by default. Only enable for unusual setups such as a private search index."
        ),
    )
    web_retrieval_trust_env_proxy: Annotated[bool, Shown(SettingsGroup.TOOLS)] = Field(
        default=False,
        description=(
            "Trust HTTP_PROXY, HTTPS_PROXY, and ALL_PROXY for web retrieval. The proxy must enforce "
            "address safety when resolving and connecting to destinations. Local URL, domain, and "
            "address checks remain enabled; direct requests, including NO_PROXY matches, remain IP-pinned. "
            "Off by default. Only enable for an operator-controlled SSRF-filtering proxy."
        ),
    )
    web_search_default_tool: Annotated[str | None, Shown(SettingsGroup.TOOLS)] = Field(
        default=None,
        description=(
            "The search instance the in-loop otari_web_search tool uses when no organization key "
            "applies, and the one an unnamed POST /api/v1/search call uses. One whose provider "
            "any-search serves, or 'none' to turn in-loop search off. When unset: the instance the "
            "legacy web_search_provider or web_search_url settings describe, else the only search "
            "instance, if there is exactly one. Not in effect yet: see docs/configuration.md."
        ),
    )
    web_fetch_default_tool: Annotated[str | None, Shown(SettingsGroup.TOOLS)] = Field(
        default=None,
        description=(
            "The fetch instance for the otari_web_fetch tool and for enriching search results, "
            "unless a search instance names its own fetch_tool. When unset: builtin_fetch, the "
            "built-in fetcher. Not in effect yet: see docs/configuration.md."
        ),
    )
    web_search_max_calls: Annotated[int | None, Shown(SettingsGroup.TOOLS)] = Field(
        default=None,
        ge=1,
        description=(
            "How many search and fetch tool calls one request may make, together. A call past it "
            "gets a tool error the model can recover from. When unset: 10. Not in effect yet: see "
            "docs/configuration.md."
        ),
    )

    # The config-file tools as loaded, before any dashboard-stored tool is
    # overlaid by ``search_tool_store_service``.
    _search_tool_baseline: dict[str, dict[str, Any]] | None = PrivateAttr(default=None)

    def web_search_provider_configured(self) -> bool:
        """Whether a licensed search API is configured for the in-loop tool.

        Both halves, because either alone runs no search: a provider with no key
        cannot authenticate, and a key with no provider names nothing to send it
        to. When this is true the deployment can search without
        ``web_search_url``, which is what lets it drop the adapter container that
        used to sit between the two.
        """
        return bool(self.web_search_provider) and bool(self.web_search_provider_api_key)

    def web_search_configured(self) -> bool:
        """Whether this deployment can run ``otari_web_search`` at all.

        The one question the request path, the per-workspace page and the tool
        catalog all ask, so they cannot disagree about whether a workspace's
        stored configuration governs anything.

        ``web_search_url`` is read through ``otari_env`` as well as off the
        field, matching every call site this replaces: the field is env-bridged,
        so the two agree, and dropping the read here would quietly narrow what
        counts as configured.
        """
        return bool(self.web_search_url or otari_env("WEB_SEARCH_URL")) or self.web_search_provider_configured()

    def search_tool_providers(self) -> set[str]:
        """The distinct providers backing the configured search tools.

        These are prefixes of the ``<provider>:<tool>`` keys that search pricing,
        usage, and per-key access lists are written against, so the allow-list
        writer has to accept them alongside real provider instances.
        """
        return {str(entry.get("provider") or name) for name, entry in self.search_tools.items()}

    def search_tools_without_backend_url(self) -> list[str]:
        """Search tools whose provider needs an ``api_base`` and has none to inherit.

        Reported as a startup warning rather than raised by
        :meth:`validate_search_tools`, because ``web_search_url`` (which a
        ``searxng`` tool inherits) can also come from a dashboard-stored
        override, and those are applied to the config after it loads. Failing at
        load time would refuse to boot a gateway the operator has in fact
        configured. Enforcement is per request instead: ``resolve_search_tool``
        refuses such a tool with a 400, and the rest of the gateway serves.
        """
        if self.web_search_url:
            return []
        return [
            name
            for name, entry in self.search_tools.items()
            if isinstance(entry, dict)
            and str(entry.get("provider") or name) in SEARCH_PROVIDERS_REQUIRING_API_BASE
            and not entry.get("api_base")
        ]

    def warn_about_half_configured_web_search(self) -> None:
        """Say so when a search provider was named but cannot be used.

        Also when ``web_search_backend_token`` was set without one: the token
        exists to gate the backend route, and that route is not mounted without
        a provider to serve it, so the setting silently does nothing.

        A warning rather than a refusal, for the reason
        :meth:`GatewayConfig.warn_about_half_configured_oauth` gives: web search
        is one optional tool, and refusing to boot would take a gateway offline
        over it. A deployment naming a provider it has no key for is also the
        ordinary state of one that has not filled the key in yet, and a compose
        file can default the name without being able to default the secret.

        But the failure is otherwise completely silent. Neither half is read
        without the other, so ``web_search_configured`` falls through to
        ``web_search_url``, and a deployment that named a provider precisely so
        it would need no URL answers every ``otari_web_search`` request with the
        not-configured 400 and says nowhere why.
        """
        if self.web_search_backend_token and not self.web_search_provider_configured():
            logger.warning(
                "web_search_backend_token is set but no web-search provider is configured, so "
                "GET /api/v1/web-search/search is not served. Set web_search_provider and "
                "web_search_provider_api_key on the process that holds the search key."
            )
        if bool(self.web_search_provider) == bool(self.web_search_provider_api_key):
            return
        missing, present = (
            ("web_search_provider_api_key", f"web_search_provider is {self.web_search_provider!r}")
            if self.web_search_provider
            else ("web_search_provider", "web_search_provider_api_key is set")
        )
        logger.warning(
            "Web search through a licensed provider is configured but will not run: %s, and %s is not set. "
            "Set both, or neither and point web_search_url at a SearXNG-shaped backend instead.",
            present,
            missing,
        )

    def validate_search_tools(self) -> None:
        """Validate the ``search_tools`` map at startup so misconfig fails fast.

        Per-entry rules live in :func:`validate_search_tool_entry`, which the
        runtime CRUD path applies to a dashboard-written tool as well.
        """
        for name, entry in self.search_tools.items():
            validate_search_tool_entry(name, entry)
            if not isinstance(entry, dict):
                continue
            provider = str(entry.get("provider") or name)
            if provider in SEARCH_PROVIDERS_REQUIRING_API_BASE and not entry.get("api_base"):
                validate_search_tool_transport(name, self.web_search_url, entry.get("api_key"))

    def validate_fetch_tools(self) -> None:
        """Validate the ``fetch_tools`` map at startup.

        Run at load, before stored search tools are overlaid, so a fetch entry
        named like a configured search entry stops startup here. One named like
        a stored search tool loads, and is left out of the fetch map with an
        error (:func:`warn_about_tool_instances`).
        """
        for name, entry in self.fetch_tools.items():
            validate_fetch_tool_entry(name, entry)
            if name in self.search_tools:
                msg = (
                    f"fetch_tools.{name} is refused: search_tools has an instance of the same name, and "
                    "names are unique across the two maps."
                )
                raise ValueError(msg)

    @field_validator("web_search_provider")
    @classmethod
    def _validate_web_search_provider(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if not normalized:
            return None
        if normalized not in WEB_SEARCH_PROVIDERS:
            msg = f"web_search_provider must be one of {sorted(WEB_SEARCH_PROVIDERS)}, got '{value}'"
            raise ValueError(msg)
        return normalized


def default_api_base(settings: ToolSettings, provider: str) -> str | None:
    """The base URL a search instance inherits when it declares no ``api_base``.

    A ``searxng`` tool falls back to ``web_search_url``, the backend the in-loop
    ``otari_web_search`` tool already speaks to over the same contract, so a
    deployment that runs one exposes it on ``POST /v1/search`` with a single
    ``provider: searxng`` line. Read off the field alone: only the in-loop reads
    fall back to the environment. Any other provider takes any-search's own.

    Public because the dashboard's add-a-search-tool form shows the same value as
    the placeholder for an omitted ``api_base``.
    """
    if provider in SEARCH_PROVIDERS_WITHOUT_ADAPTER:
        return (settings.web_search_url or "").strip() or None
    if provider not in any_search.AnySearch.get_supported_providers():
        return None
    return any_search.AnySearch.get_provider_metadata(provider).default_api_base


def _instance(kind: ToolKind, name: str, entry: Mapping[str, Any], *, library_backed: bool) -> ToolInstance:
    provider = str(entry.get("provider") or name)
    options = entry.get("options")
    options = options if isinstance(options, dict) else {}
    timeout = entry.get("timeout")
    fetch_tool = entry.get("fetch_tool") if kind == "search" else None
    return ToolInstance(
        name=name,
        kind=kind,
        provider=provider,
        library_backed=library_backed,
        api_key=entry.get("api_key") or None,
        api_base=entry.get("api_base") or None,
        timeout=float(timeout) if isinstance(timeout, (int, float)) and not isinstance(timeout, bool) else None,
        options=MappingProxyType(dict(options)),
        dropped_options=frozenset(option_problems(kind, provider, options)),
        provider_defaults=PROVIDER_DEFAULT_OPTIONS.get(provider, _NO_OPTIONS),
        fetch_tool=fetch_tool.strip() if isinstance(fetch_tool, str) and fetch_tool.strip() else None,
    )


def effective_search_instances(settings: ToolSettings) -> dict[str, ToolInstance]:
    """The configured and stored search instances, by name.

    ``search_tools`` already holds the stored rows overlaid on the config file's
    entries, a stored row winning, so this reads that one map.
    """
    served = set(any_search.AnySearch.get_supported_providers())
    return {
        name: _instance("search", name, entry, library_backed=str(entry.get("provider") or name) in served)
        for name, entry in settings.search_tools.items()
        if isinstance(entry, dict)
    }


def effective_fetch_instances(settings: ToolSettings) -> dict[str, ToolInstance]:
    """The fetch instances by name, ``builtin_fetch`` always among them.

    A fetch entry named like a search instance is left out: one named like a
    configured search entry stopped startup, so the instance it meets here is a
    stored one, which loads after the config file and is never refused for it.
    """
    instances = {
        BUILTIN_FETCH: ToolInstance(
            name=BUILTIN_FETCH, kind="fetch", provider=BUILTIN_FETCH_PROVIDER, library_backed=True
        ),
    }
    for name, entry in settings.fetch_tools.items():
        if isinstance(entry, dict) and name not in settings.search_tools and name != BUILTIN_FETCH:
            instances[name] = _instance("fetch", name, entry, library_backed=True)
    return instances


def _runtime_or_env(settings: ToolSettings, key: str) -> Any:
    """A runtime-settable value: the runtime or loaded value, else the environment.

    Clearing a runtime value sets the field to ``None``, and the value the
    configuration file set lives on in the environment, where load bridged it.
    """
    value = getattr(settings, key)
    if value is None:
        value = otari_env(key.upper())
    if isinstance(value, str):
        value = value.strip() or None
    return value


def configured_search_default(settings: ToolSettings) -> str | None:
    """``web_search_default_tool`` as set, whether it names an instance or not."""
    value = _runtime_or_env(settings, "web_search_default_tool")
    return str(value) if value is not None else None


def configured_fetch_default(settings: ToolSettings) -> str:
    """``web_fetch_default_tool`` as set, else ``builtin_fetch``."""
    value = _runtime_or_env(settings, "web_fetch_default_tool")
    return str(value) if value is not None else BUILTIN_FETCH


def effective_web_search_max_calls(settings: ToolSettings) -> int:
    """How many search and fetch calls one request may make."""
    value = _runtime_or_env(settings, "web_search_max_calls")
    try:
        calls = int(value) if value is not None else DEFAULT_WEB_SEARCH_MAX_CALLS
    except ValueError:
        return DEFAULT_WEB_SEARCH_MAX_CALLS
    return calls if calls >= 1 else DEFAULT_WEB_SEARCH_MAX_CALLS


def synthesized_search_instance(settings: ToolSettings) -> SynthesizedSearchInstance | None:
    """The search backend the legacy in-loop settings describe, or ``None``.

    In today's order of precedence: the licensed provider with its key, else
    ``web_search_url`` as a SearXNG backend searching ``web_search_engines`` or,
    when those are unset, the four engines the old backend asks for. Both are
    read with the environment as a fallback, as the in-loop backend reads them.
    An empty value counts as unset.
    """
    if settings.web_search_provider and settings.web_search_provider_api_key:
        provider = settings.web_search_provider
        return SynthesizedSearchInstance(
            provider=provider,
            api_key=settings.web_search_provider_api_key,
            provider_defaults=PROVIDER_DEFAULT_OPTIONS.get(provider, _NO_OPTIONS),
        )
    url = (settings.web_search_url or otari_env("WEB_SEARCH_URL") or "").strip()
    if not url:
        return None
    engines_setting = settings.web_search_engines or otari_env("WEB_SEARCH_ENGINES") or ""
    engines = tuple(engine.strip() for engine in engines_setting.split(",") if engine.strip())
    return SynthesizedSearchInstance(
        provider="searxng",
        api_base=url,
        engines=engines or DEFAULT_SEARXNG_ENGINES,
        provider_defaults=PROVIDER_DEFAULT_OPTIONS["searxng"],
    )


def in_loop_default(settings: ToolSettings) -> ToolInstance | SynthesizedSearchInstance | None:
    """The search backend the in-loop tool uses when no credential applies.

    In order: ``web_search_default_tool`` when it names a search instance whose
    provider any-search serves, or ``none``, which turns in-loop search off;
    else the synthesized instance while the legacy settings are present; else
    the one search instance, when there is exactly one and any-search serves its
    provider; else nothing. A default that names no such instance is treated as
    unset. The count in the third step includes instances any-search does not
    serve yet, so an adapter that arrives later never moves the default.
    """
    instances = effective_search_instances(settings)
    named = configured_search_default(settings)
    if named == NO_SEARCH_DEFAULT:
        return None
    if named is not None and (instance := instances.get(named)) is not None and instance.library_backed:
        return instance
    if (synthesized := synthesized_search_instance(settings)) is not None:
        return synthesized
    if len(instances) == 1:
        (only,) = instances.values()
        return only if only.library_backed else None
    return None


def fetch_default(settings: ToolSettings) -> ToolInstance:
    """The fetch instance for the fetch tool and for enrichment.

    ``web_fetch_default_tool`` when it names a fetch instance, else ``builtin_fetch``.
    """
    instances = effective_fetch_instances(settings)
    return instances.get(configured_fetch_default(settings)) or instances[BUILTIN_FETCH]


def enrichment_fetch_instance(settings: ToolSettings, search: ToolInstance) -> ToolInstance:
    """The fetch instance that enriches a search instance's results.

    Its ``fetch_tool`` when that names a fetch instance, else the fetch default.
    """
    if search.fetch_tool is not None and (instance := effective_fetch_instances(settings).get(search.fetch_tool)):
        return instance
    return fetch_default(settings)


def _warn_about_entry(section: str, name: str, problems: list[str]) -> None:
    if problems:
        logger.warning(
            "%s.%s breaks the instance rules and loads anyway: %s. Fix it before a later release refuses it.",
            section,
            name,
            "; ".join(problems),
        )


def warn_about_tool_instances(settings: ToolSettings) -> None:
    """Log what the instance rules let load but an operator should fix.

    Run once at startup, after stored search tools and runtime settings have
    loaded, so a name stored through the dashboard is one the checks can see.
    Names and problems only, never a key, an option's value or a URL.
    """
    search = effective_search_instances(settings)
    fetch = effective_fetch_instances(settings)
    for name, instance in search.items():
        problems = instance_name_problems(name)
        problems += list(option_problems("search", instance.provider, instance.options).values())
        _warn_about_entry("search_tools", name, problems)
        if instance.fetch_tool is not None and instance.fetch_tool not in fetch:
            logger.warning(
                "search_tools.%s.fetch_tool names no fetch instance, so the fetch default enriches its results.",
                name,
            )
    for name in settings.fetch_tools:
        if name in fetch:
            instance = fetch[name]
            problems = list(option_problems("fetch", instance.provider, instance.options).values())
            _warn_about_entry("fetch_tools", name, problems)
        elif name in search:
            logger.error(
                "fetch_tools.%s is left out: a stored search tool has the same name, and names are unique "
                "across search and fetch instances. Rename one of them.",
                name,
            )

    named = configured_search_default(settings)
    if named is not None and named != NO_SEARCH_DEFAULT:
        if named not in search:
            logger.warning("web_search_default_tool names no search instance (%s), so it is treated as unset.", named)
        elif not search[named].library_backed:
            logger.warning(
                "web_search_default_tool names a search instance (%s) whose provider the in-loop tool cannot "
                "use yet, so it is treated as unset.",
                named,
            )
    fetch_named = configured_fetch_default(settings)
    if fetch_named not in fetch:
        logger.warning(
            "web_fetch_default_tool names no fetch instance (%s), so builtin_fetch is the fetch default.",
            fetch_named,
        )
    if named != NO_SEARCH_DEFAULT and len(search) > 1 and in_loop_default(settings) is None:
        logger.warning(
            "There are %d search instances and no web_search_default_tool naming one, so the in-loop "
            "otari_web_search tool has no default to search with. Set web_search_default_tool to one of them, "
            "or to 'none' to keep in-loop search off.",
            len(search),
        )


def validate_default_tool(settings: ToolSettings, key: str, value: object) -> None:
    """Refuse a runtime default that names no instance it may name.

    Checked when the dashboard or the API writes one, and only then: at load and
    at read such a default is treated as unset with a warning instead, since the
    instance it named may have gone since it was set. A blank value clears the
    setting and is always accepted. The name is compared stripped, as every read
    takes it.
    """
    if not isinstance(value, str) or not (value := value.strip()):
        return
    if key == "web_search_default_tool":
        if value == NO_SEARCH_DEFAULT:
            return
        instance = effective_search_instances(settings).get(value)
        if instance is None:
            msg = f"web_search_default_tool must name a search instance, or be 'none'; there is no '{value}'."
            raise ValueError(msg)
        if not instance.library_backed:
            msg = (
                f"web_search_default_tool cannot name '{value}': the in-loop tool cannot search with "
                f"provider '{instance.provider}' yet."
            )
            raise ValueError(msg)
    elif key == "web_fetch_default_tool" and value not in effective_fetch_instances(settings):
        msg = f"web_fetch_default_tool must name a fetch instance, or be '{BUILTIN_FETCH}'; there is no '{value}'."
        raise ValueError(msg)
