"""The resolver: which provider, key and options a web search or fetch runs with (design G).

Table-driven over both orders, with the libraries' ``fake`` and ``exa`` metadata.
"""

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

import pytest

from gateway.core.config import GatewayConfig
from gateway.core.settings import tools as settings_tools
from gateway.exceptions.tools_exceptions import (
    SearchToolMisconfiguredError,
    SearchToolNotFoundError,
    WebSearchCredentialRefusedError,
    WebSearchOptionRefusedError,
)
from gateway.models.tools import (
    ResolvedWebSearchConfig,
    WebFetchCredential,
    WebSearchCredential,
    WebToolCredentialSource,
)
from gateway.services.pricing_service import gateway_tool_pricing_key
from gateway.services.tools import ResolvedFetch, ResolvedSearch, resolve_fetch, resolve_search
from gateway.services.tools._web_search import _resolve

_ORG = WebToolCredentialSource.ORG
_HYBRID = WebToolCredentialSource.HYBRID
_EXA = {"provider": "exa", "api_key": "exa-instance-key"}
_FAKE = {"provider": "fake"}
_SEARXNG = {"provider": "searxng", "api_base": "http://searxng:8080"}
# The settings read OTARI_ variables, so a developer's own must not leak in.
pytestmark = pytest.mark.usefixtures("no_otari_env")


@pytest.fixture(autouse=True)
def _warnings_due_again() -> Iterator[None]:
    """The resolver warns once per problem, so each test starts with none warned yet."""
    _resolve._warn_once.cache_clear()
    yield
    _resolve._warn_once.cache_clear()


def _policy(
    *,
    credential: WebSearchCredential | None = None,
    fetch_credential: WebFetchCredential | None = None,
    provider_options: dict[str, Any] | None = None,
) -> ResolvedWebSearchConfig:
    return ResolvedWebSearchConfig(
        enabled=True,
        max_results=None,
        purpose_hint=None,
        allowed_domains=None,
        blocked_domains=None,
        provider_options=provider_options,
        authorized_tools=None,
        credential=credential,
        fetch_credential=fetch_credential,
    )


def _warnings(caplog: pytest.LogCaptureFixture) -> str:
    return "\n".join(record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING)


# --------------------------------------------------------------------------- #
# Search, in order: the direct endpoint's pick, the policy's credential, the
# in-loop default, nothing
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _SearchCase:
    id: str
    config: dict[str, Any]
    expected: dict[str, Any] | None
    policy: ResolvedWebSearchConfig | None = None
    requested: str | None = None
    direct_credential: WebSearchCredential | None = None
    # Only the fields a case is about are compared; the options, when given, in full.
    options: dict[str, Any] | None = None


_SEARCH_CASES = [
    # The in-loop default, design E, rule 6.
    _SearchCase(
        "the named default",
        {"search_tools": {"exa": _EXA, "other": _FAKE}, "web_search_default_tool": "exa"},
        {
            "name": "exa",
            "provider": "exa",
            "runner": "library",
            "api_key": "exa-instance-key",
            "api_base": "https://api.exa.ai",
            "label": "exa:exa",
            "rate_keys": ("exa:exa", "otari:web_search"),
            "timeout": 15.0,
        },
    ),
    _SearchCase(
        "the named default over the legacy settings",
        {"search_tools": {"test": _FAKE}, "web_search_default_tool": "test", "web_search_url": "http://searxng:8080"},
        {"name": "test", "provider": "fake", "api_base": "", "label": "fake:test"},
    ),
    _SearchCase(
        "'none' turns in-loop search off", {"search_tools": {"test": _FAKE}, "web_search_default_tool": "none"}, None
    ),
    _SearchCase(
        "a default naming no instance is unset",
        {"search_tools": {"test": _FAKE}, "web_search_default_tool": "gone"},
        {"name": "test"},
    ),
    _SearchCase(
        "the legacy provider pair",
        {
            "search_tools": {"test": _FAKE},
            "web_search_provider": "tavily",
            "web_search_provider_api_key": "tvly-deploy",
        },
        {
            "name": "tavily",
            "provider": "tavily",
            "runner": "old",
            "api_key": "tvly-deploy",
            "api_base": "",
            "label": "otari:web_search",
            "rate_keys": ("otari:web_search",),
            "timeout": 15.0,
        },
        options={},
    ),
    _SearchCase(
        "the legacy URL, with the four engines",
        {"web_search_url": "http://searxng:8080"},
        {"name": "searxng", "provider": "searxng", "runner": "old", "api_base": "http://searxng:8080"},
        options={"engines": "duckduckgo,mojeek,qwant,wikipedia"},
    ),
    _SearchCase(
        "the legacy URL with its engines",
        {"web_search_url": "http://searxng:8080", "web_search_engines": "brave, wikipedia"},
        {"provider": "searxng"},
        options={"engines": "brave,wikipedia"},
    ),
    _SearchCase("the one instance", {"search_tools": {"test": _FAKE}}, {"name": "test", "runner": "library"}),
    _SearchCase("the one instance, any-search not serving it yet", {"search_tools": {"local": _SEARXNG}}, None),
    _SearchCase("several instances and no default", {"search_tools": {"exa": _EXA, "test": _FAKE}}, None),
    _SearchCase("nothing configured", {}, None),
    # The policy's credential wins over the in-loop default.
    _SearchCase(
        "the organization's key",
        {"search_tools": {"test": _FAKE}},
        {
            "name": None,
            "provider": "exa",
            "runner": "library",
            "api_key": "exa-org-key",
            "api_base": "https://api.exa.ai",
            "label": "org:exa",
            "rate_keys": ("otari:web_search",),
            "timeout": 15.0,
        },
        policy=_policy(credential=WebSearchCredential(provider="exa", source=_ORG, api_key="exa-org-key")),
    ),
    _SearchCase(
        "the control plane's credential, which settles nothing here",
        {},
        {"provider": "exa", "api_key": "exa-peer-key", "label": None, "rate_keys": ()},
        policy=_policy(credential=WebSearchCredential(provider="exa", source=_HYBRID, api_key="exa-peer-key")),
    ),
    _SearchCase(
        "a credential's own endpoint",
        {},
        {"api_base": "https://exa.example"},
        policy=_policy(
            credential=WebSearchCredential(
                provider="exa", source=_ORG, api_key="exa-org-key", api_base="https://exa.example"
            )
        ),
    ),
    _SearchCase(
        "a credential serves with in-loop search off",
        {"search_tools": {"test": _FAKE}, "web_search_default_tool": "none"},
        {"provider": "fake", "label": "org:fake"},
        policy=_policy(credential=WebSearchCredential(provider="fake", source=_ORG)),
    ),
    _SearchCase(
        "a Tavily key on the old backend",
        {"search_tools": {"test": _FAKE}},
        {
            "provider": "tavily",
            "runner": "old",
            "api_key": "tvly-org",
            "api_base": "",
            # Design L: the old backend keeps recording the call under otari:web_search, with no label.
            "label": None,
            "rate_keys": ("otari:web_search",),
        },
        policy=_policy(credential=WebSearchCredential(provider="tavily", source=_ORG, api_key="tvly-org")),
        options={},
    ),
    _SearchCase(
        "a SearXNG key on its own endpoint, never the deployment's",
        {"web_search_url": "http://deployment-searxng:8080", "web_search_engines": "brave"},
        {"provider": "searxng", "runner": "old", "api_key": "", "api_base": "https://org-searxng.example"},
        policy=_policy(
            credential=WebSearchCredential(provider="searxng", source=_ORG, api_base="https://org-searxng.example")
        ),
        options={},
    ),
    # The direct endpoint's pick, named or not.
    _SearchCase(
        "a named call on its instance's own key",
        {"search_tools": {"exa": _EXA}},
        {
            "name": "exa",
            "provider": "exa",
            "api_key": "exa-instance-key",
            "label": "exa:exa",
            "rate_keys": ("exa:exa",),
            "timeout": 30.0,
        },
        policy=_policy(credential=WebSearchCredential(provider="fake", source=_ORG)),
        requested="exa",
    ),
    _SearchCase(
        "a named call's own timeout",
        {"search_tools": {"exa": {**_EXA, "timeout": 7}}},
        {"timeout": 7.0},
        requested="exa",
    ),
    _SearchCase(
        "an unnamed call on the organization's key",
        {"search_tools": {"exa": {**_EXA, "timeout": 7, "options": {"type": "fast"}}}},
        {
            "name": "exa",
            "provider": "fake",
            "runner": "library",
            "api_key": "",
            "api_base": "",
            "label": "org:fake",
            "rate_keys": ("otari:web_search",),
            "timeout": 7.0,
        },
        requested="exa",
        direct_credential=WebSearchCredential(provider="fake", source=_ORG, options={"cost": 0.5}),
        options={"cost": 0.5},
    ),
    _SearchCase(
        "an unnamed call on a Tavily key",
        {"search_tools": {"exa": _EXA}},
        {
            "name": "exa",
            "provider": "tavily",
            "runner": "old",
            "api_key": "tvly-org",
            "timeout": 30.0,
            "label": "org:tavily",
            "rate_keys": ("otari:web_search",),
        },
        requested="exa",
        direct_credential=WebSearchCredential(provider="tavily", source=_ORG, api_key="tvly-org"),
    ),
    _SearchCase(
        "a SearXNG instance inherits the deployment's URL and engines",
        {
            "search_tools": {"local": {"provider": "searxng"}},
            "web_search_url": "http://searxng:8080",
            "web_search_engines": "brave",
        },
        {
            "name": "local",
            "provider": "searxng",
            "runner": "old",
            "api_base": "http://searxng:8080",
            "label": "searxng:local",
        },
        requested="local",
        options={"engines": "brave"},
    ),
    _SearchCase(
        "a SearXNG instance's own engines",
        {"search_tools": {"local": {**_SEARXNG, "options": {"engines": "qwant"}}}, "web_search_engines": "brave"},
        {"api_base": "http://searxng:8080"},
        requested="local",
        options={"engines": "qwant"},
    ),
    _SearchCase(
        "a SearXNG instance opts into POST over otari's default",
        {"search_tools": {"local": {**_SEARXNG, "options": {"method": "post"}}}},
        {"runner": "old"},
        requested="local",
        options={"method": "post"},
    ),
]


@pytest.mark.parametrize("case", _SEARCH_CASES, ids=lambda case: case.id)
def test_the_search_order(case: _SearchCase) -> None:
    resolved = resolve_search(
        GatewayConfig(**case.config),
        case.policy,
        requested=case.requested,
        request_options=None,
        direct_credential=case.direct_credential,
    )

    if case.expected is None:
        assert resolved is None
        return
    assert resolved is not None
    assert {key: getattr(resolved, key) for key in case.expected} == case.expected
    if case.options is not None:
        assert dict(resolved.options) == case.options


def test_the_fallback_rate_keys_are_the_pricing_services_keys_for_the_tools() -> None:
    search = resolve_search(GatewayConfig(search_tools={"test": _FAKE}), None, requested=None, request_options=None)
    fetch = resolve_fetch(GatewayConfig(), None, requested=None, for_search=None)

    assert search is not None
    assert search.rate_keys[-1] == gateway_tool_pricing_key("web_search")
    assert fetch.rate_keys[-1] == gateway_tool_pricing_key("web_fetch")


def test_a_name_no_search_instance_has_is_not_found() -> None:
    with pytest.raises(SearchToolNotFoundError):
        resolve_search(GatewayConfig(search_tools={"exa": _EXA}), None, requested="gone", request_options=None)


def test_a_searxng_instance_with_no_address_to_search_is_misconfigured() -> None:
    config = GatewayConfig(search_tools={"local": {"provider": "searxng"}})

    with pytest.raises(SearchToolMisconfiguredError) as refused:
        resolve_search(config, None, requested="local", request_options=None)
    assert refused.value.status_code == 400
    assert refused.value.message == "Search tool 'local' is not configured correctly."


def test_a_key_never_goes_to_an_inherited_plain_http_endpoint() -> None:
    config = GatewayConfig(
        search_tools={"local": {"provider": "searxng", "api_key": "searxng-key"}}, web_search_url="http://plain:8080"
    )

    with pytest.raises(SearchToolMisconfiguredError) as refused:
        resolve_search(config, None, requested="local", request_options=None)
    assert "https" in refused.value.message
    assert "searxng-key" not in refused.value.message


def test_a_base_url_is_passed_without_blanks_or_a_trailing_slash() -> None:
    config = GatewayConfig(search_tools={"local": {"provider": "searxng", "api_base": " http://searxng:8080/ "}})

    resolved = resolve_search(config, None, requested="local", request_options=None)

    assert resolved is not None
    assert resolved.api_base == "http://searxng:8080"


def _named(config: GatewayConfig, name: str) -> ResolvedSearch:
    resolved = resolve_search(config, None, requested=name, request_options=None)
    assert resolved is not None
    return resolved


def test_a_library_instance_takes_its_providers_endpoint_and_its_name_as_its_provider() -> None:
    resolved = _named(GatewayConfig(search_tools={"exa": {"api_key": "k"}}), "exa")
    assert (resolved.provider, resolved.api_base, resolved.runner) == ("exa", "https://api.exa.ai", "library")


@pytest.mark.parametrize(
    "entry",
    [
        {"provider": "exa", "api_key": "secret", "api_base": "http://exa.internal"},
        {"provider": "searxng", "api_key": "secret", "api_base": "http://searxng:8080"},
    ],
)
def test_a_key_never_goes_to_an_instances_own_plain_http_endpoint(entry: dict[str, Any]) -> None:
    with pytest.raises(SearchToolMisconfiguredError, match="api_base must use https when api_key is set"):
        _named(GatewayConfig(search_tools={"search": entry}), "search")


def test_a_searxng_instance_needs_no_key() -> None:
    resolved = _named(
        GatewayConfig(search_tools={"local": {"provider": "searxng", "api_base": "http://searxng:8080/"}}), "local"
    )
    assert (resolved.runner, resolved.api_key, resolved.api_base) == ("old", "", "http://searxng:8080")


@pytest.mark.parametrize(
    ("entry", "api_base"),
    [
        pytest.param({"provider": "searxng"}, "http://searxng:8080", id="inherits-web-search-url"),
        pytest.param(
            {"provider": "searxng", "api_base": "http://adapter:9000"}, "http://adapter:9000", id="its-own-wins"
        ),
    ],
)
def test_a_searxng_instances_endpoint(entry: dict[str, Any], api_base: str) -> None:
    """Contract C8: the in-loop backend's URL is the default, so one config line is enough."""
    config = GatewayConfig(search_tools={"local": entry}, web_search_url="http://searxng:8080/")
    assert _named(config, "local").api_base == api_base


@pytest.mark.parametrize(
    ("entry", "engines"),
    [
        pytest.param({"provider": "searxng"}, "duckduckgo,mojeek", id="inherits-web-search-engines"),
        pytest.param({"provider": "searxng", "options": {"engines": "wikipedia"}}, "wikipedia", id="its-own-wins"),
    ],
)
def test_a_searxng_instances_engines(entry: dict[str, Any], engines: str) -> None:
    """Contract C8: both doors query the same engines on the same backend."""
    config = GatewayConfig(
        search_tools={"local": entry}, web_search_url="http://searxng:8080", web_search_engines="duckduckgo,mojeek"
    )
    assert _named(config, "local").options["engines"] == engines


def test_the_deployments_engines_never_reach_another_provider() -> None:
    config = GatewayConfig(search_tools={"exa": {"api_key": "k"}}, web_search_engines="duckduckgo,mojeek")
    assert "engines" not in _named(config, "exa").options


def test_a_result_says_whether_a_credential_or_an_instance_serves_it() -> None:
    config = GatewayConfig(search_tools={"fake": _FAKE})
    key = WebSearchCredential(provider="fake", source=_ORG, api_key="org-key")

    on_key = resolve_search(config, None, requested="fake", request_options=None, direct_credential=key)
    on_instance = _named(config, "fake")

    assert on_key is not None and on_key.on_credential
    assert not on_instance.on_credential


def test_a_credential_for_a_provider_this_gateway_cannot_call_is_ignored(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="gateway")
    credential = WebSearchCredential(provider="unheard-of", source=_HYBRID, api_key="key-of-the-peer")

    resolved = resolve_search(
        GatewayConfig(search_tools={"test": _FAKE}),
        _policy(credential=credential),
        requested=None,
        request_options=None,
    )

    assert resolved is not None
    assert resolved.name == "test"
    assert "unheard-of" in _warnings(caplog)
    assert "key-of-the-peer" not in _warnings(caplog)


@pytest.mark.parametrize(
    "credential",
    [
        WebSearchCredential(provider="exa", source=_ORG),
        WebSearchCredential(provider="exa", source=_HYBRID, api_key=""),
        WebSearchCredential(provider="tavily", source=_ORG),
        WebSearchCredential(provider="searxng", source=_HYBRID),
        WebSearchCredential(provider="searxng", source=_HYBRID, api_base=" / "),
        WebSearchCredential(provider="tavily", source=_HYBRID, api_key="k", api_base="https://tavily-proxy.example"),
    ],
    ids=[
        "exa-without-key",
        "exa-with-an-empty-key",
        "tavily-without-key",
        "searxng-without-endpoint",
        "searxng-with-a-blank-endpoint",
        "tavily-with-an-endpoint",
    ],
)
def test_a_credential_its_provider_cannot_be_called_with_is_refused(
    credential: WebSearchCredential, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never handed on: the server's own key or endpoint would serve the tenant, or the key would go elsewhere."""
    monkeypatch.setenv("EXA_API_KEY", "the-servers-own-key")
    config = GatewayConfig(search_tools={"exa": _EXA}, web_search_url="http://searxng:8080")

    with pytest.raises(WebSearchCredentialRefusedError) as refused:
        resolve_search(config, _policy(credential=credential), requested=None, request_options=None)
    with pytest.raises(WebSearchCredentialRefusedError):
        resolve_search(config, None, requested="exa", request_options=None, direct_credential=credential)
    assert credential.provider in refused.value.message


def test_a_credentials_source_counts_by_value() -> None:
    """One built from data, its source a plain string, is still the control plane's."""
    credential = WebSearchCredential(provider="exa", source=cast(WebToolCredentialSource, "hybrid"), api_key="k")

    resolved = resolve_search(GatewayConfig(), _policy(credential=credential), requested=None, request_options=None)

    assert resolved is not None
    assert (resolved.label, resolved.rate_keys) == (None, ())


def test_a_resolved_search_never_prints_its_key_endpoint_or_options() -> None:
    config = GatewayConfig(
        search_tools={
            "test": {
                "provider": "fake",
                "api_key": "the-instance-key",
                "api_base": "https://user:pass@search.internal",
                "options": {"error": "secret-option"},
            }
        }
    )
    resolved = resolve_search(config, None, requested=None, request_options=None)

    assert resolved is not None
    for printed in (repr(resolved), str(resolved)):
        assert "the-instance-key" not in printed
        assert "search.internal" not in printed
        assert "secret-option" not in printed


# --------------------------------------------------------------------------- #
# Option layering
# --------------------------------------------------------------------------- #


def _in_loop(
    config: GatewayConfig,
    *,
    credential: WebSearchCredential | None = None,
    workspace: dict[str, Any] | None = None,
    request: dict[str, Any] | None = None,
) -> ResolvedSearch:
    resolved = resolve_search(
        config,
        _policy(credential=credential, provider_options=workspace),
        requested=None,
        request_options=request,
    )
    assert resolved is not None
    return resolved


def test_layers_apply_lowest_first() -> None:
    config = GatewayConfig(
        search_tools={"test": {"provider": "fake", "options": {"cost": 1, "delay": 0.1, "error": "a"}}}
    )

    resolved = _in_loop(config, workspace={"delay": 0.2, "error": "b"}, request={"error": "c"})

    assert dict(resolved.options) == {"cost": 1, "delay": 0.2, "error": "c"}


def test_otaris_defaults_sit_below_every_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Design E, rule 5. No provider a library serves has defaults yet, so the test gives ``fake`` some."""
    defaults = MappingProxyType({"fake": MappingProxyType({"cost": 1, "delay": 0.1, "error": "default"})})
    monkeypatch.setattr(settings_tools, "PROVIDER_DEFAULT_OPTIONS", defaults)
    monkeypatch.setattr(_resolve, "PROVIDER_DEFAULT_OPTIONS", defaults)
    config = GatewayConfig(search_tools={"test": {"provider": "fake", "options": {"delay": 0.2}}})
    credential = WebSearchCredential(provider="fake", source=_ORG, options={"delay": 0.3})

    assert dict(_in_loop(config, request={"error": "request"}).options) == {"cost": 1, "delay": 0.2, "error": "request"}
    assert dict(_in_loop(config, credential=credential).options) == {"cost": 1, "delay": 0.3, "error": "default"}


def test_the_old_code_takes_the_layers_without_otaris_defaults() -> None:
    credential = WebSearchCredential(provider="tavily", source=_ORG, api_key="k", options={"topic": "news"})

    resolved = _in_loop(GatewayConfig(), credential=credential, workspace={"topic": "finance", "search_depth": "basic"})

    assert dict(resolved.options) == {"topic": "finance", "search_depth": "basic"}


def test_the_count_is_taken_out_of_the_options_uncapped() -> None:
    config = GatewayConfig(search_tools={"test": {"provider": "fake", "options": {"max_results": 3}}})

    assert _in_loop(config).max_results == 3
    assert _in_loop(config, request={"max_results": 2, "time_range": "week"}).max_results == 2
    assert _in_loop(config, request={"max_results": 100}).max_results == 100
    resolved = _in_loop(config, request={"time_range": "week"})
    assert dict(resolved.options) == {"time_range": "week"}


def test_a_workspace_count_the_library_refuses_leaves_the_instances_count(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="gateway")
    config = GatewayConfig(search_tools={"test": {"provider": "fake", "options": {"max_results": 3}}})

    resolved = _in_loop(config, workspace={"max_results": "10"})

    assert resolved.max_results == 3
    assert "'max_results'" in _warnings(caplog)


@pytest.mark.parametrize("count", [0, True, "5", 2.5])
def test_on_the_old_code_a_count_that_is_not_a_positive_whole_number_sets_no_count(count: object) -> None:
    config = GatewayConfig(web_search_provider="brave", web_search_provider_api_key="brv")

    resolved = _in_loop(config, request={"max_results": count})

    assert resolved.max_results is None
    assert "max_results" not in resolved.options


@pytest.mark.parametrize(
    ("option", "problem"),
    [
        ({"nope": 1}, "is not one the provider knows"),
        ({"max_chars": 1}, "is not one the provider knows"),
        ({"account": "a"}, "is set only by the deployment"),
        ({"time_range": "decade"}, "has a value the provider refuses"),
        ({"max_results": 0}, "has a value the provider refuses"),
        ({"max_results": "5"}, "has a value the provider refuses"),
    ],
    ids=["unknown", "fetchs-shared-parameter", "operator-only", "time-range", "count-zero", "count-text"],
)
def test_a_request_option_the_provider_does_not_take_from_a_request_is_refused(
    option: dict[str, Any], problem: str
) -> None:
    config = GatewayConfig(search_tools={"test": _FAKE})
    (key,) = option

    for requested in (None, "test"):
        with pytest.raises(WebSearchOptionRefusedError) as refused:
            resolve_search(config, None, requested=requested, request_options=option)
        assert refused.value.status_code == 400
        assert refused.value.key == key
        assert f"option '{key}' {problem}" in refused.value.message


def test_a_dropped_option_is_warned_about_once(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="gateway")
    config = GatewayConfig(search_tools={"test": _FAKE})

    for _ in range(3):
        _in_loop(config, workspace={"nope": 1})

    assert _warnings(caplog).count("'nope'") == 1


def test_the_workspace_layer_drops_what_it_may_not_set_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="gateway")

    resolved = _in_loop(
        GatewayConfig(search_tools={"test": _FAKE}),
        workspace={"nope": "secret-value", "account": "secret-account", "time_range": "decade", "cost": 2},
    )

    assert dict(resolved.options) == {"cost": 2}
    warnings = _warnings(caplog)
    assert "'nope'" in warnings
    assert "'account'" in warnings
    assert "'time_range'" in warnings
    assert "decade" not in warnings
    assert "workspace" in warnings
    assert "secret-value" not in warnings
    assert "secret-account" not in warnings


def test_a_credentials_options_the_provider_refuses_are_dropped_and_its_operator_only_ones_kept(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="gateway")
    credential = WebSearchCredential(
        provider="fake",
        source=_HYBRID,
        options={"nope": "secret-value", "max_results": 0, "account": "the-orgs-account"},
    )

    resolved = _in_loop(GatewayConfig(), credential=credential)

    assert dict(resolved.options) == {"account": "the-orgs-account"}
    assert "'nope'" in _warnings(caplog)
    assert "'max_results'" in _warnings(caplog)
    assert "control plane" in _warnings(caplog)
    assert "secret-value" not in _warnings(caplog)


def test_an_instances_options_that_broke_the_rules_at_load_are_dropped(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="gateway")
    options = {"type": "neural", "typo": "secret-value", "numResults": 3}
    config = GatewayConfig(search_tools={"exa": {**_EXA, "options": options}})

    resolved = _in_loop(config)

    assert dict(resolved.options) == {"numResults": 3}
    assert "'type'" in _warnings(caplog)
    assert "'typo'" in _warnings(caplog)
    assert "search instance 'exa'" in _warnings(caplog)
    assert "secret-value" not in _warnings(caplog)


def test_an_instance_sets_an_operator_only_option() -> None:
    config = GatewayConfig(search_tools={"test": {"provider": "fake", "options": {"account": "a"}}})

    assert dict(_in_loop(config).options) == {"account": "a"}


def test_the_direct_endpoint_leaves_the_workspace_layer_out() -> None:
    resolved = resolve_search(
        GatewayConfig(search_tools={"test": _FAKE}),
        _policy(provider_options={"cost": 2}),
        requested="test",
        request_options={"delay": 0.1},
    )

    assert resolved is not None
    assert dict(resolved.options) == {"delay": 0.1}


def test_what_runs_on_the_old_backend_takes_its_options_unchecked() -> None:
    config = GatewayConfig(web_search_provider="brave", web_search_provider_api_key="brv")

    resolved = _in_loop(config, workspace={"anything": 1}, request={"whatever": 2, "max_chars": 3})

    assert resolved.runner == "old"
    assert dict(resolved.options) == {"anything": 1, "whatever": 2, "max_chars": 3}


# --------------------------------------------------------------------------- #
# Fetch, in order: a named instance, the search instance's fetch_tool, the
# policy's fetch credential, the fetch default, builtin_fetch
# --------------------------------------------------------------------------- #

_EXA_FETCH = {"provider": "exa", "api_key": "exa-fetch-key"}
_ORG_FETCH = WebFetchCredential(provider="fake", source=_ORG, options={"title": "T"})


@dataclass(frozen=True)
class _FetchCase:
    id: str
    config: dict[str, Any]
    expected: dict[str, Any]
    policy: ResolvedWebSearchConfig | None = None
    requested: str | None = None
    # The search instance whose results are enriched; None for the fetch tool.
    enriching: str | None = None


_FETCH_CASES = [
    _FetchCase(
        "builtin_fetch",
        {},
        {
            "name": "builtin_fetch",
            "provider": "builtin",
            "runner": "library",
            "api_key": "",
            "api_base": "",
            "label": "builtin:builtin_fetch",
            "rate_keys": ("builtin:builtin_fetch", "otari:web_fetch"),
            # The built-in fetcher's own deadline.
            "timeout": 5.0,
        },
    ),
    _FetchCase(
        "the fetch default",
        {"fetch_tools": {"exa-fetch": _EXA_FETCH}, "web_fetch_default_tool": "exa-fetch"},
        {
            "name": "exa-fetch",
            "provider": "exa",
            "api_key": "exa-fetch-key",
            "api_base": "https://api.exa.ai",
            "label": "exa:exa-fetch",
            "rate_keys": ("exa:exa-fetch", "otari:web_fetch"),
            "timeout": 15.0,
        },
    ),
    _FetchCase(
        "a fetch default naming no instance falls back to builtin_fetch",
        {"web_fetch_default_tool": "gone"},
        {"name": "builtin_fetch"},
    ),
    _FetchCase(
        "the organization's fetch key over the fetch default",
        {"fetch_tools": {"exa-fetch": _EXA_FETCH}, "web_fetch_default_tool": "exa-fetch"},
        {"name": None, "provider": "fake", "label": "org:fake", "rate_keys": ("otari:web_fetch",), "timeout": 15.0},
        policy=_policy(fetch_credential=_ORG_FETCH),
    ),
    _FetchCase(
        "the control plane's fetch key",
        {},
        {"provider": "exa", "api_key": "exa-peer", "api_base": "https://api.exa.ai", "label": None, "rate_keys": ()},
        policy=_policy(fetch_credential=WebFetchCredential(provider="exa", source=_HYBRID, api_key="exa-peer")),
    ),
    _FetchCase(
        "a search instance's fetch_tool over the fetch key, for enrichment",
        {"search_tools": {"test": {**_FAKE, "fetch_tool": "exa-fetch"}}, "fetch_tools": {"exa-fetch": _EXA_FETCH}},
        {"name": "exa-fetch", "label": "exa:exa-fetch"},
        policy=_policy(fetch_credential=_ORG_FETCH),
        enriching="test",
    ),
    _FetchCase(
        "the fetch tool ignores a search instance's fetch_tool",
        {"search_tools": {"test": {**_FAKE, "fetch_tool": "exa-fetch"}}, "fetch_tools": {"exa-fetch": _EXA_FETCH}},
        {"name": "builtin_fetch"},
    ),
    _FetchCase(
        "a fetch_tool naming no instance is unset",
        {"search_tools": {"test": {**_FAKE, "fetch_tool": "gone"}}},
        {"name": None, "label": "org:fake"},
        policy=_policy(fetch_credential=_ORG_FETCH),
        enriching="test",
    ),
    _FetchCase(
        "a named fetch instance on its own key",
        {"fetch_tools": {"exa-fetch": {**_EXA_FETCH, "timeout": 4}}},
        {"name": "exa-fetch", "api_key": "exa-fetch-key", "timeout": 4.0},
        policy=_policy(fetch_credential=_ORG_FETCH),
        requested="exa-fetch",
    ),
    _FetchCase(
        "a fetch key on builtin is ignored",
        {},
        {"name": "builtin_fetch"},
        policy=_policy(fetch_credential=WebFetchCredential(provider="builtin", source=_HYBRID)),
    ),
]


@pytest.mark.parametrize("case", _FETCH_CASES, ids=lambda case: case.id)
def test_the_fetch_order(case: _FetchCase) -> None:
    config = GatewayConfig(**case.config)
    for_search = None
    if case.enriching is not None:
        for_search = resolve_search(config, None, requested=case.enriching, request_options=None)

    resolved = resolve_fetch(config, case.policy, requested=case.requested, for_search=for_search)

    assert isinstance(resolved, ResolvedFetch)
    assert {key: getattr(resolved, key) for key in case.expected} == case.expected


def test_a_name_no_fetch_instance_has_is_not_found() -> None:
    with pytest.raises(SearchToolNotFoundError):
        resolve_fetch(GatewayConfig(), None, requested="gone", for_search=None)


def test_a_fetch_instance_any_fetch_no_longer_serves_names_nothing(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="gateway")
    config = GatewayConfig(fetch_tools={"gone-fetch": {"provider": "retired"}}, web_fetch_default_tool="gone-fetch")

    assert resolve_fetch(config, None, requested=None, for_search=None).name == "builtin_fetch"
    assert "'gone-fetch'" in _warnings(caplog)
    with pytest.raises(SearchToolNotFoundError):
        resolve_fetch(config, None, requested="gone-fetch", for_search=None)


def test_a_fetch_key_without_the_key_its_provider_needs_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXA_API_KEY", "the-servers-own-key")
    credential = WebFetchCredential(provider="exa", source=_ORG)

    with pytest.raises(WebSearchCredentialRefusedError):
        resolve_fetch(GatewayConfig(), _policy(fetch_credential=credential), requested=None, for_search=None)


def test_fetch_options_layer_the_instances_and_take_out_the_count(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="gateway")
    options = {"title": "T", "max_chars": 500, "max_results": 3, "account": "a"}
    config = GatewayConfig(fetch_tools={"test-fetch": {"provider": "fake", "options": options}})

    resolved = resolve_fetch(config, None, requested="test-fetch", for_search=None)

    assert resolved.max_chars == 500
    assert dict(resolved.options) == {"title": "T", "account": "a"}
    assert "'max_results'" in _warnings(caplog)


def test_fetch_takes_no_workspace_layer() -> None:
    policy = _policy(provider_options={"title": "from the workspace"}, fetch_credential=_ORG_FETCH)

    resolved = resolve_fetch(GatewayConfig(), policy, requested=None, for_search=None)

    assert dict(resolved.options) == {"title": "T"}
