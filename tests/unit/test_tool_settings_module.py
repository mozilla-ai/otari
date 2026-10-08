"""The web tools' settings: the instance maps, the defaults, the synthesized instance and the checks."""

import logging
import os
import subprocess
import sys
import textwrap
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import yaml

from gateway.core.config import GatewayConfig, load_config
from gateway.core.settings.tools import (
    BUILTIN_FETCH,
    DEFAULT_SEARXNG_ENGINES,
    DEFAULT_WEB_SEARCH_MAX_CALLS,
    SynthesizedSearchInstance,
    ToolInstance,
    default_api_base,
    effective_fetch_instances,
    effective_search_instances,
    effective_web_search_max_calls,
    enrichment_fetch_instance,
    fetch_default,
    in_loop_default,
    synthesized_search_instance,
    validate_default_tool,
    warn_about_tool_instances,
)
from gateway.services.tool_settings_service import apply_override

_EXA = {"provider": "exa", "api_key": "exa-secret-key"}
_SEARXNG = {"provider": "searxng", "api_base": "http://searxng:8080"}
_RUNTIME_ENV = (
    "OTARI_WEB_SEARCH_URL",
    "OTARI_WEB_SEARCH_ENGINES",
    "OTARI_WEB_SEARCH_DEFAULT_TOOL",
    "OTARI_WEB_FETCH_DEFAULT_TOOL",
    "OTARI_WEB_SEARCH_MAX_CALLS",
)


@pytest.fixture(autouse=True)
def _no_runtime_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The read sites fall back to the environment, so a developer's own must not leak in."""
    for name in _RUNTIME_ENV:
        monkeypatch.delenv(name, raising=False)
    # load_config bridges YAML values into os.environ; undo that after each test.
    with mock.patch.dict(os.environ):
        yield


def _warnings(caplog: pytest.LogCaptureFixture, config: GatewayConfig) -> str:
    caplog.set_level(logging.WARNING, logger="gateway")
    caplog.clear()
    warn_about_tool_instances(config)
    return "\n".join(record.getMessage() for record in caplog.records)


# --------------------------------------------------------------------------- #
# The in-loop default: the named default or 'none', else the legacy settings,
# else the one instance any-search serves, else nothing
# --------------------------------------------------------------------------- #


def test_the_named_default_wins() -> None:
    config = GatewayConfig(
        search_tools={"exa": _EXA, "other": {"provider": "fake"}},
        web_search_default_tool="other",
        web_search_url="http://searxng:8080",
    )
    default = in_loop_default(config)
    assert isinstance(default, ToolInstance)
    assert default.name == "other"


def test_none_turns_in_loop_search_off_whatever_else_is_configured() -> None:
    config = GatewayConfig(
        search_tools={"exa": _EXA},
        web_search_default_tool="none",
        web_search_url="http://searxng:8080",
        web_search_provider="tavily",
        web_search_provider_api_key="tvly-key",
    )
    assert in_loop_default(config) is None


@pytest.mark.parametrize("spelled", ["None", "NONE", " none "])
def test_none_is_read_in_any_case(spelled: str) -> None:
    """YAML reads None as a string, and an operator who writes it means 'none'."""
    config = GatewayConfig(search_tools={"exa": _EXA}, web_search_default_tool=spelled)
    assert in_loop_default(config) is None
    validate_default_tool(config, "web_search_default_tool", spelled)


def test_the_legacy_settings_come_before_a_single_instance() -> None:
    """A search_tools entry never replaces the legacy settings by accident."""
    config = GatewayConfig(search_tools={"exa": _EXA}, web_search_url="http://searxng:8080")
    default = in_loop_default(config)
    assert isinstance(default, SynthesizedSearchInstance)
    assert default.provider == "searxng"


def test_a_single_served_instance_is_the_default() -> None:
    default = in_loop_default(GatewayConfig(search_tools={"exa": _EXA}))
    assert isinstance(default, ToolInstance)
    assert default.name == "exa"


def test_a_single_instance_any_search_does_not_serve_is_no_default() -> None:
    assert in_loop_default(GatewayConfig(search_tools={"local": _SEARXNG})) is None


def test_two_instances_and_no_default_give_nothing() -> None:
    """The count includes an instance any-search does not serve, so a later adapter never moves the default."""
    assert in_loop_default(GatewayConfig(search_tools={"exa": _EXA, "local": _SEARXNG})) is None


def test_nothing_configured_gives_nothing() -> None:
    assert in_loop_default(GatewayConfig()) is None


@pytest.mark.parametrize("named", ["missing", "local"])
def test_a_default_naming_no_served_instance_is_treated_as_unset(named: str) -> None:
    """One that names nothing, or one any-search does not serve, is treated as unset."""
    config = GatewayConfig(search_tools={"local": _SEARXNG}, web_search_default_tool=named)
    assert in_loop_default(config) is None
    config.web_search_url = "http://legacy:8080"
    default = in_loop_default(config)
    assert isinstance(default, SynthesizedSearchInstance)


def test_a_dangling_default_falls_through_to_the_single_instance() -> None:
    config = GatewayConfig(search_tools={"exa": _EXA}, web_search_default_tool="gone")
    default = in_loop_default(config)
    assert isinstance(default, ToolInstance)
    assert default.name == "exa"


def test_a_stored_instance_counts_like_a_configured_one() -> None:
    """Stored rows are overlaid onto search_tools, so the maps see them the same way."""
    config = GatewayConfig()
    config.search_tools = {"stored": {"provider": "exa", "api_key": "k"}}
    default = in_loop_default(config)
    assert isinstance(default, ToolInstance)
    assert default.name == "stored"


# --------------------------------------------------------------------------- #
# The synthesized instance
# --------------------------------------------------------------------------- #


def test_the_provider_pair_describes_the_synthesized_instance() -> None:
    config = GatewayConfig(web_search_provider="tavily", web_search_provider_api_key="tvly-key")
    synthesized = synthesized_search_instance(config)
    assert synthesized == SynthesizedSearchInstance(
        provider="tavily", api_key="tvly-key", provider_defaults={"include_raw_content": True}
    )
    assert synthesized.name == "tavily"
    assert "tvly-key" not in repr(synthesized)


def test_the_provider_wins_over_the_url() -> None:
    config = GatewayConfig(
        web_search_provider="brave", web_search_provider_api_key="brave-key", web_search_url="http://searxng:8080"
    )
    synthesized = synthesized_search_instance(config)
    assert synthesized is not None
    assert synthesized.provider == "brave"


def test_half_a_provider_pair_leaves_the_url() -> None:
    config = GatewayConfig(web_search_provider="brave", web_search_url="http://searxng:8080")
    synthesized = synthesized_search_instance(config)
    assert synthesized is not None
    assert synthesized.provider == "searxng"


def test_the_url_with_engines_describes_a_searxng_instance() -> None:
    config = GatewayConfig(web_search_url="http://searxng:8080", web_search_engines="mojeek, wikipedia,")
    assert synthesized_search_instance(config) == SynthesizedSearchInstance(
        provider="searxng",
        api_base="http://searxng:8080",
        engines=("mojeek", "wikipedia"),
        provider_defaults={"method": "get"},
    )


def test_the_url_without_engines_searches_the_old_backends_four() -> None:
    synthesized = synthesized_search_instance(GatewayConfig(web_search_url="http://searxng:8080"))
    assert synthesized is not None
    assert synthesized.engines == DEFAULT_SEARXNG_ENGINES == ("duckduckgo", "mojeek", "qwant", "wikipedia")


def test_the_url_and_engines_fall_back_to_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    # Built first, so the fields stay unset and only the read's fallback sees the variables,
    # as after a runtime value is cleared.
    config = GatewayConfig()
    monkeypatch.setenv("OTARI_WEB_SEARCH_URL", "http://env-searxng:8080")
    monkeypatch.setenv("OTARI_WEB_SEARCH_ENGINES", "qwant")
    synthesized = synthesized_search_instance(config)
    assert synthesized is not None
    assert (synthesized.api_base, synthesized.engines) == ("http://env-searxng:8080", ("qwant",))


def test_no_legacy_settings_no_synthesized_instance() -> None:
    assert synthesized_search_instance(GatewayConfig(web_search_url="  ")) is None


def test_the_synthesized_instance_is_in_neither_map() -> None:
    config = GatewayConfig(web_search_url="http://searxng:8080")
    assert effective_search_instances(config) == {}
    assert list(effective_fetch_instances(config)) == [BUILTIN_FETCH]


# --------------------------------------------------------------------------- #
# Entries at load: the provider per capability, the key from the metadata
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "search_tools,expected",
    [
        ({"exa": {"api_key": "k"}}, None),
        ({"web": {"provider": "exa", "api_key": "k"}}, None),
        ({"exa": {"api_key": "k", "api_base": "https://exa.internal"}}, None),
        (
            {"exa": {"api_key": "k", "api_base": "http://exa.internal"}},
            "api_base must use https when api_key is set",
        ),
        (
            {"local": {"provider": "searxng", "api_key": "k", "api_base": "http://searxng:8080"}},
            "api_base must use https when api_key is set",
        ),
        ({"web": {"api_key": "k"}}, "is not a supported search provider"),
        ({"exa": {}}, "api_key is required for provider 'exa'"),
        ({"searxng": {"api_base": "http://searxng:8080"}}, None),
        ({"local": {"provider": "searxng", "api_base": "http://searxng:8080"}}, None),
        # A missing backend URL is reported as a startup warning, not raised: see
        # the search_tools_without_backend_url tests below.
        ({"searxng": {}}, None),
        # A keyless provider any-search serves needs no key; the metadata says so.
        ({"test": {"provider": "fake"}}, None),
        # A fetch-only name is not a search provider.
        ({"builtin": {}}, "is not a supported search provider"),
        ({"exa": {"api_key": "k", "options": "nope"}}, "options must be a mapping"),
        ({"exa": {"api_key": "k", "timeout": "soon"}}, "timeout must be a number"),
        ({"exa": {"api_key": "k", "timeout": 0}}, "timeout must be greater than 0"),
        ({"exa": {"api_key": "k", "timeout": -5}}, "timeout must be greater than 0"),
        ({"exa": {"api_key": "k", "timeout": 0.5}}, None),
        ({"ex/a": {"provider": "exa", "api_key": "k"}}, "must not contain '/'"),
        ({"exa": {"api_key": "k", "fetch_tool": ""}}, "fetch_tool must name a fetch instance"),
        ({"exa": {"api_key": "k", "fetch_tool": 3}}, "fetch_tool must name a fetch instance"),
        # Lenient while the legacy settings exist: these load, with a warning.
        ({"exa:main": {"provider": "exa", "api_key": "k"}}, None),
        ({"none": {"provider": "exa", "api_key": "k"}}, None),
        ({"exa": {"api_key": "k", "options": {"type": "neural", "typo": 1}}}, None),
    ],
)
def test_validate_search_tools(search_tools: dict[str, Any], expected: str | None) -> None:
    config = GatewayConfig(search_tools=search_tools)
    if expected is None:
        config.validate_search_tools()
        return
    with pytest.raises(ValueError, match=expected):
        config.validate_search_tools()


def test_validate_accepts_a_searxng_tool_that_inherits_the_web_search_url() -> None:
    """The api_base a searxng tool omits is the one the in-loop tool already uses."""
    config = GatewayConfig(
        search_tools={"local": {"provider": "searxng"}},
        web_search_url="http://searxng:8080",
    )
    config.validate_search_tools()
    assert config.search_tools_without_backend_url() == []


def test_validate_refuses_a_credentialed_searxng_tool_inheriting_http() -> None:
    config = GatewayConfig(
        search_tools={"local": {"provider": "searxng", "api_key": "k"}},
        web_search_url="http://searxng:8080",
    )
    with pytest.raises(ValueError, match="api_base must use https when api_key is set"):
        config.validate_search_tools()


def test_a_searxng_tool_with_no_backend_url_anywhere_is_reported() -> None:
    """Startup warns instead of failing: a dashboard-stored URL lands after load."""
    config = GatewayConfig(search_tools={"local": {"provider": "searxng"}, "exa": {"api_key": "k"}})
    assert config.search_tools_without_backend_url() == ["local"]


def test_a_searxng_tool_with_its_own_api_base_is_not_reported() -> None:
    config = GatewayConfig(search_tools={"local": {"provider": "searxng", "api_base": "http://adapter:9000"}})
    assert config.search_tools_without_backend_url() == []


@pytest.mark.parametrize(
    "fetch_tools,expected",
    [
        ({"exa-fetch": {"provider": "exa", "api_key": "k"}}, None),
        ({"test-fetch": {"provider": "fake"}}, None),
        ({"exa-fetch": {"provider": "exa"}}, "api_key is required for provider 'exa'"),
        ({"mine": {"provider": "builtin"}}, "is not a supported fetch provider"),
        ({"local": {"provider": "searxng"}}, "is not a supported fetch provider"),
        ({"exa-fetch": {"provider": "exa", "api_key": "k", "api_base": "http://exa"}}, "must use https"),
        ({"exa-fetch": {"provider": "exa", "api_key": "k", "timeout": 0}}, "timeout must be greater than 0"),
        # The fetch map is new, so the name rules hold at load from the start.
        ({"exa:fetch": {"provider": "exa", "api_key": "k"}}, "its name contains ':'"),
        ({BUILTIN_FETCH: {"provider": "fake"}}, "its name is reserved"),
        ({"none": {"provider": "fake"}}, "its name is reserved"),
        ({"Builtin_Fetch": {"provider": "fake"}}, "its name is reserved"),
        ({"a/b": {"provider": "fake"}}, "must not contain '/'"),
    ],
)
def test_validate_fetch_tools(fetch_tools: dict[str, Any], expected: str | None) -> None:
    config = GatewayConfig(fetch_tools=fetch_tools)
    if expected is None:
        config.validate_fetch_tools()
        return
    with pytest.raises(ValueError, match=expected):
        config.validate_fetch_tools()


@pytest.mark.parametrize(
    "config,validate",
    [
        (GatewayConfig(search_tools={"mine": {"provider": "nope"}}), GatewayConfig.validate_search_tools),
        (GatewayConfig(fetch_tools={"mine": {"provider": "nope"}}), GatewayConfig.validate_fetch_tools),
    ],
)
def test_an_unknown_provider_is_refused_without_offering_the_test_one(
    config: GatewayConfig, validate: Callable[[GatewayConfig], None]
) -> None:
    with pytest.raises(ValueError, match="is not a supported") as refused:
        validate(config)
    assert "exa" in str(refused.value)
    assert "fake" not in str(refused.value)


def test_a_fetch_entry_named_like_a_configured_search_entry_stops_startup() -> None:
    config = GatewayConfig(search_tools={"exa": _EXA}, fetch_tools={"exa": {"provider": "exa", "api_key": "k"}})
    with pytest.raises(ValueError, match=r"fetch_tools\.exa is refused: search_tools has an instance"):
        config.validate_fetch_tools()


def test_load_config_runs_the_fetch_checks(tmp_path: Path) -> None:
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump({"fetch_tools": {"bad:name": {"provider": "fake"}}}))
    with pytest.raises(ValueError, match="its name contains ':'"):
        load_config(str(path))


def test_a_fetch_entry_named_like_a_stored_search_tool_is_left_out(caplog: pytest.LogCaptureFixture) -> None:
    """Stored rows load after startup, so the fetch entry goes, with an error, rather than the gateway."""
    config = GatewayConfig(fetch_tools={"shared": {"provider": "fake"}})
    config.validate_fetch_tools()
    config.search_tools = {"shared": {"provider": "exa", "api_key": "k"}}
    assert list(effective_fetch_instances(config)) == [BUILTIN_FETCH]
    caplog.set_level(logging.ERROR, logger="gateway")
    warn_about_tool_instances(config)
    assert any(
        record.levelno == logging.ERROR and "fetch_tools.shared is left out" in record.getMessage()
        for record in caplog.records
    )


def test_a_search_entry_named_builtin_fetch_loads_beside_the_implicit_instance(
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = GatewayConfig(search_tools={BUILTIN_FETCH: _EXA})
    config.validate_search_tools()
    assert BUILTIN_FETCH in effective_search_instances(config)
    assert effective_fetch_instances(config)[BUILTIN_FETCH].provider == "builtin"
    assert f"search_tools.{BUILTIN_FETCH} breaks the instance rules" in _warnings(caplog, config)


# --------------------------------------------------------------------------- #
# The instance maps
# --------------------------------------------------------------------------- #


def test_instances_are_flagged_library_backed_or_not() -> None:
    config = GatewayConfig(search_tools={"exa": _EXA, "local": _SEARXNG})
    instances = effective_search_instances(config)
    assert instances["exa"].library_backed is True
    assert instances["local"].library_backed is False
    assert instances["local"].provider_defaults == {"method": "get"}


def test_builtin_fetch_is_always_a_fetch_instance() -> None:
    fetch = effective_fetch_instances(GatewayConfig(fetch_tools={"exa-fetch": {"provider": "exa", "api_key": "k"}}))
    assert list(fetch) == [BUILTIN_FETCH, "exa-fetch"]
    assert fetch[BUILTIN_FETCH] == ToolInstance(
        name=BUILTIN_FETCH, kind="fetch", provider="builtin", library_backed=True
    )
    assert fetch["exa-fetch"].library_backed is True


def test_an_instance_is_hashable_and_cannot_change_the_configuration() -> None:
    config = GatewayConfig(search_tools={"local": {**_SEARXNG, "options": {"categories": "news"}}})
    instance = effective_search_instances(config)["local"]
    assert hash(instance) == hash(effective_search_instances(config)["local"])
    with pytest.raises(TypeError):
        instance.options["categories"] = "it"  # type: ignore[index]
    with pytest.raises(TypeError):
        instance.provider_defaults["method"] = "post"  # type: ignore[index]
    assert config.search_tools["local"]["options"] == {"categories": "news"}


def test_nested_option_values_are_not_shared_with_the_configuration() -> None:
    entry = {**_EXA, "options": {"contents": {"text": True}, "includeDomains": ["a.example"]}}
    config = GatewayConfig(search_tools={"exa": entry})
    instance = effective_search_instances(config)["exa"]
    instance.options["contents"]["text"] = False
    instance.options["includeDomains"].append("b.example")
    assert config.search_tools["exa"]["options"] == {"contents": {"text": True}, "includeDomains": ["a.example"]}


def test_an_instance_keeps_its_key_and_base_out_of_its_repr() -> None:
    entry = {"provider": "exa", "api_key": "exa-secret-key", "api_base": "https://user:pw@exa.internal"}
    text = repr(effective_search_instances(GatewayConfig(search_tools={"exa": entry}))["exa"])
    assert "exa-secret-key" not in text
    assert "user:pw" not in text


# --------------------------------------------------------------------------- #
# Option checks, and the lenient load of names and options that break the rules
# --------------------------------------------------------------------------- #


def test_options_the_schema_refuses_are_recorded_to_be_dropped() -> None:
    entry = {**_EXA, "options": {"type": "neural", "typo": True, "numResults": 3, "category": "news"}}
    instance = effective_search_instances(GatewayConfig(search_tools={"exa": entry}))["exa"]
    assert instance.dropped_options == frozenset({"type", "typo"})
    assert instance.options == entry["options"]


def test_option_types_are_not_checked() -> None:
    """Exa's contents option is an object that also takes false; only names and fixed lists are checked."""
    entry = {**_EXA, "options": {"contents": False, "numResults": "many"}}
    assert effective_search_instances(GatewayConfig(search_tools={"exa": entry}))["exa"].dropped_options == frozenset()


def test_options_are_checked_against_the_fake_providers_metadata() -> None:
    config = GatewayConfig(
        search_tools={"test": {"provider": "fake", "options": {"hits": 2, "account": "a", "nope": 1}}},
        fetch_tools={"test-fetch": {"provider": "fake", "options": {"title": "T", "nope": 1}}},
    )
    assert effective_search_instances(config)["test"].dropped_options == frozenset({"nope"})
    assert effective_fetch_instances(config)["test-fetch"].dropped_options == frozenset({"nope"})


def test_exa_fetch_options_are_checked_against_its_own_metadata() -> None:
    entry = {"provider": "exa", "api_key": "k", "options": {"maxAgeHours": 1, "type": "auto"}}
    instance = effective_fetch_instances(GatewayConfig(fetch_tools={"exa-fetch": entry}))["exa-fetch"]
    assert instance.dropped_options == frozenset({"type"})


def test_searxng_options_go_unchecked(caplog: pytest.LogCaptureFixture) -> None:
    """There is no schema to check them against until any-search has a SearXNG adapter."""
    config = GatewayConfig(search_tools={"local": {**_SEARXNG, "options": {"anything": 1, "categories": "news"}}})
    config.validate_search_tools()
    assert effective_search_instances(config)["local"].dropped_options == frozenset()
    assert _warnings(caplog, config) == ""


def test_the_lenient_load_warns_with_names_and_never_values(caplog: pytest.LogCaptureFixture) -> None:
    config = GatewayConfig(
        search_tools={
            "exa:main": {**_EXA, "options": {"type": "neural-secret-value", "secretkeyname": "hidden-value"}},
            "none": {"provider": "fake"},
        },
        web_search_default_tool="exa:main",
    )
    config.validate_search_tools()
    logged = _warnings(caplog, config)
    assert "search_tools.exa:main breaks the instance rules and loads anyway" in logged
    assert "its name contains ':'" in logged
    assert "option 'type' has a value the provider refuses" in logged
    assert "option 'secretkeyname' is not one the provider knows" in logged
    assert "search_tools.none breaks the instance rules" in logged
    assert "its name is reserved" in logged
    for value in ("neural-secret-value", "hidden-value", "exa-secret-key"):
        assert value not in logged


# --------------------------------------------------------------------------- #
# Fetch for enrichment, and defaults or fetch_tools that name nothing
# --------------------------------------------------------------------------- #


def test_a_search_instance_is_enriched_by_its_own_fetch_tool() -> None:
    config = GatewayConfig(
        search_tools={"exa": {**_EXA, "fetch_tool": "exa-fetch"}, "other": {"provider": "fake"}},
        fetch_tools={"exa-fetch": {"provider": "exa", "api_key": "k"}},
    )
    search = effective_search_instances(config)
    assert enrichment_fetch_instance(config, search["exa"]).name == "exa-fetch"
    assert enrichment_fetch_instance(config, search["other"]).name == BUILTIN_FETCH


def test_without_a_fetch_tool_the_fetch_default_enriches() -> None:
    config = GatewayConfig(
        search_tools={"exa": _EXA},
        fetch_tools={"exa-fetch": {"provider": "exa", "api_key": "k"}},
        web_fetch_default_tool="exa-fetch",
    )
    assert enrichment_fetch_instance(config, effective_search_instances(config)["exa"]).name == "exa-fetch"


def test_a_dangling_fetch_tool_falls_back_to_the_fetch_default(caplog: pytest.LogCaptureFixture) -> None:
    config = GatewayConfig(search_tools={"exa": {**_EXA, "fetch_tool": "gone"}})
    config.validate_search_tools()
    assert enrichment_fetch_instance(config, effective_search_instances(config)["exa"]).name == BUILTIN_FETCH
    assert "search_tools.exa.fetch_tool names no fetch instance" in _warnings(caplog, config)


def test_a_dangling_fetch_default_falls_back_to_builtin_fetch(caplog: pytest.LogCaptureFixture) -> None:
    config = GatewayConfig(web_fetch_default_tool="gone")
    assert fetch_default(config).name == BUILTIN_FETCH
    assert "web_fetch_default_tool names no fetch instance (gone)" in _warnings(caplog, config)


def test_a_dangling_search_default_warns(caplog: pytest.LogCaptureFixture) -> None:
    logged = _warnings(caplog, GatewayConfig(search_tools={"exa": _EXA}, web_search_default_tool="gone"))
    assert "web_search_default_tool names no search instance (gone)" in logged


def test_a_search_default_on_a_provider_without_adapter_warns(caplog: pytest.LogCaptureFixture) -> None:
    logged = _warnings(caplog, GatewayConfig(search_tools={"local": _SEARXNG}, web_search_default_tool="local"))
    assert "cannot use yet" in logged


# --------------------------------------------------------------------------- #
# The warning for several search instances and no default
# --------------------------------------------------------------------------- #


def test_several_instances_and_no_default_warn(caplog: pytest.LogCaptureFixture) -> None:
    logged = _warnings(caplog, GatewayConfig(search_tools={"exa": _EXA, "local": _SEARXNG}))
    assert "There are 2 search instances and no usable web_search_default_tool" in logged


def test_the_warning_names_the_instances_a_default_may_name(caplog: pytest.LogCaptureFixture) -> None:
    logged = _warnings(
        caplog, GatewayConfig(search_tools={"exa": _EXA, "other": {"provider": "fake"}, "local": _SEARXNG})
    )
    assert "Set web_search_default_tool to one of exa, other, or to 'none'" in logged


def test_the_warning_with_no_usable_instance_says_so(caplog: pytest.LogCaptureFixture) -> None:
    config = GatewayConfig(search_tools={"local": _SEARXNG, "other": {**_SEARXNG, "api_base": "http://other:8080"}})
    logged = _warnings(caplog, config)
    assert "None of them is on a provider the in-loop tool can use yet" in logged
    assert "Set web_search_default_tool to one of" not in logged


@pytest.mark.parametrize(
    "overrides",
    [
        {"web_search_default_tool": "exa"},
        {"web_search_default_tool": "none"},
        {"web_search_url": "http://searxng:8080"},
        {"web_search_provider": "tavily", "web_search_provider_api_key": "tvly-key"},
    ],
)
def test_a_default_or_legacy_settings_silence_the_warning(
    overrides: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    config = GatewayConfig(search_tools={"exa": _EXA, "local": _SEARXNG}, **overrides)
    assert "search instances" not in _warnings(caplog, config)


def test_a_runtime_default_counts(caplog: pytest.LogCaptureFixture) -> None:
    """The check runs after the stored tool settings apply, so a default set in the dashboard silences it."""
    config = GatewayConfig(search_tools={"exa": _EXA, "local": _SEARXNG})
    apply_override(config, "web_search_default_tool", "exa")
    assert "search instances" not in _warnings(caplog, config)


# --------------------------------------------------------------------------- #
# The runtime settings: values, write-time checks, clearing
# --------------------------------------------------------------------------- #


def test_the_call_cap_defaults_to_ten() -> None:
    assert effective_web_search_max_calls(GatewayConfig()) == DEFAULT_WEB_SEARCH_MAX_CALLS == 10
    assert effective_web_search_max_calls(GatewayConfig(web_search_max_calls=3)) == 3


@pytest.mark.parametrize(
    "key,value",
    [
        ("web_search_default_tool", "exa"),
        ("web_search_default_tool", "none"),
        ("web_search_default_tool", "stored"),
        ("web_search_default_tool", None),
        ("web_search_default_tool", ""),
        ("web_fetch_default_tool", BUILTIN_FETCH),
        ("web_fetch_default_tool", "exa-fetch"),
        ("web_fetch_default_tool", None),
    ],
)
def test_a_default_naming_an_instance_is_accepted_at_write(key: str, value: str | None) -> None:
    config = GatewayConfig(
        search_tools={"exa": _EXA, "local": _SEARXNG},
        fetch_tools={"exa-fetch": {"provider": "exa", "api_key": "k"}},
    )
    config.search_tools = {**config.search_tools, "stored": {"provider": "fake"}}
    validate_default_tool(config, key, value)


def test_a_default_is_compared_stripped_at_write_as_every_read_takes_it() -> None:
    config = GatewayConfig(search_tools={"exa": _EXA}, fetch_tools={"exa-fetch": {"provider": "exa", "api_key": "k"}})
    validate_default_tool(config, "web_search_default_tool", " exa ")
    validate_default_tool(config, "web_fetch_default_tool", "exa-fetch ")
    apply_override(config, "web_search_default_tool", " exa ")
    default = in_loop_default(config)
    assert isinstance(default, ToolInstance)
    assert default.name == "exa"


@pytest.mark.parametrize(
    "key,value,expected",
    [
        ("web_search_default_tool", "gone", "must name a search instance, or be 'none'"),
        ("web_search_default_tool", "local", "cannot search with provider 'searxng' yet"),
        ("web_search_default_tool", "exa-fetch", "must name a search instance"),
        ("web_search_default_tool", BUILTIN_FETCH, "must name a search instance"),
        ("web_fetch_default_tool", "gone", "must name a fetch instance, or be 'builtin_fetch'"),
        ("web_fetch_default_tool", "exa", "must name a fetch instance"),
        ("web_fetch_default_tool", "none", "must name a fetch instance"),
    ],
)
def test_a_default_naming_no_instance_is_refused_at_write(key: str, value: str, expected: str) -> None:
    config = GatewayConfig(
        search_tools={"exa": _EXA, "local": _SEARXNG},
        fetch_tools={"exa-fetch": {"provider": "exa", "api_key": "k"}},
    )
    with pytest.raises(ValueError, match=expected):
        validate_default_tool(config, key, value)


def _in_loop_default_name(config: GatewayConfig) -> str | None:
    default = in_loop_default(config)
    return default.name if default is not None else None


def _fetch_default_name(config: GatewayConfig) -> str:
    return fetch_default(config).name


@pytest.mark.parametrize(
    "key,configured,runtime,read,built_in",
    [
        # No built-in search default: with two instances and no legacy settings, nothing.
        ("web_search_default_tool", "exa", "other", _in_loop_default_name, None),
        ("web_fetch_default_tool", "exa-fetch", "other-fetch", _fetch_default_name, BUILTIN_FETCH),
        ("web_search_max_calls", 4, 7, effective_web_search_max_calls, DEFAULT_WEB_SEARCH_MAX_CALLS),
    ],
)
def test_clearing_a_runtime_setting_falls_back_to_the_file_then_the_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    configured: Any,
    runtime: Any,
    read: Callable[[GatewayConfig], Any],
    built_in: Any,
) -> None:
    """Clearing falls back to the configuration file's value, as the other bridged tool settings do."""
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "config.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "search_tools": {"exa": _EXA, "other": {"provider": "fake"}},
                "fetch_tools": {
                    "exa-fetch": {"provider": "exa", "api_key": "k"},
                    "other-fetch": {"provider": "fake"},
                },
                key: configured,
            }
        )
    )
    config = load_config(str(path))
    assert read(config) == configured

    apply_override(config, key, runtime)
    assert read(config) == runtime

    apply_override(config, key, None)
    assert read(config) == configured

    # With no file or environment value either, the built-in default.
    os.environ.pop(f"OTARI_{key.upper()}")
    assert read(config) == built_in


# --------------------------------------------------------------------------- #
# The base URL an instance inherits
# --------------------------------------------------------------------------- #


def test_default_api_base(monkeypatch: pytest.MonkeyPatch) -> None:
    config = GatewayConfig(web_search_url=" http://searxng:8080 ")
    assert default_api_base(config, "searxng") == "http://searxng:8080"
    assert default_api_base(config, "exa") == "https://api.exa.ai"
    assert default_api_base(config, "unknown") is None
    # The field alone: only the in-loop reads fall back to the environment.
    unset = GatewayConfig()
    monkeypatch.setenv("OTARI_WEB_SEARCH_URL", "http://env:8080")
    assert default_api_base(unset, "searxng") is None


# --------------------------------------------------------------------------- #
# The libraries' log filters
# --------------------------------------------------------------------------- #

_FILTERS_AFTER_CREATE_APP = textwrap.dedent(
    """
    import logging

    from any_fetch._logging import RedactProviderUrls as FetchFilter
    from any_search._logging import RedactProviderUrls as SearchFilter

    from gateway.core.config import GatewayConfig
    from gateway.main import create_app

    httpx_logger = logging.getLogger("httpx")
    before = [type(existing) for existing in httpx_logger.filters]
    assert SearchFilter not in before and FetchFilter not in before, before
    create_app(GatewayConfig(database_url="sqlite:///:memory:", master_key="sk-test-master"))
    after = [type(existing) for existing in httpx_logger.filters]
    assert SearchFilter in after and FetchFilter in after, after
    print("OK")
    """
)


def test_create_app_installs_both_libraries_log_filters() -> None:
    """In a fresh interpreter: an earlier test may already have installed them for this process."""
    result = subprocess.run(
        [sys.executable, "-c", _FILTERS_AFTER_CREATE_APP],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "OK" in result.stdout
