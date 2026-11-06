"""The account the legacy SearXNG settings get at startup: what the breaking release converts or refuses."""

import logging
from collections.abc import Iterator
from typing import Any

import pytest

from gateway.core.config import GatewayConfig
from gateway.services.tools import LegacySearchValues, reset_search_tool_cache, warn_about_legacy_search_settings
from gateway.services.tools._legacy_search_settings import _moves_unnamed_calls, legacy_search_settings_warnings

_URL = "https://user:secret@searxng.example"
_NONE = LegacySearchValues()
_SEARXNG = {"provider": "searxng", "api_base": "http://own:8080", "options": {"engines": "wikinews"}}
_EXA = {"provider": "exa", "api_key": "sk-exa"}
pytestmark = pytest.mark.usefixtures("no_otari_env")


@pytest.fixture(autouse=True)
def _no_stored_tools() -> Iterator[None]:
    reset_search_tool_cache()
    yield
    reset_search_tool_cache()


def _warnings(
    *,
    from_file: LegacySearchValues = _NONE,
    stored: LegacySearchValues = _NONE,
    inherited: LegacySearchValues | None = None,
    configured: dict[str, dict[str, Any]] | None = None,
    stored_tools: dict[str, dict[str, Any]] | None = None,
    other_tool_names: frozenset[str] = frozenset(),
    url_is_in_loop_default: bool = False,
    conversion_sets_default: bool = False,
    moves_unnamed_calls: bool = False,
) -> list[str]:
    """The warnings, with tools inheriting the stored values over the file's unless ``inherited`` says otherwise."""
    if inherited is None:
        inherited = LegacySearchValues(url=stored.url or from_file.url, engines=stored.engines or from_file.engines)
    return legacy_search_settings_warnings(
        from_file=from_file,
        stored=stored,
        inherited=inherited,
        configured_tools=configured or {},
        stored_tools=stored_tools or {},
        other_tool_names=other_tool_names,
        url_is_in_loop_default=url_is_in_loop_default,
        conversion_sets_default=conversion_sets_default,
        moves_unnamed_calls=moves_unnamed_calls,
    )


def test_nothing_legacy_warns_of_nothing() -> None:
    assert _warnings(configured={"local": _SEARXNG}) == []


@pytest.mark.parametrize("stored", [_NONE, LegacySearchValues(url=_URL + "/stored")], ids=["alone", "beside-stored"])
def test_a_url_in_the_file_or_the_environment_is_refused_even_beside_a_stored_one(stored: LegacySearchValues) -> None:
    warning = _warnings(from_file=LegacySearchValues(url=_URL), stored=stored)[0]
    assert "configuration file or the environment (OTARI_WEB_SEARCH_URL)" in warning
    assert "refuses to start while it is set" in warning
    assert "set OTARI_WEB_SEARCH_URL= in .env" in warning


@pytest.mark.parametrize("stored_engines", ["", "arxiv"], ids=["none-stored", "stored-too"])
def test_engines_in_the_file_or_the_environment_are_refused_even_beside_stored_ones(stored_engines: str) -> None:
    warning = _warnings(
        from_file=LegacySearchValues(engines="wikinews"), stored=LegacySearchValues(engines=stored_engines)
    )[0]
    assert warning.startswith("web_search_engines is set in the configuration file")
    assert "engines option of a searxng search tool" in warning


def test_engines_from_the_file_beside_a_stored_url_say_which_engines_it_converts_with() -> None:
    engines_refused, conversion = _warnings(
        stored=LegacySearchValues(url=_URL), from_file=LegacySearchValues(engines="wikinews")
    )
    assert "refuses to start while it is set" in engines_refused
    assert "store the engines it should keep through the dashboard" in engines_refused
    assert "converts it into a stored search tool" in conversion


def test_a_stored_url_with_no_default_in_effect_converts_with_the_four_engines_and_becomes_it() -> None:
    (warning,) = _warnings(stored=LegacySearchValues(url=_URL), conversion_sets_default=True)
    assert "converts it into a stored search tool named searxng" in warning
    assert "duckduckgo, mojeek, qwant, wikipedia" in warning
    assert "web_search_default_tool set to searxng" in warning


def test_a_stored_url_keeps_its_stored_engines_and_leaves_a_default_in_effect() -> None:
    (warning,) = _warnings(stored=LegacySearchValues(url=_URL, engines="wikinews"))
    assert "with the stored web_search_engines." in warning
    assert "web_search_default_tool set" not in warning


@pytest.mark.parametrize(
    ("tools", "other_tool_names", "moves", "why"),
    [
        pytest.param({"searxng": _SEARXNG}, frozenset(), False, "another address or engines", id="name-taken"),
        pytest.param({}, frozenset({"searxng"}), False, "a fetch tool, or a stored tool", id="name-taken-otherwise"),
        pytest.param({"exa": _EXA}, frozenset(), True, "off the tool that serves them", id="moves-unnamed-calls"),
    ],
)
def test_a_stored_url_that_cannot_convert_is_refused(
    tools: dict[str, dict[str, Any]], other_tool_names: frozenset[str], moves: bool, why: str
) -> None:
    conversion, fix_first = _warnings(
        stored=LegacySearchValues(url=_URL),
        configured=tools,
        other_tool_names=other_tool_names,
        moves_unnamed_calls=moves,
    )
    assert "refuses to start" in conversion and why in conversion
    assert fix_first.startswith("Fix the stored legacy search settings through the dashboard")


def test_a_searxng_tool_with_the_converted_values_is_kept() -> None:
    same = {"provider": "searxng", "api_base": _URL + "/", "options": {"engines": "duckduckgo, mojeek,qwant,wikipedia"}}
    (warning,) = _warnings(stored=LegacySearchValues(url=_URL), stored_tools={"searxng": same})
    assert "keeps the search tool named searxng, which has the same address and engines" in warning


def test_stored_engines_nothing_stored_uses_are_deleted() -> None:
    (warning,) = _warnings(stored=LegacySearchValues(engines="wikinews"))
    assert "neither a stored web_search_url nor a stored search tool uses it" in warning
    assert warning.endswith("deletes the stored value.")


def test_a_configured_tool_inheriting_from_the_file_or_the_dashboard_is_refused() -> None:
    inheriting = {"local": {"provider": "searxng"}}

    (from_file,) = _warnings(
        from_file=LegacySearchValues(url=_URL), configured=inheriting, url_is_in_loop_default=True
    )[1:]
    assert "search_tools.local in the configuration file inherits web_search_url" in from_file
    assert "its own api_base" in from_file

    warnings = _warnings(stored=LegacySearchValues(url=_URL), configured=inheriting)
    assert any("search_tools.local in the configuration file inherits web_search_url" in w for w in warnings)


def test_a_stored_tool_inheriting_a_stored_value_gets_it_copied_and_one_from_the_file_is_refused() -> None:
    stored_tool = {"local": {"provider": "searxng", "api_base": "http://own:8080"}}

    copied = _warnings(stored=LegacySearchValues(engines="wikinews"), stored_tools=stored_tool)
    assert copied == [
        "The stored search tool local inherits web_search_engines, which is also stored. The breaking release "
        "that removes web_search_engines copies the value into the tool's own engines option."
    ]

    refused = _warnings(from_file=LegacySearchValues(engines="wikinews"), stored_tools=stored_tool)
    assert "from the configuration file or the environment" in refused[1]
    assert refused[-1].startswith("Fix the stored legacy search settings")


def test_an_engines_option_present_but_empty_inherits_nothing() -> None:
    own_empty = {"local": {"provider": "searxng", "api_base": "http://own:8080", "options": {"engines": ""}}}
    assert _warnings(from_file=LegacySearchValues(engines="wikinews"), configured=own_empty)[1:] == []


def test_an_entry_named_searxng_with_no_provider_counts_as_one() -> None:
    warnings = _warnings(from_file=LegacySearchValues(url=_URL), configured={"searxng": {}})
    assert any("search_tools.searxng in the configuration file inherits web_search_url" in w for w in warnings)


def test_no_warning_carries_a_value() -> None:
    warnings = _warnings(
        from_file=LegacySearchValues(url=_URL, engines="secret-engine"),
        stored=LegacySearchValues(url=_URL + "/stored"),
        configured={"local": {"provider": "searxng"}},
        stored_tools={"kept": {"provider": "searxng"}},
    )
    assert warnings
    assert not any("secret" in warning for warning in warnings)


def _logged_for(caplog: pytest.LogCaptureFixture, config: GatewayConfig) -> str:
    """The startup account for ``config``, whose web_search_url is the stored one."""
    caplog.set_level(logging.WARNING, logger="gateway")
    warn_about_legacy_search_settings(
        config, from_file=_NONE, stored_settings={"web_search_url": config.web_search_url}, stored_tools_loaded=True
    )
    return "\n".join(record.getMessage() for record in caplog.records)


@pytest.mark.parametrize(
    "default",
    [pytest.param("none", id="none"), pytest.param("ghost", id="names-no-tool")],
)
def test_converting_beside_the_one_tool_unnamed_calls_use_is_refused(
    caplog: pytest.LogCaptureFixture, default: str
) -> None:
    config = GatewayConfig(web_search_url=_URL, web_search_default_tool=default, search_tools={"exa": dict(_EXA)})
    assert "off the tool that serves them, so it refuses to start" in _logged_for(caplog, config)


def test_a_stored_url_with_no_tool_beside_it_converts_and_becomes_the_default(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logged = _logged_for(caplog, GatewayConfig(web_search_url=_URL))
    assert "converts it into a stored search tool named searxng" in logged
    assert "web_search_default_tool set to searxng" in logged


def test_a_stored_url_beside_a_default_tool_converts_and_leaves_the_default(caplog: pytest.LogCaptureFixture) -> None:
    config = GatewayConfig(
        web_search_url=_URL, web_search_default_tool="exa", search_tools={"exa": dict(_EXA), "local": dict(_SEARXNG)}
    )
    logged = _logged_for(caplog, config)
    assert "converts it into a stored search tool named searxng" in logged
    assert "web_search_default_tool set" not in logged


def test_startup_logs_the_account_from_the_file_and_the_stored_settings(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="gateway")
    config = GatewayConfig(web_search_url=_URL)

    warn_about_legacy_search_settings(
        config,
        from_file=LegacySearchValues.in_effect(config),
        stored_settings={"web_search_engines": "wikinews"},
        stored_tools_loaded=True,
    )

    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "web_search_url is set in the configuration file or the environment" in logged
    assert "web_search_engines is stored through the dashboard, and neither a stored web_search_url" in logged
    assert "secret" not in logged


def test_a_searxng_tool_inheriting_the_stored_values_is_kept() -> None:
    stored = LegacySearchValues(url=_URL, engines="wikinews")
    warnings = _warnings(stored=stored, stored_tools={"searxng": {"provider": "searxng"}})
    assert "keeps the search tool named searxng, which has the same address and engines" in warnings[0]
    assert all("refuses" not in warning for warning in warnings)


def test_the_converted_engines_in_another_order_are_the_same() -> None:
    same = {"provider": "searxng", "api_base": _URL, "options": {"engines": "wikipedia,qwant,mojeek,duckduckgo"}}
    (warning,) = _warnings(stored=LegacySearchValues(url=_URL), configured={"searxng": same})
    assert "keeps the search tool named searxng" in warning


@pytest.mark.parametrize(
    ("stored", "in_loop", "advice"),
    [
        pytest.param(_NONE, True, "name it in web_search_default_tool", id="the-loop-searches-it"),
        pytest.param(_NONE, False, "The model's search does not run on it, so unset it", id="another-default"),
        pytest.param(LegacySearchValues(url=_URL), True, "does not run on it", id="a-stored-url-hides-it"),
    ],
)
def test_a_url_in_the_file_is_replaced_only_when_the_model_searches_on_it(
    stored: LegacySearchValues, in_loop: bool, advice: str
) -> None:
    warning = _warnings(from_file=LegacySearchValues(url=_URL), stored=stored, url_is_in_loop_default=in_loop)[0]
    assert advice in warning


def test_a_configured_tool_inheriting_a_stored_value_is_fixed_in_the_file() -> None:
    warnings = _warnings(
        stored=LegacySearchValues(engines="wikinews"),
        configured={"local": {"provider": "searxng", "api_base": "http://own"}},
    )
    assert any("search_tools.local in the configuration file inherits web_search_engines" in w for w in warnings)
    assert not any(w.startswith("Fix the stored legacy search settings") for w in warnings)


def test_a_value_cleared_on_the_dashboard_is_not_inherited() -> None:
    warnings = _warnings(
        from_file=LegacySearchValues(url=_URL), inherited=_NONE, configured={"local": {"provider": "searxng"}}
    )
    assert len(warnings) == 1 and warnings[0].startswith("web_search_url is set in the configuration file")


def test_without_the_stored_tools_only_the_file_values_are_accounted_for(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="gateway")
    config = GatewayConfig(web_search_url=_URL)

    warn_about_legacy_search_settings(
        config,
        from_file=LegacySearchValues(url=_URL),
        stored_settings={"web_search_url": _URL + "/stored"},
        stored_tools_loaded=False,
    )

    logged = [record.getMessage() for record in caplog.records]
    assert logged[0].startswith("web_search_url is set in the configuration file")
    assert "named in web_search_default_tool when the model searches on it" in logged[0]
    assert logged[1].startswith("The stored search tools did not load")
    assert len(logged) == 2


def test_a_provider_pair_does_not_keep_the_converted_tool_from_becoming_the_default(
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = GatewayConfig(web_search_url=_URL, web_search_provider="tavily", web_search_provider_api_key="tvly-key")
    logged = _logged_for(caplog, config)
    assert "converts it into a stored search tool named searxng" in logged
    assert "web_search_default_tool set to searxng" in logged
    assert "refuses web_search_provider too, so the model's search moves from it to this tool" in logged


def test_a_url_in_the_file_a_searxng_tool_inherits_is_given_to_that_tool_first() -> None:
    (warning, inherits) = _warnings(
        from_file=LegacySearchValues(url=_URL),
        configured={"local": {"provider": "searxng", "options": {"engines": "a"}}},
    )
    assert "Give each searxng search tool that inherits it its own api_base, then unset it" in warning
    assert "search_tools.local in the configuration file inherits web_search_url" in inherits


def test_a_configured_searxng_tool_inheriting_the_stored_url_is_not_kept() -> None:
    four = {"provider": "searxng", "options": {"engines": "duckduckgo,mojeek,qwant,wikipedia"}}
    warnings = _warnings(stored=LegacySearchValues(url=_URL), configured={"searxng": four})
    assert "another address or engines exists, so it refuses to start" in warnings[0]
    assert "search_tools.searxng in the configuration file inherits web_search_url" in warnings[1]


def test_the_keep_message_without_a_default_reads_as_two_actions() -> None:
    same = {"provider": "searxng", "api_base": _URL, "options": {"engines": "duckduckgo,mojeek,qwant,wikipedia"}}
    (warning,) = _warnings(stored=LegacySearchValues(url=_URL), stored_tools={"searxng": same})
    assert warning.endswith("which has the same address and engines, and deletes the stored value.")


def test_stored_engines_naming_no_engine_convert_with_the_four() -> None:
    (warning,) = _warnings(stored=LegacySearchValues(url=_URL, engines=" , "))
    assert "with the engines duckduckgo, mojeek, qwant, wikipedia" in warning


def test_nothing_legacy_anywhere_logs_nothing_even_when_the_stored_tools_did_not_load(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="gateway")
    warn_about_legacy_search_settings(GatewayConfig(), from_file=_NONE, stored_settings={}, stored_tools_loaded=False)
    assert caplog.records == []


def test_engines_spelled_in_another_case_are_other_engines() -> None:
    other = {"provider": "searxng", "api_base": _URL, "options": {"engines": "Wikipedia,qwant,mojeek,duckduckgo"}}
    conversion, _fix_first = _warnings(stored=LegacySearchValues(url=_URL), stored_tools={"searxng": other})
    assert "another address or engines exists, so it refuses to start" in conversion


def test_a_stored_searxng_tool_is_compared_with_only_the_values_the_release_copies() -> None:
    warnings = _warnings(
        stored=LegacySearchValues(url=_URL),
        inherited=LegacySearchValues(url=_URL, engines="duckduckgo,mojeek,qwant,wikipedia"),
        stored_tools={"searxng": {"provider": "searxng"}},
    )
    assert "another address or engines exists, so it refuses to start" in warnings[0]


def test_unnamed_calls_moved_by_a_url_the_model_does_not_search_on_say_to_clear_it() -> None:
    conversion, _fix_first = _warnings(stored=LegacySearchValues(url=_URL), moves_unnamed_calls=True)
    assert conversion.endswith("the model's search does not run on web_search_url, so clear it through the dashboard.")


def test_moving_unnamed_calls_to_a_chosen_default_says_the_model_follows() -> None:
    conversion, _fix_first = _warnings(
        stored=LegacySearchValues(url=_URL), conversion_sets_default=True, moves_unnamed_calls=True
    )
    assert "which the model's search then uses too" in conversion


def test_without_the_stored_tools_a_configured_tool_inheriting_is_still_named(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="gateway")
    config = GatewayConfig(web_search_url=_URL, search_tools={"local": {"provider": "searxng"}})

    warn_about_legacy_search_settings(
        config, from_file=LegacySearchValues(url=_URL), stored_settings={}, stored_tools_loaded=False
    )

    logged = [record.getMessage() for record in caplog.records]
    assert any("search_tools.local in the configuration file inherits web_search_url" in line for line in logged)


def test_converting_beside_a_tool_that_cannot_run_moves_nothing(caplog: pytest.LogCaptureFixture) -> None:
    config = GatewayConfig(
        web_search_url=_URL, web_search_default_tool="none", search_tools={"exa": {"provider": "exa"}}
    )
    logged = _logged_for(caplog, config)
    assert "converts it into a stored search tool named searxng" in logged
    assert "refuses to start" not in logged


def test_a_stored_default_inheriting_the_stored_url_keeps_its_unnamed_calls() -> None:
    config = GatewayConfig(
        web_search_url=_URL, web_search_default_tool="local", search_tools={"local": {"provider": "searxng"}}
    )
    assert not _moves_unnamed_calls(config, LegacySearchValues(url=_URL), False, {"local"})
