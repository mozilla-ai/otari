"""What the legacy SearXNG settings become in the breaking release that removes them.

``web_search_url`` and ``web_search_engines`` name the SearXNG instance and
engines the in-loop search uses when no search tool is its default, and a
``searxng`` search tool without an ``api_base`` or ``engines`` of its own
inherits them. Both settings are deprecated. The breaking release that
removes them converts a value stored through the dashboard into a stored
search tool when that is unambiguous; a value in the configuration file or the
environment, which otari cannot rewrite, stops the start until it is removed.
:func:`legacy_search_settings_warnings` says, for each value in effect, which
of the two happens and what an operator changes, through the dashboard for a
stored value, to avoid a refusal. It names settings and tools, never a value:
a URL can carry credentials.
"""

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any

from gateway.core.config import GatewayConfig
from gateway.core.env import otari_env
from gateway.core.settings.tools import (
    DEFAULT_SEARXNG_ENGINES,
    SynthesizedSearchInstance,
    inherits_web_search_engines,
    inherits_web_search_url,
)
from gateway.log_config import logger
from gateway.services.tools._search_tool_overlay import (
    config_file_search_tools,
    stored_row_names,
    stored_search_tool_names,
    stored_tool_names,
)
from gateway.services.tools._web_search import in_loop_default, unnamed_direct_search_tool

# The name the breaking release gives the search tool a stored web_search_url becomes, and its provider.
CONVERTED_TOOL = "searxng"
CONVERTED_PROVIDER = "searxng"

_REMOVAL = "It is deprecated, and the breaking release that removes it"
_FIX_FIRST = (
    "Fix the stored legacy search settings through the dashboard before upgrading to that release: "
    "a start it refuses may already have migrated the database."
)
_OWN_FIELD = {"web_search_url": "api_base", "web_search_engines": "engines option"}


@dataclass(frozen=True)
class LegacySearchValues:
    """``web_search_url`` and ``web_search_engines`` from one source, ``""`` where that source sets none."""

    url: str = ""
    engines: str = ""

    @classmethod
    def of(cls, url: object, engines: object) -> "LegacySearchValues":
        """The two values with surrounding blanks removed, ``None`` read as unset."""
        return cls(url=str(url or "").strip(), engines=str(engines or "").strip())

    @classmethod
    def in_effect(cls, config: GatewayConfig) -> "LegacySearchValues":
        """The values ``config`` holds, with the environment as a fallback, as the in-loop search reads them."""
        return cls.of(
            config.web_search_url or otari_env("WEB_SEARCH_URL"),
            config.web_search_engines or otari_env("WEB_SEARCH_ENGINES"),
        )


def converted_engines(stored: LegacySearchValues) -> tuple[str, ...]:
    """The engines the search tool a stored ``web_search_url`` becomes searches: the stored ones, else the four."""
    return _engine_list(stored.engines) or DEFAULT_SEARXNG_ENGINES


def legacy_search_settings_warnings(
    *,
    from_file: LegacySearchValues,
    stored: LegacySearchValues,
    inherited: LegacySearchValues,
    configured_tools: Mapping[str, Mapping[str, Any]],
    stored_tools: Mapping[str, Mapping[str, Any]],
    other_tool_names: Collection[str],
    url_is_in_loop_default: bool,
    conversion_sets_default: bool,
    moves_unnamed_calls: bool,
    provider_pair_set: bool = False,
) -> list[str]:
    """One warning per legacy SearXNG value in effect and per search tool that inherits one, in that order.

    ``from_file`` holds the configuration file's and the environment's values,
    ``stored`` the dashboard's, which win over them, and ``inherited`` the
    values a ``searxng`` tool without its own takes: the settings' fields once
    the stored values apply, which an empty stored value clears.
    ``configured_tools`` and ``stored_tools`` are the search tools by name, a
    stored one hiding a configured one of its name, and ``other_tool_names``
    the names a fetch tool, or a stored row that did not load, holds.
    ``url_is_in_loop_default`` says the in-loop search runs on
    ``web_search_url``. ``conversion_sets_default`` says no
    ``web_search_default_tool`` is in effect, so the conversion names the new
    tool there; a provider pair, which the release refuses, does not count,
    and ``provider_pair_set`` says one is set.
    ``moves_unnamed_calls`` says the conversion would move the direct search
    endpoint's calls that name no tool off the tool that serves them. Empty
    when nothing legacy is in effect.
    """
    url_source, engines_source = _sources(inherited, stored)
    url_inherited = _file_url_inherited(inherited, stored, {**configured_tools, **stored_tools})
    warnings = file_value_warnings(
        from_file, stored, url_is_in_loop_default=url_is_in_loop_default, url_inherited=url_inherited
    )
    refuses_stored = False

    if stored.url:
        conversion, refused = _stored_url_conversion(
            stored,
            inherited,
            configured_tools,
            stored_tools,
            other_tool_names,
            sets_default=conversion_sets_default,
            moves_unnamed_calls=moves_unnamed_calls,
        )
        if conversion_sets_default and provider_pair_set and not refused:
            conversion += (
                " That release refuses web_search_provider too, so the model's search moves from it to this tool."
            )
        refuses_stored = refused
        warnings.append(f"web_search_url is stored through the dashboard. {conversion}")
    elif stored.engines and not any(_inherits_engines(name, entry) for name, entry in stored_tools.items()):
        warnings.append(
            "web_search_engines is stored through the dashboard, and neither a stored web_search_url nor a stored "
            f"search tool uses it. {_REMOVAL} deletes the stored value."
        )

    warnings += configured_tool_warnings(configured_tools, inherited, stored)
    for name, entry in sorted(stored_tools.items()):
        for setting, source in _inherited(name, entry, url_source, engines_source):
            if source == "stored":
                warnings.append(
                    f"The stored search tool {name} inherits {setting}, which is also stored. The breaking release "
                    f"that removes {setting} copies the value into the tool's own {_OWN_FIELD[setting]}."
                )
            else:
                warnings.append(
                    f"The stored search tool {name} inherits {setting} from the configuration file or the "
                    f"environment. The breaking release that removes {setting} refuses to start while it does: "
                    f"give the tool its own {_OWN_FIELD[setting]} through the dashboard."
                )
                refuses_stored = True

    if refuses_stored:
        warnings.append(_FIX_FIRST)
    return warnings


def configured_tool_warnings(
    configured_tools: Mapping[str, Mapping[str, Any]], inherited: LegacySearchValues, stored: LegacySearchValues
) -> list[str]:
    """The refusals of the configuration file's searxng tools that inherit a legacy value, which otari cannot copy."""
    url_source, engines_source = _sources(inherited, stored)
    return [
        f"search_tools.{name} in the configuration file inherits {setting}. The breaking release that removes "
        f"{setting} refuses to start while it does: give search_tools.{name} its own {_OWN_FIELD[setting]}."
        for name, entry in sorted(configured_tools.items())
        for setting, _source in _inherited(name, entry, url_source, engines_source)
    ]


def file_value_warnings(
    from_file: LegacySearchValues,
    stored: LegacySearchValues,
    *,
    url_is_in_loop_default: bool | None,
    url_inherited: bool,
) -> list[str]:
    """The refusals of the values in the configuration file or the environment, which need no stored tool.

    ``url_is_in_loop_default`` is ``None`` when it cannot be told, and
    ``url_inherited`` says a searxng search tool takes its address from the
    file's ``web_search_url``.
    """
    warnings: list[str] = []
    if from_file.url:
        warning = (
            "web_search_url is set in the configuration file or the environment (OTARI_WEB_SEARCH_URL). "
            f"{_REMOVAL} refuses to start while it is set. "
        )
        if url_is_in_loop_default is None:
            warning += (
                "Replace it with a search tool with provider searxng, its own api_base, an engines option and a "
                "price, named in web_search_default_tool when the model searches on it, then unset web_search_url"
            )
        elif url_is_in_loop_default and not stored.url:
            warning += (
                "Add a search tool with provider searxng, its own api_base, an engines option and a price, name it "
                "in web_search_default_tool, then unset web_search_url"
            )
        elif url_inherited:
            warning += "Give each searxng search tool that inherits it its own api_base, then unset it"
        else:
            warning += "The model's search does not run on it, so unset it"
        warnings.append(
            warning + " (in docker-compose, set OTARI_WEB_SEARCH_URL= in .env, which turns compose's default off)."
        )
    if from_file.engines:
        warning = (
            "web_search_engines is set in the configuration file or the environment (OTARI_WEB_SEARCH_ENGINES). "
            f"{_REMOVAL} refuses to start while it is set. Move the engines into the engines option of a "
            "searxng search tool, then unset web_search_engines."
        )
        if stored.url and not stored.engines:
            warning += (
                " The stored web_search_url converts with the engines "
                f"{', '.join(DEFAULT_SEARXNG_ENGINES)}: store the engines it should keep through the dashboard."
            )
        warnings.append(warning)
    return warnings


def _stored_url_conversion(
    stored: LegacySearchValues,
    inherited: LegacySearchValues,
    configured_tools: Mapping[str, Mapping[str, Any]],
    stored_tools: Mapping[str, Mapping[str, Any]],
    other_tool_names: Collection[str],
    *,
    sets_default: bool,
    moves_unnamed_calls: bool,
) -> tuple[str, bool]:
    """What the breaking release does with a stored web_search_url, and whether it refuses to start over it."""
    engines = converted_engines(stored)
    holder = stored_tools.get(CONVERTED_TOOL) or configured_tools.get(CONVERTED_TOOL)
    # Only a stored tool gets the stored values copied in; a configured one inheriting them is refused.
    holder_inherits = stored if CONVERTED_TOOL in stored_tools else LegacySearchValues()
    way_out = "rename that tool, or replace web_search_url with a searxng search tool through the dashboard."
    if CONVERTED_TOOL in other_tool_names:
        return (
            f"{_REMOVAL} would convert it into a search tool named {CONVERTED_TOOL}, and a fetch tool, or a stored "
            f"tool otari could not read, has that name, so it refuses to start: {way_out}",
            True,
        )
    if holder is not None and not _same_values(CONVERTED_TOOL, holder, holder_inherits, stored.url, engines):
        return (
            f"{_REMOVAL} would convert it into a search tool named {CONVERTED_TOOL}, and a search tool of that name "
            f"with another address or engines exists, so it refuses to start: {way_out}",
            True,
        )
    if moves_unnamed_calls:
        default = " and name it in web_search_default_tool" if sets_default else ""
        way_out = (
            "replace web_search_url with a searxng search tool through the dashboard, and set "
            "web_search_default_tool to the tool those calls should use, which the model's search then uses too."
            if sets_default
            else "the model's search does not run on web_search_url, so clear it through the dashboard."
        )
        return (
            f"{_REMOVAL} would convert it into a search tool named {CONVERTED_TOOL}{default}, which moves the "
            f"search API's calls that name no tool off the tool that serves them, so it refuses to start: {way_out}",
            True,
        )
    if holder is not None:
        default = ", names it in web_search_default_tool" if sets_default else ""
        return (
            f"{_REMOVAL} keeps the search tool named {CONVERTED_TOOL}, which has the same address and engines"
            f"{default}, and deletes the stored value.",
            False,
        )
    engine_source = (
        "the stored web_search_engines" if _engine_list(stored.engines) else "the engines " + ", ".join(engines)
    )
    default = f", and web_search_default_tool set to {CONVERTED_TOOL}" if sets_default else ""
    return (
        f"{_REMOVAL} converts it into a stored search tool named {CONVERTED_TOOL} on SearXNG, "
        f"with {engine_source}{default}.",
        False,
    )


def _same_values(
    name: str, entry: Mapping[str, Any], inherited: LegacySearchValues, url: str, engines: tuple[str, ...]
) -> bool:
    """Whether a search tool, with what it inherits, has the address and engines the conversion would give it."""
    options = entry.get("options")
    own = options if isinstance(options, Mapping) else {}
    api_base = inherited.url if inherits_web_search_url(name, entry) else str(entry.get("api_base") or "").strip()
    tool_engines = inherited.engines if _inherits_engines(name, entry) else own.get("engines")
    return (
        str(entry.get("provider") or name) == CONVERTED_PROVIDER
        and api_base.rstrip("/") == url.rstrip("/")
        and _engine_set(tool_engines) == _engine_set(",".join(engines))
    )


def _file_url_inherited(
    inherited: LegacySearchValues, stored: LegacySearchValues, tools: Mapping[str, Mapping[str, Any]]
) -> bool:
    """Whether a searxng search tool takes its address from a ``web_search_url`` that is not stored."""
    return bool(inherited.url) and not stored.url and any(inherits_web_search_url(n, e) for n, e in tools.items())


def _engine_list(value: object) -> tuple[str, ...]:
    return tuple(name.strip() for name in str(value or "").split(",") if name.strip())


def _engine_set(value: object) -> frozenset[str]:
    """The engines a comma-separated value names, in any order."""
    return frozenset(_engine_list(value))


def _inherits_engines(name: str, entry: Mapping[str, Any]) -> bool:
    options = entry.get("options")
    return inherits_web_search_engines(
        str(entry.get("provider") or name), options if isinstance(options, Mapping) else {}
    )


def _sources(inherited: LegacySearchValues, stored: LegacySearchValues) -> tuple[str, str]:
    """Where the URL and the engines a searxng tool inherits come from: "stored", "file", or "" for none."""
    url_source = "stored" if stored.url else "file" if inherited.url else ""
    engines_source = "stored" if stored.engines else "file" if inherited.engines else ""
    return url_source, engines_source


def _inherited(name: str, entry: Mapping[str, Any], url_source: str, engines_source: str) -> list[tuple[str, str]]:
    """Each legacy setting the entry inherits, with the source of the value it takes."""
    inherited: list[tuple[str, str]] = []
    if url_source and inherits_web_search_url(name, entry):
        inherited.append(("web_search_url", url_source))
    if engines_source and _inherits_engines(name, entry):
        inherited.append(("web_search_engines", engines_source))
    return inherited


def warn_about_legacy_search_settings(
    config: GatewayConfig,
    *,
    from_file: LegacySearchValues,
    stored_settings: Mapping[str, object],
    stored_tools_loaded: bool,
) -> None:
    """Log what the breaking release does with the legacy SearXNG settings in effect, once stored tools have loaded.

    ``from_file`` is read before the dashboard's stored settings apply, and
    ``stored_settings`` are those settings. Whether the conversion moves the
    direct endpoint's unnamed calls is found by picking their tool on
    ``config`` and on a copy with the conversion applied. When the stored
    tools did not load, only the configuration file's and the environment's
    values are accounted for, since the rest depends on those tools.
    """
    stored = LegacySearchValues.of(stored_settings.get("web_search_url"), stored_settings.get("web_search_engines"))
    inherited = LegacySearchValues.of(config.web_search_url, config.web_search_engines)
    if not any((from_file.url, from_file.engines, stored.url, stored.engines, inherited.url, inherited.engines)):
        return
    if not stored_tools_loaded:
        # The in-loop default may name a stored tool, and a stored tool may shadow a configured one,
        # so without them neither can be told.
        configured = config_file_search_tools(config)
        warnings = file_value_warnings(
            from_file, stored, url_is_in_loop_default=None, url_inherited=False
        ) + configured_tool_warnings(configured, inherited, stored)
        if stored.url or stored.engines:
            warnings.append(
                "The stored search tools did not load, so this start does not say what the breaking release that "
                "removes web_search_url and web_search_engines does with their stored values."
            )
    else:
        stored_names = stored_search_tool_names()
        without_pair = config.model_copy(update={"web_search_provider": None, "web_search_provider_api_key": None})
        conversion_sets_default = bool(stored.url) and _searches_on_url(without_pair)
        warnings = legacy_search_settings_warnings(
            from_file=from_file,
            stored=stored,
            inherited=inherited,
            configured_tools={
                name: entry for name, entry in config_file_search_tools(config).items() if name not in stored_names
            },
            stored_tools={name: entry for name, entry in config.search_tools.items() if name in stored_names},
            other_tool_names=frozenset(config.fetch_tools) | (stored_row_names() - stored_tool_names()),
            url_is_in_loop_default=_searches_on_url(config),
            conversion_sets_default=conversion_sets_default,
            moves_unnamed_calls=bool(stored.url)
            and _moves_unnamed_calls(config, stored, conversion_sets_default, stored_names),
            provider_pair_set=config.web_search_provider_configured(),
        )
    for warning in warnings:
        logger.warning("%s", warning)


def _moves_unnamed_calls(
    config: GatewayConfig, stored: LegacySearchValues, sets_default: bool, stored_names: Collection[str]
) -> bool:
    """Whether converting the stored web_search_url changes the tool a direct search naming none runs on.

    A search that had no tool to run on before is not moved: the converted
    tool serves it after.
    """
    before = unnamed_direct_search_tool(config)
    if before is None:
        return False
    converted = {
        "provider": CONVERTED_PROVIDER,
        "api_base": stored.url,
        "options": {"engines": ",".join(converted_engines(stored))},
    }
    # The stored tools get the stored values they inherit copied in, as the conversion does.
    tools = {
        name: _with_stored_copies(name, entry, stored) if name in stored_names else entry
        for name, entry in config.search_tools.items()
    }
    update: dict[str, Any] = {
        "search_tools": {**tools, CONVERTED_TOOL: converted},
        "web_search_url": None,
        "web_search_engines": None,
    }
    if sets_default:
        update["web_search_default_tool"] = CONVERTED_TOOL
    return unnamed_direct_search_tool(config.model_copy(update=update)) != before


def _searches_on_url(config: GatewayConfig) -> bool:
    """Whether the in-loop search runs on ``web_search_url``, the backend the legacy settings describe."""
    in_loop = in_loop_default(config)
    return isinstance(in_loop, SynthesizedSearchInstance) and in_loop.provider == CONVERTED_PROVIDER


def _with_stored_copies(name: str, entry: Mapping[str, Any], stored: LegacySearchValues) -> dict[str, Any]:
    """A stored tool's entry with the stored values it inherits copied into its own fields."""
    copied = dict(entry)
    if stored.url and inherits_web_search_url(name, entry):
        copied["api_base"] = stored.url
    if stored.engines and _inherits_engines(name, entry):
        options = entry.get("options")
        copied["options"] = {**(options if isinstance(options, Mapping) else {}), "engines": stored.engines}
    return copied
