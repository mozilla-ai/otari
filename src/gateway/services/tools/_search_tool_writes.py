"""What a write through ``/api/v1/search-tools`` must obey, and the default it stores first.

The instance rules of the web search design (names, entries, options, the
lenient load) as a write meets them: refused in what the write sets, so an
instance stored before the rules keeps working until it is edited. Each
refusal is a ``SearchToolRefusedError``, a 422.
"""

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any

from gateway.core.config import GatewayConfig
from gateway.core.settings.tools import (
    BUILTIN_FETCH,
    RESERVED_INSTANCE_NAMES,
    SEARCH_PROVIDERS_REQUIRING_API_BASE,
    ToolKind,
    effective_fetch_instances,
    instance_name_problems,
    option_problems,
    search_default_to_pin,
    validate_fetch_tool_entry,
    validate_search_tool_entry,
    validate_search_tool_transport,
)
from gateway.exceptions.tools_exceptions import SearchToolRefusedError
from gateway.services.tool_settings_service import validate_url


def _section(kind: ToolKind) -> str:
    """The configuration file's map for ``kind``, as its messages name it."""
    return "search_tools" if kind == "search" else "fetch_tools"


def check_tool_entry(
    name: str, entry: Mapping[str, Any], *, kind: ToolKind, inherited_api_base: str | None = None
) -> None:
    """Hold a dashboard-written instance to the rules the config file is held to.

    The same validation startup runs on, per kind, so an instance saved here can
    never be one that would refuse to boot from a config file. A ``searxng``
    instance with no ``api_base`` is checked with the one it inherits.
    """
    try:
        if kind == "search":
            validate_search_tool_entry(name, dict(entry))
        else:
            validate_fetch_tool_entry(name, dict(entry))
    except ValueError as exc:
        raise SearchToolRefusedError(str(exc)) from None
    provider = str(entry.get("provider") or name)
    api_base = entry.get("api_base")
    if not api_base and kind == "search" and provider in SEARCH_PROVIDERS_REQUIRING_API_BASE:
        api_base = inherited_api_base
    if api_base:
        try:
            validate_url(str(api_base))
            validate_search_tool_transport(name, api_base, entry.get("api_key"))
        except ValueError as exc:
            raise SearchToolRefusedError(str(exc)) from None


def check_tool_write(
    config: GatewayConfig,
    name: str,
    entry: Mapping[str, Any],
    *,
    kind: ToolKind,
    sets_name: bool,
    sets_options: bool,
    sets_fetch_tool: bool,
) -> None:
    """Refuse what the instance rules let load but not be written.

    Only in what the write sets: a search instance stored before the rules keeps
    loading with its name and options, so a change that leaves them alone, such
    as rotating its key, does not trip on them. A fetch instance's name is held
    to the rules by its entry's own validation.
    """
    provider = str(entry.get("provider") or name)
    problems = instance_name_problems(name) if sets_name and kind == "search" else []
    if sets_options:
        problems += option_problems(kind, provider, entry.get("options") or {}).values()
    if problems:
        raise SearchToolRefusedError(f"{_section(kind)}.{name} is refused: {'; '.join(problems)}.")
    fetch_tool = entry.get("fetch_tool")
    if not sets_fetch_tool or fetch_tool is None:
        return
    if kind == "fetch":
        raise SearchToolRefusedError(f"fetch_tools.{name}.fetch_tool is refused: only a search instance has one.")
    if fetch_tool not in effective_fetch_instances(config):
        raise SearchToolRefusedError(
            f"search_tools.{name}.fetch_tool must name a fetch instance, or {BUILTIN_FETCH}; "
            f"there is no '{fetch_tool}'."
        )


def check_kind_unchanged(name: str, kind: ToolKind, requested: ToolKind | None) -> None:
    """Refuse an update that changes an instance's kind, so it never moves between the search and fetch maps."""
    if requested is not None and requested != kind:
        raise SearchToolRefusedError(
            f"'{name}' is a {kind} instance, and an instance's kind cannot change: delete it and create it "
            f"again as a {requested} instance."
        )


def check_tool_name_is_free(config: GatewayConfig, name: str, kind: ToolKind) -> None:
    """Refuse a name an instance of the other kind already has.

    Names are unique across search and fetch instances, because the pricing key
    ``<provider>:<instance>`` carries no capability. A stored instance may still
    take the name of a config-file one of its own kind, which it then overrides.
    """
    other: ToolKind = "fetch" if kind == "search" else "search"
    if name in (config.fetch_tools if kind == "search" else config.search_tools):
        raise SearchToolRefusedError(
            f"A {other} instance named '{name}' exists; names are unique across search and fetch instances."
        )


@dataclass(frozen=True)
class DefaultPin:
    """The ``web_search_default_tool`` a create stores before it adds a second search instance.

    ``notice`` is what the create's response tells the operator about it.
    """

    name: str
    notice: str


def _pin_refusal(pinned: str, *, stored: bool) -> str:
    """Why the create cannot pin ``pinned``, and how the operator gets past it."""
    way_out = (
        f"delete '{pinned}' and create it again under another name, since a stored name cannot change"
        if stored
        else f"rename '{pinned}' in the configuration file"
    )
    return (
        f"Adding a second search instance would leave the in-loop tool with no default, so this create first "
        f"sets web_search_default_tool to '{pinned}', the only search instance until now. That name is reserved "
        f"and cannot be the default: {way_out}, then add this one."
    )


def default_to_pin(config: GatewayConfig, new_name: str, *, stored_names: Collection[str]) -> DefaultPin | None:
    """The default a create of the search instance ``new_name`` stores first, if any.

    When the one search instance is the in-loop default only because it is
    alone, adding a second would turn in-loop search off, so the create names it
    as the default in the same commit. An instance whose name is reserved cannot
    be named, so that create is refused; ``stored_names`` says whether the way
    out is to recreate the instance or to rename it in the configuration file.
    """
    pinned = search_default_to_pin(config, new_name)
    if pinned is None:
        return None
    if pinned.name.lower() in RESERVED_INSTANCE_NAMES:
        raise SearchToolRefusedError(_pin_refusal(pinned.name, stored=pinned.name in stored_names))
    return DefaultPin(
        name=pinned.name,
        notice=(
            f"web_search_default_tool is now '{pinned.name}'. The in-loop tool searched with it as the only search "
            f"instance, and adding '{new_name}' would otherwise have turned in-loop search off. This runtime value "
            "wins over the configuration file until it is cleared."
        ),
    )
