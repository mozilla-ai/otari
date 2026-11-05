"""The in-loop default: which search the in-loop tool runs on when no credential applies.

Whether a deployment can run ``otari_web_search`` at all is whether this default
yields a search, so the request path, the tool catalog, the Playground and the
workspace page all ask :func:`web_search_configured` and cannot disagree.
"""

import any_search
from gateway.core.settings.tools import (
    NO_SEARCH_DEFAULT,
    SynthesizedSearchInstance,
    ToolInstance,
    ToolSettings,
    configured_search_default,
    dangling_reference_warnings,
    default_api_base,
    effective_search_instances,
    synthesized_search_instance,
    warn_about_instances,
)
from gateway.log_config import logger


def in_loop_default(settings: ToolSettings) -> ToolInstance | SynthesizedSearchInstance | None:
    """The search backend the in-loop tool uses when no credential applies.

    In order: ``web_search_default_tool`` when it names a search instance
    any-search can run, or ``none``, which turns the deployment's own off;
    else the synthesized instance while the legacy settings are present; else
    the one search instance, when there is exactly one and any-search can run
    it; else nothing. A default that names no such instance is treated as
    unset. The count in the third step includes instances any-search does not
    serve yet, so an adapter that arrives later never moves the default.
    """
    instances = effective_search_instances(settings)
    named = configured_search_default(settings)
    if named == NO_SEARCH_DEFAULT:
        return None
    if named is not None and (instance := instances.get(named)) is not None and _runs(settings, instance):
        return instance
    if (synthesized := synthesized_search_instance(settings)) is not None:
        return synthesized
    if len(instances) == 1:
        (only,) = instances.values()
        return only if _runs(settings, only) else None
    return None


def _runs(settings: ToolSettings, instance: ToolInstance) -> bool:
    """Whether any-search serves the instance's provider, with an address if it needs one."""
    if not instance.library_backed:
        return False
    needs_address = any_search.AnySearch.get_provider_metadata(instance.provider).requires_api_base
    return not needs_address or bool(instance.api_base or default_api_base(settings, instance.provider))


def web_search_configured(settings: ToolSettings) -> bool:
    """Whether this deployment can run ``otari_web_search`` at all: whether the in-loop default yields a search.

    A deployment whose only search configuration is one instance any-search
    serves answers yes, and ``web_search_default_tool: none`` answers no,
    whatever else is set.
    """
    return in_loop_default(settings) is not None


def search_default_to_pin(settings: ToolSettings, new_name: str) -> ToolInstance | None:
    """The search instance to store as ``web_search_default_tool`` before ``new_name`` is added, if any.

    Adding a second search instance would end step 3 of the in-loop default,
    which needs exactly one, and so turn in-loop search off by accident. So
    when the one instance there is the in-loop default through that step, the
    write that adds the second names it first. A ``new_name`` the map already
    holds replaces an entry rather than adding one.
    """
    instances = effective_search_instances(settings)
    if new_name in instances or len(instances) != 1:
        return None
    (only,) = instances.values()
    default = in_loop_default(settings)
    from_step_3 = configured_search_default(settings) != only.name
    return only if from_step_3 and isinstance(default, ToolInstance) and default.name == only.name else None


def warn_about_tool_instances(settings: ToolSettings) -> None:
    """Log what the instance rules let load but an operator should fix.

    Run once at startup, after stored search tools and runtime settings have
    loaded, so a name stored through the dashboard is one the checks can see.
    Names and problems only, never a key, an option's value or a URL.
    """
    search = effective_search_instances(settings)
    warn_about_instances(settings, search, settings.fetch_tools)
    for value, message in dangling_reference_warnings(settings).values():
        logger.warning(message, value)
    named = configured_search_default(settings)
    if named != NO_SEARCH_DEFAULT and len(search) > 1 and in_loop_default(settings) is None:
        usable = sorted(name for name, instance in search.items() if _runs(settings, instance))
        logger.warning(
            "There are %d search instances and no usable web_search_default_tool, so the in-loop "
            "otari_web_search tool has no default to search with. %s",
            len(search),
            f"Set web_search_default_tool to one of {', '.join(usable)}, or to 'none' to keep in-loop search off."
            if usable
            else "None of them is on a provider the in-loop tool can use yet; set web_search_default_tool to "
            "'none' to say in-loop search stays off.",
        )
