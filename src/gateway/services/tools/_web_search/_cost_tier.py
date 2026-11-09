"""A tool whose cost tier has a value its provider refuses does not run.

A provider's cost tier picks the price of each call, and a tool is priced for
the tier its configuration names. Load drops a value the provider refuses,
with a warning, so a call would run at the provider's default tier, at
another price than the tool's.
"""

import any_fetch
import any_search
from gateway.core.settings.tools import ToolInstance


def refused_cost_tier(instance: ToolInstance) -> str | None:
    """The name of the instance's cost tier option when its provider refuses the value, else ``None``."""
    tier = _cost_tier(instance)
    return tier if tier in instance.dropped_options else None


def _cost_tier(instance: ToolInstance) -> str | None:
    """The provider's cost tier option, one at most; none for a provider the instance's library does not serve."""
    if instance.kind == "search":
        if instance.provider not in any_search.AnySearch.get_supported_providers():
            return None
        search_options = any_search.AnySearch.get_provider_metadata(instance.provider).options
        return next((option.name for option in search_options if option.cost_tier), None)
    if instance.provider not in any_fetch.AnyFetch.get_supported_providers():
        return None
    fetch_options = any_fetch.AnyFetch.get_provider_metadata(instance.provider).options
    return next((option.name for option in fetch_options if option.cost_tier), None)
