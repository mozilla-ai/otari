"""How many web search results a request gets when it names no ceiling of its own."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gateway.core.env import otari_env
from gateway.log_config import logger
from gateway.services.web_retrieval_backend import DEFAULT_MAX_RESULTS

if TYPE_CHECKING:
    from gateway.core.config import GatewayConfig


def web_search_max_results_baseline(config: GatewayConfig | None) -> int:
    """How many results a request that names no ``max_results`` of its own gets.

    The deployment's own setting, or the backend's built-in default when it has none.
    """
    config_max = config.web_search_max_results if config is not None else None
    if config_max is not None:
        return config_max
    max_env = otari_env("WEB_SEARCH_MAX_RESULTS")
    if max_env:
        try:
            parsed_max = int(max_env)
        except ValueError:
            logger.warning("OTARI_WEB_SEARCH_MAX_RESULTS=%r is not an int; ignoring", max_env)
        else:
            if parsed_max >= 1:
                return parsed_max
            logger.warning("OTARI_WEB_SEARCH_MAX_RESULTS=%r is not >= 1; ignoring", max_env)
    return DEFAULT_MAX_RESULTS
