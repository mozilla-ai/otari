"""The plugin's own settings, read from its own environment variables.

A plugin does not get fields on ``GatewayConfig``: adding two there would put a
surface Otari does not ship into the config every deployment loads, and into
the settings page's allowlists, for a feature most deployments do not install.
So the knobs live here and keep the ``OTARI_ALERT_`` names they had while the
feature was core, because those names are what a deployment already writes in
its environment and renaming them would buy nothing.

Validated at import, and :func:`otari_alerts.register` imports this module, so
a bad value fails the boot rather than the first evaluation tick.
"""

from typing import Final

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AlertSettings(BaseSettings):
    """Environment-only settings for the budget alerts plugin."""

    model_config = SettingsConfigDict(env_prefix="OTARI_", extra="ignore")

    alert_evaluation_interval_sec: int = Field(
        default=60,
        ge=0,
        description=(
            "How often to check the organization budget ceilings against their alert rules' "
            "thresholds and send what has crossed one. 0 disables alert evaluation entirely, "
            "leaving configured rules in place and sending nothing."
        ),
    )
    alert_allow_private_hosts: bool = Field(
        default=False,
        description=(
            "SSRF gate: allow an alert destination whose URL names a host (json, mailto, gotify, "
            "ntfy, matrix and the rest of that group) that resolves to a private, loopback or "
            "reserved address. Off by default. Enable it to alert an internal chat server or "
            "webhook receiver on the deployment's own network. Schemas with endpoints compiled "
            "into Apprise (slack, discord, pagerduty and the rest) have no address to check and "
            "are unaffected."
        ),
    )


# Read once so an unparseable value fails the boot. The interval is a startup
# decision either way: the evaluator task is created in the lifespan.
settings: Final = AlertSettings()


def allow_private_hosts() -> bool:
    """Whether the SSRF gate is open, read from the environment on every call.

    Re-read rather than taken off :data:`settings`, matching how the gateway's
    sibling gates (``OTARI_WEB_SEARCH_ALLOW_PRIVATE_HOSTS`` and the rest) are
    consulted: it is checked only on a rule write, never on a hot path, and a
    per-call read is what lets a process set it without being restarted first.
    """
    return AlertSettings().alert_allow_private_hosts
