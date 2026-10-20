"""Settings for the scheduled health checks of the models the hosted providers serve."""

from typing import Annotated

from pydantic import BaseModel, Field

from gateway.core.settings_view import SettingsGroup, Shown


class HealthCheckFeatureSettings(BaseModel):
    health_checks_enabled: Annotated[bool, Shown(SettingsGroup.MODELS)] = Field(
        default=True,
        description=(
            "Offer the Health page, which checks the models the hosted providers serve on a schedule an operator "
            "turns on there. Off removes the page, its routes and its worker. Requires restart."
        ),
    )
