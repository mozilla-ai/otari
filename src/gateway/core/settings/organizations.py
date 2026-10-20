"""Settings for organizations: how much invitation mail an organization and an address may be sent."""

from typing import Annotated

from pydantic import BaseModel, Field

from gateway.core.settings_view import SettingsGroup, Shown

# The window both invitation mail limits are counted over.
INVITATION_MAIL_WINDOW_HOURS = 24


class OrganizationSettings(BaseModel):
    invitation_mail_limit_per_organization: Annotated[int | None, Shown(SettingsGroup.MAIL)] = Field(
        default=250,
        ge=1,
        description=(
            "Invitation emails one organization may send in 24 hours (None disables this limit). An invitation "
            "over the limit is still created, and its accept link is returned to share by hand. Counted in "
            "rate_limit_store."
        ),
    )
    invitation_mail_limit_per_address: Annotated[int | None, Shown(SettingsGroup.MAIL)] = Field(
        default=5,
        ge=1,
        description=(
            "Invitation emails one address may receive in 24 hours, from every organization together (None "
            "disables this limit). An invitation over the limit is still created, and its accept link is returned "
            "to share by hand. Counted in rate_limit_store."
        ),
    )
