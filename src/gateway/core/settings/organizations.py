"""Settings for organizations: how far an invitation goes, and how much invitation mail is sent."""

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
    invitation_per_address_per_day: Annotated[int, Shown(SettingsGroup.MAIL)] = Field(
        default=3,
        ge=1,
        description=(
            "Invitations one organization, and one inviting account, may each send to one address in 24 "
            "hours. It does not bound an address across organizations; invitation_mail_limit_per_address "
            "does that. Counted in rate_limit_store."
        ),
    )
    invitation_per_day: Annotated[int, Shown(SettingsGroup.MAIL)] = Field(
        default=100,
        ge=1,
        description=(
            "Invitations one organization, and one inviting account, may each ask for in 24 hours. Counted "
            "in rate_limit_store."
        ),
    )
    invitation_resend_limit: Annotated[int, Shown(SettingsGroup.MAIL)] = Field(
        default=5,
        ge=1,
        description=(
            "How many times one invitation may be sent again after its first send. A deployment that "
            "also wants to bound how often an address hears from it sets invitation_per_address_per_day, "
            "which is counted over 24 hours rather than over the invitation's life."
        ),
    )
    invitation_resend_wait_minutes: Annotated[int, Shown(SettingsGroup.MAIL)] = Field(
        default=15,
        ge=1,
        description="How long to wait between sends of one invitation, in minutes.",
    )
