"""Settings for deliberate product feedback."""

from typing import Annotated

from pydantic import BaseModel, Field

from gateway.core.settings_view import SettingsGroup, Shown


class FeedbackSettings(BaseModel):
    feedback_enabled: Annotated[bool, Shown(SettingsGroup.GENERAL)] = Field(
        default=True,
        description="Allow deliberate feedback submissions to the Otari team at api.otari.ai. Requires restart.",
    )
