"""Settings for the check for new otari releases and security fixes."""

import os
from typing import Annotated

from pydantic import BaseModel, Field

from gateway.core.settings_view import OMITTED, SettingsGroup, Shown

DEFAULT_UPDATE_CHECK_URL = "https://api.otari.ai/api/v1/update-checks"


class UpdateCheckSettings(BaseModel):
    update_check: Annotated[bool | None, Shown(SettingsGroup.GENERAL)] = Field(
        default=None,
        description=(
            "Check otari.ai for new releases and security fixes, sending this deployment's version and a random "
            "install id. Unset asks an operator in the dashboard and sends nothing until one answers; true or "
            "false answers for them and locks the dashboard switch."
        ),
    )
    update_check_url: Annotated[str, OMITTED] = Field(
        default=DEFAULT_UPDATE_CHECK_URL,
        description="Where the update check is sent.",
    )


def preset_answer(config: UpdateCheckSettings) -> bool | None:
    """The answer the deployment's own configuration gives, or None when an operator is to be asked.

    ``DO_NOT_TRACK`` (https://consoledonottrack.com) and ``CI`` answer no, unless ``update_check`` says otherwise,
    so a CI run and an operator who opts out of every tool at once are neither asked nor counted.
    """
    if config.update_check is not None:
        return config.update_check
    if os.environ.get("DO_NOT_TRACK", "").strip().lower() in {"1", "true", "yes"} or os.environ.get("CI"):
        return False
    return None
