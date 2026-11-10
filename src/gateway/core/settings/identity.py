"""Settings for identity: how a person signs in, and who their account belongs to."""

from typing import Annotated

from pydantic import BaseModel, Field

from gateway.core.settings_view import SettingsGroup, Shown


class IdentitySettings(BaseModel):
    oidc_allow_private_hosts: Annotated[bool, Shown(SettingsGroup.TOOLS)] = Field(
        default=False,
        description=(
            "Let an organization's own OIDC issuer be discovered at a private, loopback or link-local "
            "address, for an identity provider on the deployment's own network. Off by default, because "
            "an organization administrator types the issuer in rather than the operator choosing it."
        ),
    )
