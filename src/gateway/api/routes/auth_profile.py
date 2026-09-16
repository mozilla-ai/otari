"""The name a signed-in identity goes by, which is the one thing about itself it can change.

``PATCH /api/v1/auth/profile`` always acts on the caller's own identity: it takes
no id, and an operator changing somebody else's name belongs to the deployment
administration surface rather than here.

It sits beside the password and passkey routes because it answers the same
question they do, "what may the signed-in identity change about itself", and it
is what the account page was missing. ``full_name`` is published on the
membership context so the sidebar can draw a person rather than a role
(mozilla-ai/otari#832), and an identity added to a roster by address carries
none until somebody supplies one; without this there was nowhere to.

The address is deliberately not editable here. Changing a sign-in address is a
credential change with a verification flow behind it, and
``PUT /api/v1/auth/password`` already refuses one
(``EmailChangeNotSupportedError``); this endpoint would be the wrong place to
open that door quietly.
"""

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import CurrentIdentity, get_db, verify_master_key
from gateway.models.tenancy import MAX_FULL_NAME_LENGTH, CallerIdentityPublic
from gateway.services.tenancy.user_service import update_full_name

# Declared on the router rather than left to arrive through ``CurrentIdentity``,
# matching `auth_password.py`: the credential check is then a property of the
# route and not of an argument a handler could forget.
router = APIRouter(
    prefix="/auth/profile",
    tags=["auth"],
    dependencies=[Depends(verify_master_key)],
)


class UpdateProfileRequest(BaseModel):
    """The caller's own display name, or ``null`` to go back to having none."""

    model_config = {"json_schema_extra": {"example": {"full_name": "Ada Lovelace"}}}

    # Required rather than optional, so there is no third meaning to argue
    # about: a name, or null for no name. An omitted field on a body with one
    # field is a request that meant something and lost it.
    #
    # Bounded at the column width, which is what a longer value would be
    # truncated to or refused by depending on the engine. Surrounding whitespace
    # is collapsed by the service afterwards, so this bounds the raw value a
    # little more tightly than the stored one; a 255-character name padded with
    # spaces is the only thing that costs, and it is not a name anybody types.
    full_name: str | None = Field(
        max_length=MAX_FULL_NAME_LENGTH,
        description=(
            "The name to be known by on this deployment, or null to have none. Whitespace is "
            "collapsed, and a value with nothing else in it is stored as null, which leaves "
            "every surface naming this identity by its address again."
        ),
    )


@router.patch("")
async def update_own_profile(
    body: UpdateProfileRequest,
    identity: CurrentIdentity,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CallerIdentityPublic:
    """Change the name the caller is known by on this deployment.

    Always the caller's own identity. The same shape
    ``GET /api/v1/organizations/me`` carries as ``caller`` comes back, so a
    client can seat the answer where it read the old value rather than refetch
    the whole membership context.
    """
    updated = await update_full_name(db, identity, full_name=body.full_name)
    return CallerIdentityPublic(
        user_id=updated.id,
        email=updated.email,
        full_name=updated.full_name,
        has_password=updated.hashed_password is not None,
    )


__all__ = ["router"]
