"""Who the signed-in identity is, and the two things about itself it can change: its name and its avatar.

``GET`` and ``PATCH /api/v1/auth/profile``, and ``PUT /api/v1/auth/profile/avatar``,
always act on the caller's own identity: they take no id, and an operator
reading or changing somebody else's name belongs to the deployment
administration surface rather than here.

``GET`` answers whether or not the caller belongs to an organization, which
``GET /api/v1/organizations/me`` cannot: it 404s for an identity with no active
membership, and the dashboard still has to know whose session it holds to name
the account and to tell an invitation for another address apart.

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
from gateway.models.tenancy import AVATAR_FACE_COUNT, MAX_FULL_NAME_LENGTH, CallerIdentityPublic
from gateway.services.tenancy import describe_caller, update_avatar_face, update_full_name

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


class UpdateAvatarRequest(BaseModel):
    """The caller's own avatar: a preset face by index, or ``null`` for the one their id gives them."""

    model_config = {"json_schema_extra": {"example": {"face": 1234}}}

    # Required for the reason ``full_name`` is: a face, or null for none.
    face: int | None = Field(
        strict=True,
        ge=0,
        lt=AVATAR_FACE_COUNT,
        description=(
            "The index of a preset face from the dashboard's avatar set, or null to go back to "
            "the face this identity's id gives it."
        ),
    )


@router.get("")
async def get_own_profile(
    identity: CurrentIdentity,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CallerIdentityPublic:
    """Report who the caller is, whether or not they belong to an organization.

    The same shape ``GET /api/v1/organizations/me`` carries as ``caller``, for
    an identity that context cannot describe because it has no active
    membership.
    """
    return await describe_caller(db, identity)


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
    return await describe_caller(db, updated)


@router.put("/avatar")
async def update_own_avatar(
    body: UpdateAvatarRequest,
    identity: CurrentIdentity,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CallerIdentityPublic:
    """Change the avatar the caller is drawn with, everywhere their organization lists them.

    Always the caller's own identity, and the same answer as ``PATCH``, for the
    same reason.
    """
    updated = await update_avatar_face(db, identity, face=body.face)
    return await describe_caller(db, updated)


__all__ = ["router"]
