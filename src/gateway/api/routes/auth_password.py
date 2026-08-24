"""Setting and changing the password a dashboard sign-in uses.

One endpoint, ``PUT /v1/auth/password``, and it always acts on the caller's own
identity. There is deliberately no way here for an admin to set somebody else's
password: an address on the roster with no password is an identity waiting for
the signup and reset flows in #650, not one an operator should be able to take
over.

**Claiming a deployment.** First boot leaves a standalone deployment with an
operator identity that has no address and no password
(`gateway.services.tenancy.provisioning_service`), and with the master key still
accepted as the dashboard login. The first call here *by that identity* supplies
an address and a password, and that single act is what retires master-key
sign-in and turns email and password into the steady-state login
(mozilla-ai/otari-ai#1716). Nothing schedules it and nothing expires: a
deployment that never claims goes on signing in with the master key
indefinitely, including one where every member has since set a password of their
own, because the credential this retires resolves to the operator and to nobody
else (#702).

**Which proof is required, and why it differs.** The table is small and every
row has a reason:

| Caller | Identity has a password | Required |
| --- | --- | --- |
| Master key, in a header | no | nothing further: the claim |
| Master key, in a header | yes | nothing further: operator recovery |
| Session cookie | no | nothing further: only a master-key sign-in could have minted that session |
| Session cookie | yes | the current password |

The master key never has to present the current one because it is the
deployment-wide credential: a caller holding it can already do anything the
management API can, so demanding a password they have forgotten would lock the
dashboard while leaving the API open. That is what keeps this change from being
able to strand an operator, and it is why no separate reset endpoint is needed
before #650 ships one for people who hold no master key.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import CurrentIdentity, get_db, verify_master_key
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, hash_session_token
from gateway.services.password_service import MIN_PASSWORD_LENGTH
from gateway.services.tenancy.email_address import MAX_EMAIL_LENGTH, validated_email
from gateway.services.tenancy.errors import CurrentPasswordRequiredError, EmailChangeNotSupportedError
from gateway.services.tenancy.user_service import operator_has_password, set_password, update_password

# Auth is declared on the router rather than left to arrive through
# ``CurrentIdentity``, matching `organizations.py`: the credential check is then
# a property of the route and not of an argument a handler could forget.
# A deliberately generous ceiling, and deliberately *not* ``MAX_PASSWORD_BYTES``.
# The real limit is counted in bytes, so the readable refusal ("at most 72 bytes;
# accented characters count for more than one") has to come from
# ``_validate_password`` in the service rather than from a character count here.
# A schema bound of 72 would pre-empt it with a less useful 422 and would refuse
# a 73-character password with the wrong explanation. This bound exists only to
# stop an absurd body being buffered and hashed, which is a different job from
# the policy. The sign-in schema bounds its own password at ``MAX_PASSWORD_BYTES``
# for the opposite reason: nothing longer could match a stored hash, so there is
# no policy message to preserve there.
_MAX_SUBMITTED_PASSWORD = 1024

router = APIRouter(
    prefix="/v1/auth/password",
    tags=["auth"],
    dependencies=[Depends(verify_master_key)],
)


class SetPasswordRequest(BaseModel):
    """Set or change the signed-in identity's password.

    The example is the first-boot claim, because that is the call an operator
    makes first and the one whose required fields are not obvious from the
    schema: ``email`` is optional here in general and *required* when the
    identity has no address yet. Without it the generated Postman body carries
    only ``new_password``, which is the one shape that cannot complete the flow
    the docs walk through.
    """

    model_config = {
        "json_schema_extra": {
            "example": {
                "email": "operator@example.com",
                "new_password": "a-real-password",
            }
        }
    }

    new_password: str = Field(
        min_length=MIN_PASSWORD_LENGTH,
        max_length=_MAX_SUBMITTED_PASSWORD,
        description=(
            "The password to sign in with from now on. At least "
            f"{MIN_PASSWORD_LENGTH} characters, and at most 72 bytes, which is bcrypt's ceiling."
        ),
    )
    current_password: str | None = Field(
        default=None,
        max_length=_MAX_SUBMITTED_PASSWORD,
        description=(
            "The password being replaced. Required when the identity already has one and the "
            "request is authenticated by the session cookie; ignored when the master key is sent "
            "in a header, which needs no proof of the old password (it still needs `email` when "
            "the identity has no sign-in address yet)."
        ),
    )
    # The column is ``varchar(255)``, so a longer address is a 500 from the
    # driver rather than a refusal the caller can read. This bounds the raw
    # value; ``validated_email`` bounds the normalized one, which is a different
    # number when lower-casing lengthens the string.
    email: str | None = Field(
        default=None,
        max_length=MAX_EMAIL_LENGTH,
        description=(
            "The address to sign in with. Required when the identity has none, which is the state "
            "first boot leaves the operator in, including when the master key is what authenticates "
            "the call. Resubmitting the address the identity already holds is accepted and ignored; "
            "only a *different* address is refused, because changing one is not supported yet."
        ),
    )


class PasswordResponse(BaseModel):
    """What the identity signs in with now."""

    email: str = Field(description="The address this identity signs in with.")
    master_key_sign_in_retired: bool = Field(
        description=(
            "Whether POST /v1/auth/session has stopped accepting the master key as a dashboard "
            "login. True once the operator identity has a password, which is what claiming the "
            "deployment means; a member setting their own password leaves an unclaimed deployment "
            "on the master key. Either way the master key stays the credential for the management API."
        )
    )


@router.put("")
async def set_dashboard_password(
    body: SetPasswordRequest,
    request: Request,
    identity: CurrentIdentity,
    db: Annotated[AsyncSession, Depends(get_db)],
    master_key: Annotated[str | None, Depends(verify_master_key)],
) -> PasswordResponse:
    """Set or change the password the caller signs in to the dashboard with.

    Always the caller's own identity. Supply ``email`` when it has no sign-in
    address yet, which is the state first boot leaves the operator in, and
    ``current_password`` when it already has a password and the request is
    authenticated by the session cookie. The master key in a header is what
    excuses ``current_password``, which is how a forgotten password is
    recovered; it does not excuse ``email``, because an identity with no address
    has nothing to sign in with whoever is asking. The operator setting a password
    for the first time retires master-key sign-in on this deployment.

    Every other session this identity holds ends, the caller's own excepted, so
    a cookie stolen before the change does not outlive it.
    """
    # ``verify_master_key`` returns the raw key when the request was
    # header-authenticated and None when the session cookie authenticated it, so
    # this is the "does the caller hold the deployment credential" signal the
    # table in the module docstring turns on, rather than a re-read of the
    # headers here.
    by_master_key = master_key is not None
    keep_session = _caller_session_hash(request, by_master_key=by_master_key)
    # Resubmitting the address the identity already holds is not a change, and
    # refusing it would break any client that keeps one form for both claiming
    # and changing a password. Normalized on both sides, because the stored
    # value is only lower-cased when this tree wrote it: a row from a
    # convergence backfill or from an operator's own SQL can carry any casing,
    # which ``UserRepository.get_by_email`` already accounts for, and comparing
    # a normalized candidate against a raw column would refuse an identity its
    # own address.
    #
    # The stored side gets ``.strip().lower()`` and not ``validated_email``,
    # matching how ``get_by_email`` normalizes its argument. Running the shape
    # check over a column value would raise ``InvalidEmailError`` for a stored
    # address that is malformed or over-width once normalized, which tells a
    # caller who submitted a perfectly good address that theirs is invalid, and
    # quotes the stored one back at them. Those rows are exactly the ones this
    # comparison exists for, so it must not be the thing that refuses them.
    if body.email is not None and identity.email is not None:
        if validated_email(body.email) != identity.email.strip().lower():
            raise EmailChangeNotSupportedError
        body = body.model_copy(update={"email": None})
    if identity.hashed_password is not None and not by_master_key:
        if body.current_password is None:
            raise CurrentPasswordRequiredError
        await update_password(
            db,
            identity,
            current_password=body.current_password,
            new_password=body.new_password,
            keep_session_token_hash=keep_session,
        )
    else:
        await set_password(
            db,
            identity,
            new_password=body.new_password,
            email=body.email,
            keep_session_token_hash=keep_session,
        )
    assert identity.email is not None  # guaranteed by set_password
    # Read back rather than hardcoded True: this route acts on the caller's own
    # identity, and only the operator's password retires master-key sign-in
    # (#702). A member changing theirs on an unclaimed deployment answers False,
    # which is what stops the dashboard telling them it retired a login that is
    # still the only one their operator has.
    return PasswordResponse(email=identity.email, master_key_sign_in_retired=await operator_has_password(db))


def _caller_session_hash(request: Request, *, by_master_key: bool) -> str | None:
    """The stored hash of the caller's own session, so the change does not sign them out.

    None for a header-authenticated caller. A request carrying header
    credentials is never authenticated by its cookie (``get_session_identity``
    refuses to look), so a cookie that happens to ride along on such a request
    is not "the caller's session" and gets no exemption.
    """
    if by_master_key:
        return None
    token = request.cookies.get(SESSION_COOKIE_NAME)
    return hash_session_token(token) if token else None


__all__ = ["router"]
