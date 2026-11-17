"""Deleting the signed-in account, at its owner's request.

``DELETE /api/v1/auth/account`` always acts on the caller's own identity. The
caller types the account's address to confirm, so a stray click cannot delete
it, and gives its password when it has one.
"""

from fastapi import APIRouter, BackgroundTasks, Depends, Request, Response, status
from pydantic import BaseModel, Field

from gateway.api.deps import AccountDeletionServiceDep, CurrentIdentity, PersonGrowthSignalPortDep, verify_master_key
from gateway.core.addresses import MAX_STORED_ADDRESS_LENGTH
from gateway.services.dashboard_session_service import clear_session_cookie, request_is_https
from gateway.services.password_service import MAX_PASSWORD_BYTES

router = APIRouter(
    prefix="/auth/account",
    tags=["auth"],
    dependencies=[Depends(verify_master_key)],
)


class DeleteAccountRequest(BaseModel):
    """The confirmation an account deletion needs."""

    model_config = {"json_schema_extra": {"example": {"confirm_email": "ada@example.com", "current_password": "..."}}}

    confirm_email: str = Field(
        max_length=MAX_STORED_ADDRESS_LENGTH,
        description="The account's email address, typed to confirm the deletion.",
    )
    current_password: str | None = Field(
        default=None,
        max_length=MAX_PASSWORD_BYTES,
        description="The account's password. Required when the account has one.",
    )


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
async def delete_own_account(
    body: DeleteAccountRequest,
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    identity: CurrentIdentity,
    accounts: AccountDeletionServiceDep,
    growth: PersonGrowthSignalPortDep,
) -> None:
    """Delete the caller's account and sign them out.

    Erases the account's sign-in identity, sessions, passkeys, Playground
    conversations, agent traces, telemetry, files and keys. Usage rows are
    kept for the deployment's totals under a new id that does not identify
    the person, and an organization nobody else is active in is renamed, its
    keys revoked and its stored credentials cleared. Invitations to the
    account's address are withdrawn everywhere. Refused while the account is the last active owner of
    an organization other people are active in.
    """
    deleted = await accounts.delete(identity, confirm_email=body.confirm_email, current_password=body.current_password)
    if growth is not None and deleted.email is not None:
        await growth.record_account_deleted(background_tasks=background_tasks, user_id=deleted.id, email=deleted.email)
    clear_session_cookie(response, secure=request_is_https(request))
