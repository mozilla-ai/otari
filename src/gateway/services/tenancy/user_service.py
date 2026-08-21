"""Password sign-in for the reconciled control plane's identities.

Ported from the platform's ``user_service`` (``authenticate`` and
``update_password``), which is where the shape of this comes from. What it is
*for* here is narrower, and settled by mozilla-ai/otari-ai#1716: the master key
bootstraps a standalone deployment and stays its deployment-wide API credential,
while a password against a `gateway.services.dashboard_session_service` session
becomes the steady-state dashboard login. Retiring a login is not retiring a
credential, so nothing here touches what authenticates ``/v1/keys``,
``/v1/users``, or the rest of the management surface.

Four departures from the port, each for a reason that belongs to this edition:

- **Sign-in failures collapse into one error.** See ``InvalidCredentialsError``.
- **``email_verified_at`` is written but not checked.** The verification flow is
  #650's, and enforcing a gate before the flow that clears it exists would lock
  out the operator who just claimed the deployment. Setting a password through
  the deployment credential stamps the column, so when #650 turns the check on
  it is an added gate rather than a migration: the identities that exist by then
  already pass it.
- **A password can be set without proving the old one, by the master key.** That
  is the claim path and the recovery path in one, and it is not a weakening: a
  caller holding the master key can already do anything the management API can
  do, so asking them for a password they have forgotten would only lock the
  dashboard while leaving the API wide open.
- **A deactivated identity does not authenticate.** The platform's
  ``authenticate`` does not check ``is_active``; this one does, so deactivating
  someone ends their access now rather than when their cookie expires, which is
  the rule ``resolve_dashboard_session`` already follows.

An identity with an address and no password is the normal state for someone an
admin added to the roster (`organization_service`); they cannot sign in until
#650 gives them a way to set one, and nothing here lets an admin set it for
them.
"""

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.models.tenancy import User
from gateway.repositories.tenancy import UserRepository
from gateway.services.dashboard_session_service import revoke_user_dashboard_sessions
from gateway.services.password_service import (
    MAX_PASSWORD_BYTES,
    MIN_PASSWORD_LENGTH,
    hash_password_async,
    verify_absent_password_async,
    verify_password_async,
)
from gateway.services.tenancy.email_address import validated_email
from gateway.services.tenancy.errors import (
    CurrentPasswordIncorrectError,
    EmailAlreadyInUseError,
    InvalidCredentialsError,
    PasswordNotSetError,
    PasswordPolicyError,
    SignInAddressRequiredError,
    UnmodifiedPasswordError,
)

# The unique index alembic creates on ``user.email`` (c4b6d8e0f2a3).
_EMAIL_UNIQUE_INDEX = "ix_user_email"


async def authenticate(db: AsyncSession, *, email: str, password: str) -> User:
    """Return the identity this address and password authenticate.

    Every failure raises the same ``InvalidCredentialsError``, and the ones that
    have no stored hash to check against still pay for one verification, so a
    caller cannot learn from the answer or from its timing which addresses hold
    an account here.

    A deactivated identity does not authenticate, which matches
    ``resolve_dashboard_session``: deactivating someone has to end their access
    now, not once their current session expires.
    """
    identity = await UserRepository(db).get_by_email(email)
    if identity is None or not identity.is_active or identity.hashed_password is None:
        await verify_absent_password_async(password)
        raise InvalidCredentialsError
    if not await verify_password_async(password, identity.hashed_password):
        raise InvalidCredentialsError
    return identity


async def update_password(
    db: AsyncSession,
    identity: User,
    *,
    current_password: str,
    new_password: str,
    keep_session_token_hash: str | None = None,
) -> None:
    """Change a password the caller can prove they already hold.

    The port, unchanged in substance: no stored password is a refusal rather
    than an opening, a wrong current password is its own 400, and re-submitting
    the same password is refused rather than silently rehashed.
    """
    if identity.hashed_password is None:
        raise PasswordNotSetError
    if not await verify_password_async(current_password, identity.hashed_password):
        raise CurrentPasswordIncorrectError
    if current_password == new_password:
        raise UnmodifiedPasswordError
    await set_password(
        db,
        identity,
        new_password=new_password,
        keep_session_token_hash=keep_session_token_hash,
    )


async def set_password(
    db: AsyncSession,
    identity: User,
    *,
    new_password: str,
    email: str | None = None,
    keep_session_token_hash: str | None = None,
) -> None:
    """Set a password without proof of the previous one, and commit.

    Reachable only by a caller who has already proved deployment authority: the
    master key in a header, or a session on a deployment where no password has
    been set yet, which is a session the master key must have minted. The route
    owns that gate (`gateway.api.routes.auth_password`); this function does not
    re-derive it, and must not be called from anywhere that has not applied it.

    ``email`` claims a sign-in address for an identity that has none, which is
    the state first boot leaves the operator in. It is refused if another
    identity already holds it, because the column is unique and the address is
    the handle sign-in matches on.

    ``email_verified_at`` is stamped when this call is the one that makes the
    identity able to sign in at all: a first password, or a newly supplied
    address. Both are reachable only through deployment authority, which is what
    the stamp records, and both are what the module docstring promises #650 will
    find already satisfied. It deliberately covers the identity adopted from an
    existing tenancy (`docs/access-control.md`), which arrives *with* an address,
    so a claim on it supplies no ``email`` and would otherwise leave the column
    NULL and that operator locked out the day #650 turns the gate on. A later
    password change stamps nothing: proving the current password says the caller
    owns the account, not the address.

    Every other session this identity holds is revoked, ``keep_session_token_hash``
    excepted, so a stolen cookie does not outlive the password it was minted
    under. The caller passes its own session's hash to stay signed in; a
    header-authenticated caller passes nothing and every session ends.
    """
    _validate_password(new_password)
    vouches_for_the_address = email is not None or identity.hashed_password is None
    if email is not None:
        identity.email = await _claimable_email(db, identity, email)
    elif identity.email is None:
        raise SignInAddressRequiredError
    if vouches_for_the_address:
        identity.email_verified_at = datetime.now(UTC)
    identity.hashed_password = await hash_password_async(new_password)
    # The write and the revocation are inside the try together, and that is not
    # tidiness: the revocation issues a DELETE, which autoflushes the pending
    # UPDATE first, so a duplicate address raises *there* rather than at the
    # commit below. Wrapping only the commit looks right and catches nothing,
    # which is what the integration race test pins.
    try:
        db.add(identity)
        await revoke_user_dashboard_sessions(db, identity.id, keep_token_hash=keep_session_token_hash)
        await db.commit()
    except IntegrityError as exc:
        # ``_claimable_email`` is a preflight, not a lock: two claims of the same
        # address can both pass it and the unique index decides between them.
        # The loser reports the conflict its preflight would have reported,
        # rather than a 500.
        await db.rollback()
        if email is not None and _is_email_conflict(exc):
            raise EmailAlreadyInUseError(validated_email(email)) from None
        # Any other constraint is a bug here, not a race a caller can act on.
        # Reporting it as "that address is taken" would be a lie that sends
        # someone chasing the wrong thing.
        raise
    await db.refresh(identity)


async def has_password_identity(db: AsyncSession) -> bool:
    """Whether any identity on this deployment can sign in with a password.

    This is what "the deployment has been claimed" means, and it is the switch
    between the two sign-in credentials: while it is False the master key is
    still accepted as a dashboard login, and once it is True that login is
    retired and the master key is an API credential only (otari-ai#1716). Read
    off the identities rather than a settings row, so it cannot disagree with
    whether a password sign-in could actually succeed.
    """
    found = await db.execute(select(col(User.id)).where(col(User.hashed_password).is_not(None)).limit(1))
    return found.first() is not None


def _validate_password(password: str) -> None:
    """Refuse a password bcrypt would reject or that is too short to be one."""
    if len(password) < MIN_PASSWORD_LENGTH:
        raise PasswordPolicyError(f"A password must be at least {MIN_PASSWORD_LENGTH} characters")
    if len(password.encode()) > MAX_PASSWORD_BYTES:
        raise PasswordPolicyError(
            f"A password must be at most {MAX_PASSWORD_BYTES} bytes; "
            "accented and non-Latin characters count for more than one each"
        )


def _is_email_conflict(exc: IntegrityError) -> bool:
    """Whether this integrity error is the unique index on ``user.email``.

    Matched on the constraint the engine names rather than on "an IntegrityError
    happened", so a different violation keeps its own error instead of being
    reported as a taken address.

    Both engines are recognized from the message today: PostgreSQL names the
    index in it (``ix_user_email``) and SQLite names the column
    (``UNIQUE constraint failed: user.email``). The ``constraint_name`` check
    ahead of it is belt and braces rather than the PostgreSQL path: SQLAlchemy's
    asyncpg wrapper does not carry that attribute, so it never fires on this
    stack, and it is kept because a driver that does expose it (psycopg, which
    ``TEST_DATABASE_URL`` may name) is then matched structurally instead of by
    text.
    """
    orig = exc.orig
    constraint = getattr(orig, "constraint_name", None)
    if constraint == _EMAIL_UNIQUE_INDEX:
        return True
    detail = str(orig).lower()
    return _EMAIL_UNIQUE_INDEX in detail or "user.email" in detail


async def _claimable_email(db: AsyncSession, identity: User, email: str) -> str:
    """Normalize an address and refuse one another identity already holds.

    Checked here rather than left to the unique index, which is case-sensitive
    on both engines: two rows differing only in case would both insert and then
    ``get_by_email``, which matches case-insensitively, would have to pick one.
    """
    candidate = validated_email(email)
    holder = await UserRepository(db).get_by_email(candidate)
    if holder is not None and holder.id != identity.id:
        raise EmailAlreadyInUseError(candidate)
    return candidate


__all__ = [
    "authenticate",
    "has_password_identity",
    "set_password",
    "update_password",
]
