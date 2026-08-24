"""Passkey registration, sign-in, and the credential list a person manages.

WebAuthn in one paragraph, because the rest of this file assumes it. A passkey
is a key pair whose private half never leaves the authenticator (a laptop's
secure enclave, a phone, a hardware key). Registering one has the authenticator
generate the pair and hand back the public half; signing in has it sign a
server-chosen challenge with the private half. Nothing this deployment stores
can be used to sign in as anybody, which is the property that makes the table
worth having over ``user.hashed_password``.

**Verification is py_webauthn's, not ours.** Checking an attestation object and
an assertion signature means CBOR decoding, COSE key parsing, the flags byte,
the RP-ID hash, the counter and the client-data JSON, and each of those has a
way to be checked that looks right and is not. The library does that; this
module owns the parts that are this deployment's: which relying party it is,
where challenges live, which credentials belong to whom, and what a failure
tells the caller.

**Every ceremony is two calls, and the server owns what joins them.** The
options call issues a challenge and records it; the verify call spends it. The
challenge row is deleted as it is read, so an assertion replayed against a
captured challenge matches nothing. See `models.tenancy.WebAuthnChallenge` for
why that record is a table rather than a cookie or a dictionary.

**Sign-in here is usernameless.** ``begin_authentication`` publishes no
``allowCredentials`` list, so the browser offers whichever passkey it holds for
this relying party and the assertion names the credential that answered. Two
consequences, both deliberate: an unauthenticated caller cannot ask this
deployment which passkeys an address holds (the list would be exactly that
oracle), and a passkey signs a person in without them typing anything, which is
the whole affordance. The cost is that a credential registered on an
authenticator that cannot store discoverable credentials will not be offered;
registration asks for a discoverable one, so that is a property of old hardware
rather than of a normal passkey.

**A passkey never bootstraps an identity.** Registration is performed from
inside a session, so it only ever adds a credential to somebody who is already
here. There is no passkey signup: a deployment is claimed with the master key
and a password (`user_service`), and a passkey is something an identity adds
afterwards.
"""

import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col
from webauthn import (
    generate_authentication_options,
    generate_registration_options,
    options_to_json,
    verify_authentication_response,
    verify_registration_response,
)
from webauthn.helpers import base64url_to_bytes, bytes_to_base64url, parse_client_data_json
from webauthn.helpers.exceptions import (
    InvalidAuthenticationResponse,
    InvalidJSONStructure,
    InvalidRegistrationResponse,
)
from webauthn.helpers.structs import (
    AuthenticatorSelectionCriteria,
    PublicKeyCredentialDescriptor,
    ResidentKeyRequirement,
    UserVerificationRequirement,
)

from gateway.core.config import GatewayConfig, RelyingParty
from gateway.log_config import logger
from gateway.models.tenancy import (
    WEBAUTHN_CHALLENGE_TTL_SECONDS,
    User,
    WebAuthnChallenge,
    WebAuthnCredential,
    WebAuthnCredentialPublic,
)
from gateway.services.tenancy.errors import (
    PasskeyAlreadyRegisteredError,
    PasskeyCeremonyError,
    PasskeyLimitReachedError,
    PasskeyNameTakenError,
    PasskeyNotFoundError,
    PasskeySignInFailedError,
    PasskeysNotConfiguredError,
)

# How many passkeys one identity may hold. Generous rather than tight: the
# reason to have more than one is that losing the only one is a lockout, so a
# person is *expected* to register their laptop, their phone and a hardware key.
# It exists because the table is written by an authenticated caller in a loop
# they control, and an unbounded list is a list somebody eventually fills.
MAX_PASSKEYS_PER_IDENTITY = 20

# What a passkey is called when its owner does not say. Deliberately not derived
# from the user agent: a name is what distinguishes one row from another, and
# every passkey registered from the same browser would get the same string.
DEFAULT_PASSKEY_NAME = "Passkey"


def require_relying_party(config: GatewayConfig) -> RelyingParty:
    """The configured relying party, or refuse the ceremony.

    Every entry point starts here, so "this deployment cannot do passkeys" is
    answered once, by a 503 naming the setting, rather than by a ceremony that
    starts and then fails inside a browser with nothing to correlate it with.
    """
    relying_party = config.webauthn_relying_party
    if relying_party is None:
        raise PasskeysNotConfiguredError
    return relying_party


async def _issue_challenge(
    db: AsyncSession, challenge: bytes, *, ceremony: str, user_id: uuid.UUID | None
) -> None:
    """Record a challenge so the matching verify call can spend it.

    Expired rows are swept here rather than by a background task, exactly as
    ``create_dashboard_session`` sweeps expired sessions: this table is written
    once per ceremony, so the sweep runs as often as it needs to and never on a
    read path. The caller owns the transaction and commits.
    """
    now = datetime.now(UTC)
    await db.execute(delete(WebAuthnChallenge).where(col(WebAuthnChallenge.expires_at) < now))
    db.add(
        WebAuthnChallenge(
            challenge=bytes_to_base64url(challenge),
            ceremony=ceremony,
            user_id=user_id,
            expires_at=now + timedelta(seconds=WEBAUTHN_CHALLENGE_TTL_SECONDS),
        )
    )


async def _spend_challenge(db: AsyncSession, challenge: bytes, *, ceremony: str) -> uuid.UUID | None:
    """Consume a challenge, or refuse: unknown, expired, or the wrong ceremony.

    Returns the identity the challenge was issued to, which is null for an
    authentication ceremony.

    **One conditional DELETE, not a SELECT and then a delete.** Single use is
    the whole security property of a challenge, and a read-then-write cannot
    provide it: two requests replaying one assertion would both find the row,
    both verify, and both be handed a session. Deleting by primary key and
    reading what came back means exactly one of them gets the row, because the
    loser blocks on the winner's lock and then matches nothing. This is the same
    no-check-then-act rule the budget reservation follows, applied to a nonce.

    The delete is staged on the caller's transaction, so a verification that
    fails afterwards rolls it back and the challenge survives for the retry the
    caller is about to make; only a ceremony that completes actually retires one.

    ``ceremony`` is compared after the fact rather than added to the WHERE
    clause. A challenge answered with the wrong ceremony is spent either way:
    it is a challenge this server issued, somebody has just used it, and leaving
    it live so it can be tried again the other way is the opposite of what
    single-use means.
    """
    encoded = bytes_to_base64url(challenge)
    row = (
        await db.execute(
            delete(WebAuthnChallenge)
            .where(col(WebAuthnChallenge.challenge) == encoded)
            .returning(
                col(WebAuthnChallenge.ceremony),
                col(WebAuthnChallenge.user_id),
                col(WebAuthnChallenge.expires_at),
            )
        )
    ).first()
    if row is None:
        raise PasskeyCeremonyError
    spent_ceremony: str = row[0]
    issued_to: uuid.UUID | None = row[1]
    expires_at: datetime = row[2]
    if spent_ceremony != ceremony:
        raise PasskeyCeremonyError
    # SQLite hands the timestamp back naive, as the rest of this schema's
    # readers already handle (``dashboard_session_service._as_utc``).
    if (expires_at if expires_at.tzinfo is not None else expires_at.replace(tzinfo=UTC)) < datetime.now(UTC):
        raise PasskeyCeremonyError
    return issued_to


def _challenge_of(response: dict[str, Any]) -> bytes:
    """The challenge a ceremony response is answering, read off its clientDataJSON.

    Taken from the response so the stored row can be found, and then handed to
    the library as ``expected_challenge`` so the library compares it against the
    very same field. That looks circular and is not: the value's *authenticity*
    comes from having matched a row this server issued and from the signature
    over the client data, neither of which this function claims to establish.
    Its only job is the lookup key.
    """
    try:
        client_data = base64url_to_bytes(response["response"]["clientDataJSON"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PasskeyCeremonyError from exc
    # The library's own parser, so a malformed or truncated client-data blob is
    # rejected the same way here as it is during verification.
    try:
        return parse_client_data_json(client_data).challenge
    except (InvalidJSONStructure, ValueError) as exc:
        raise PasskeyCeremonyError from exc


async def _credentials_for(db: AsyncSession, user_id: uuid.UUID, rp_id: str) -> list[WebAuthnCredential]:
    """One identity's passkeys under the *current* relying-party ID.

    Filtered by ``rp_id`` rather than listed wholesale: a row registered under a
    previous ID cannot be asserted now, so offering it to a browser produces a
    passkey prompt that can only fail. See `models.tenancy.WebAuthnCredential`.
    """
    result = await db.execute(
        select(WebAuthnCredential)
        .where(col(WebAuthnCredential.user_id) == user_id, col(WebAuthnCredential.rp_id) == rp_id)
        .order_by(col(WebAuthnCredential.created_at))
    )
    return list(result.scalars().all())


async def begin_registration(db: AsyncSession, config: GatewayConfig, identity: User) -> dict[str, Any]:
    """Issue registration options for a signed-in identity to hand to its browser.

    ``exclude_credentials`` names what this identity already holds, so an
    authenticator that is already registered declines up front with "you already
    have a passkey here" rather than minting a second credential for the same
    device. It is safe to publish: the caller is signed in and it is their own
    list.

    A discoverable (resident) credential is *required*, not preferred, because
    sign-in here is usernameless and a non-discoverable credential would
    register successfully and then never be offered at sign-in, which is the
    worst of the two outcomes. User verification is preferred rather than
    required: a hardware key with no PIN is still a second factor worth having,
    and requiring it would turn those authenticators away.

    The caller commits.
    """
    relying_party = require_relying_party(config)
    existing = await _credentials_for(db, identity.id, relying_party.rp_id)
    # Refused here as well as after verification. The check on the way back is
    # the one that actually holds (two ceremonies can start at once), but making
    # somebody touch their security key and *then* be told no is the worse way
    # to say it.
    if len(existing) >= MAX_PASSKEYS_PER_IDENTITY:
        raise PasskeyLimitReachedError(MAX_PASSKEYS_PER_IDENTITY)
    options = generate_registration_options(
        rp_id=relying_party.rp_id,
        rp_name=relying_party.name,
        # The identity's UUID, not its email: this is an opaque handle the
        # authenticator files the credential under, and an address that later
        # changes would leave the authenticator showing a stale one forever.
        user_id=identity.id.bytes,
        # What the authenticator shows the person while they choose. An
        # address-less bootstrap operator falls back to a label rather than an
        # empty prompt.
        user_name=identity.email or "operator",
        user_display_name=identity.full_name or identity.email or "otari operator",
        exclude_credentials=[
            PublicKeyCredentialDescriptor(id=base64url_to_bytes(credential.credential_id))
            for credential in existing
        ],
        authenticator_selection=AuthenticatorSelectionCriteria(
            resident_key=ResidentKeyRequirement.REQUIRED,
            user_verification=UserVerificationRequirement.PREFERRED,
        ),
    )
    await _issue_challenge(db, options.challenge, ceremony="registration", user_id=identity.id)
    parsed: dict[str, Any] = json.loads(options_to_json(options))
    return parsed


async def finish_registration(
    db: AsyncSession,
    config: GatewayConfig,
    identity: User,
    response: dict[str, Any],
    name: str | None,
) -> WebAuthnCredential:
    """Verify a registration response and store the passkey it produced.

    The challenge must be one *this identity* was issued: a registration
    challenge names its identity, and a response carrying somebody else's is
    refused here rather than silently attributing their ceremony to this caller.

    The caller commits.
    """
    relying_party = require_relying_party(config)
    challenge = _challenge_of(response)
    issued_to = await _spend_challenge(db, challenge, ceremony="registration")
    if issued_to != identity.id:
        raise PasskeyCeremonyError

    try:
        verified = verify_registration_response(
            credential=response,
            expected_challenge=challenge,
            expected_rp_id=relying_party.rp_id,
            expected_origin=list(relying_party.origins),
        )
    except (InvalidRegistrationResponse, InvalidJSONStructure, ValueError) as exc:
        # The library's reason is worth having when an operator reports "my
        # passkey will not register", and it names no secret: it describes the
        # shape of a payload the caller sent. The caller still gets one sentence.
        logger.info("Passkey registration did not verify for identity %s: %s", identity.id, exc)
        raise PasskeyCeremonyError from None

    existing = await _credentials_for(db, identity.id, relying_party.rp_id)
    if len(existing) >= MAX_PASSKEYS_PER_IDENTITY:
        raise PasskeyLimitReachedError(MAX_PASSKEYS_PER_IDENTITY)

    credential = WebAuthnCredential(
        user_id=identity.id,
        credential_id=bytes_to_base64url(verified.credential_id),
        public_key=bytes_to_base64url(verified.credential_public_key),
        rp_id=relying_party.rp_id,
        sign_count=verified.sign_count,
        transports=_transports_of(response),
        backed_up=verified.credential_backed_up,
        aaguid=verified.aaguid,
        name=_unique_name(name, existing),
    )
    db.add(credential)
    try:
        await db.flush()
    except IntegrityError as exc:
        await db.rollback()
        # ``exclude_credentials`` already tells a cooperating authenticator not
        # to answer twice, so reaching this means either a client that ignored
        # it or the same authenticator registered against another identity. The
        # unique index is what actually decides it, because two concurrent
        # ceremonies would both pass a SELECT.
        logger.info("Passkey registration collided with an existing credential: %s", exc)
        raise PasskeyAlreadyRegisteredError from None
    return credential


def _transports_of(response: dict[str, Any]) -> list[str]:
    """The transports the browser reported, filtered to strings.

    Advisory data used to hint the next ceremony ("this one is on a USB key"),
    so an authenticator that reports nothing, or reports something unexpected,
    costs a hint rather than a registration.
    """
    raw = response.get("response", {}).get("transports") if isinstance(response.get("response"), dict) else None
    if not isinstance(raw, list):
        return []
    # Bounded in both directions, because this is the one field on the row that
    # is stored as the client sent it rather than as the verifier derived it.
    # The vocabulary's longest member is "smart-card", so 32 is already generous
    # and an authenticator inventing a longer name loses a hint rather than a
    # registration.
    return [item for item in raw if isinstance(item, str) and len(item) <= 32][:8]


def _unique_name(requested: str | None, existing: list[WebAuthnCredential]) -> str:
    """A label for a new passkey that does not collide with this identity's others.

    The unique constraint on ``(user_id, name)`` is what keeps a list from
    showing the same name twice, and a person registering three passkeys without
    naming any of them should not have to. A requested name that collides is
    refused (they chose it); a defaulted one is numbered.
    """
    taken = {credential.name for credential in existing}
    if requested is not None:
        cleaned = requested.strip()
        if cleaned:
            if cleaned in taken:
                raise PasskeyNameTakenError(cleaned)
            return cleaned
    if DEFAULT_PASSKEY_NAME not in taken:
        return DEFAULT_PASSKEY_NAME
    for suffix in range(2, MAX_PASSKEYS_PER_IDENTITY + 2):
        candidate = f"{DEFAULT_PASSKEY_NAME} {suffix}"
        if candidate not in taken:
            return candidate
    raise PasskeyNameTakenError(DEFAULT_PASSKEY_NAME)


async def begin_authentication(db: AsyncSession, config: GatewayConfig) -> dict[str, Any]:
    """Issue sign-in options, naming no credentials and no identity.

    Unauthenticated, and the empty ``allowCredentials`` is the point: publishing
    a list would let anyone ask which passkeys this deployment holds, and the
    browser does not need one to offer a discoverable credential. The challenge
    row therefore carries no ``user_id``; who is signing in is learned from the
    assertion.

    The caller commits.
    """
    relying_party = require_relying_party(config)
    options = generate_authentication_options(
        rp_id=relying_party.rp_id,
        user_verification=UserVerificationRequirement.PREFERRED,
    )
    await _issue_challenge(db, options.challenge, ceremony="authentication", user_id=None)
    parsed: dict[str, Any] = json.loads(options_to_json(options))
    return parsed


async def finish_authentication(db: AsyncSession, config: GatewayConfig, response: dict[str, Any]) -> User:
    """Verify an assertion and return the identity whose passkey signed it.

    Four things have to hold, and the order matters: the challenge is one this
    server issued for *this* ceremony, the credential is one it stores under the
    current relying-party ID, the signature checks out against that credential's
    public key, and the identity behind it is still active. A deactivated
    identity is refused here for the reason ``resolve_dashboard_session``
    refuses one: deactivating somebody has to end their access now, and a
    passkey they still hold is exactly the access being ended.

    The caller commits, which is what retires the challenge; a refusal rolls
    back and leaves it for the retry.
    """
    relying_party = require_relying_party(config)
    # Every way the challenge half can fail is re-raised as the sign-in failure,
    # not as ``PasskeyCeremonyError``. Two reasons, and the first is the one that
    # matters: this path is reachable unauthenticated, and answering a spent or
    # unknown challenge with a different status from a bad signature tells a
    # caller which half they got wrong. The second is that a 400 here would be
    # wrong anyway: nothing is malformed about a replayed assertion, the caller
    # is simply not authenticated.
    try:
        challenge = _challenge_of(response)
        await _spend_challenge(db, challenge, ceremony="authentication")
    except PasskeyCeremonyError:
        raise PasskeySignInFailedError from None

    raw_id = response.get("rawId") or response.get("id")
    if not isinstance(raw_id, str):
        raise PasskeySignInFailedError
    row = (
        await db.execute(
            select(WebAuthnCredential, User)
            .join(User, col(User.id) == WebAuthnCredential.user_id)
            .where(
                col(WebAuthnCredential.credential_id) == raw_id,
                col(WebAuthnCredential.rp_id) == relying_party.rp_id,
            )
        )
    ).first()
    if row is None:
        raise PasskeySignInFailedError
    # Annotated rather than tuple-unpacked: a two-entity ``Row`` types as
    # ``Any`` under mypy strict, following ``resolve_dashboard_session``.
    credential: WebAuthnCredential = row[0]
    identity: User = row[1]
    if not identity.is_active:
        raise PasskeySignInFailedError

    try:
        verified = verify_authentication_response(
            credential=response,
            expected_challenge=challenge,
            expected_rp_id=relying_party.rp_id,
            expected_origin=list(relying_party.origins),
            credential_public_key=base64url_to_bytes(credential.public_key),
            credential_current_sign_count=credential.sign_count,
        )
    except (InvalidAuthenticationResponse, InvalidJSONStructure, ValueError) as exc:
        logger.info("Passkey assertion did not verify for credential %s: %s", credential.id, exc)
        raise PasskeySignInFailedError from None

    # The library refuses a counter that went *backwards*, which is the clone
    # signal. Storing the new one is what keeps that check meaningful on the
    # next assertion. An authenticator that keeps no counter reports 0 forever
    # and is neither refused nor flagged: that is the normal state of a synced
    # platform passkey, and treating it as suspicious would refuse most of them.
    credential.sign_count = verified.new_sign_count
    credential.backed_up = verified.credential_backed_up
    credential.last_used_at = datetime.now(UTC)
    db.add(credential)
    return identity


async def list_credentials(db: AsyncSession, user_id: uuid.UUID) -> list[WebAuthnCredential]:
    """Every passkey this identity holds, newest last, for the settings page.

    **Not filtered by relying-party ID, unlike the ceremonies.** An earlier
    version hid the rows the current configuration cannot assert, which was the
    wrong call twice over: a person whose deployment moved its relying-party ID
    saw an empty list with no explanation, and the rows they could no longer see
    were exactly the ones they needed the id of in order to delete. Each row
    says whether it is still usable instead (``WebAuthnCredentialPublic``), so
    the page can show an orphan and offer the one action left for it.

    Takes no config for the same reason the route no longer requires one: a
    deployment that has stopped being configured for passkeys still has to let
    somebody clean up after it.
    """
    result = await db.execute(
        select(WebAuthnCredential)
        .where(col(WebAuthnCredential.user_id) == user_id)
        .order_by(col(WebAuthnCredential.created_at))
    )
    return list(result.scalars().all())


async def rename_credential(
    db: AsyncSession, user_id: uuid.UUID, credential_id: uuid.UUID, name: str
) -> WebAuthnCredential:
    """Relabel one of this identity's passkeys. The caller commits.

    Scoped to ``user_id`` in the same statement that finds the row, not checked
    afterwards: a passkey belonging to somebody else must be indistinguishable
    from one that does not exist, which is the rule ``TenancyNotFoundError``
    states.
    """
    credential = await _owned_credential(db, user_id, credential_id)
    cleaned = name.strip()
    if not cleaned:
        raise PasskeyNameTakenError(name)
    if cleaned == credential.name:
        return credential
    clash = (
        await db.execute(
            select(func.count())
            .select_from(WebAuthnCredential)
            .where(
                col(WebAuthnCredential.user_id) == user_id,
                col(WebAuthnCredential.name) == cleaned,
                col(WebAuthnCredential.id) != credential_id,
            )
        )
    ).scalar_one()
    if clash:
        raise PasskeyNameTakenError(cleaned)
    credential.name = cleaned
    db.add(credential)
    return credential


async def delete_credential(db: AsyncSession, user_id: uuid.UUID, credential_id: uuid.UUID) -> None:
    """Remove one of this identity's passkeys. The caller commits.

    Deleting the last one is allowed. It is not a lockout on its own: this
    deployment's steady-state login is still an email and a password, and a
    passkey is something added beside it. Refusing here would instead strand
    somebody whose authenticator is lost, which is precisely when this button is
    pressed.
    """
    credential = await _owned_credential(db, user_id, credential_id)
    await db.delete(credential)


async def _owned_credential(
    db: AsyncSession, user_id: uuid.UUID, credential_id: uuid.UUID
) -> WebAuthnCredential:
    """One passkey, if it is this identity's; a 404 either way if it is not."""
    credential = (
        await db.execute(
            select(WebAuthnCredential).where(
                col(WebAuthnCredential.id) == credential_id,
                col(WebAuthnCredential.user_id) == user_id,
            )
        )
    ).scalar_one_or_none()
    if credential is None:
        raise PasskeyNotFoundError(credential_id)
    return credential


async def has_any_credential(db: AsyncSession, config: GatewayConfig) -> bool:
    """Whether any passkey on this deployment could answer a sign-in right now.

    Read by ``GET /v1/bootstrap`` to decide whether to publish ``passkey`` as a
    sign-in method. A deployment that is configured for passkeys but holds none
    would otherwise show a sign-in button whose only possible outcome is "no
    passkey found", which is the shape the login page already avoids for the
    master key.

    Scoped to the current relying-party ID for the same reason the list is: rows
    under a previous one cannot answer.
    """
    relying_party = config.webauthn_relying_party
    if relying_party is None:
        return False
    found = (
        await db.execute(
            select(col(WebAuthnCredential.id)).where(col(WebAuthnCredential.rp_id) == relying_party.rp_id).limit(1)
        )
    ).first()
    return found is not None


def to_public(credential: WebAuthnCredential, *, relying_party_id: str | None) -> WebAuthnCredentialPublic:
    """The wire shape of a passkey. Carries no key material; see the model.

    ``relying_party_id`` is this deployment's current one, or None when it has
    none, and is what decides ``is_usable``. Passed in rather than read from a
    config here so this stays a pure projection.
    """
    return WebAuthnCredentialPublic(
        id=credential.id,
        name=credential.name,
        credential_id=credential.credential_id,
        rp_id=credential.rp_id,
        is_usable=relying_party_id is not None and credential.rp_id == relying_party_id,
        transports=list(credential.transports),
        backed_up=credential.backed_up,
        created_at=credential.created_at,
        last_used_at=credential.last_used_at,
    )
