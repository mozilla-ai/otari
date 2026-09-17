"""Stored guardrail definitions: validate against the catalog, encrypt, persist.

The write side of the picker `services/guardrail_catalog.py` publishes. A
definition names a guardrail the catalog offers and carries the arguments that
build and call it; this module holds the submitted arguments to what the catalog
says that guardrail accepts, splits the secrets out, and encrypts them.

Every rule is read off the catalog rather than written here. A list of names, of
required arguments or of which ones are credentials could only ever drift from
the picker it exists to accept, and the drift would show up as a form offering a
field the store refuses. So the store is wrong exactly when the catalog is.

There is no in-memory overlay and no refresher, unlike the sibling credential
stores whose shape this otherwise follows. What reads these rows is the runner,
at startup and again after each write, and a definition is what it reads them
as. This module commits its own writes,
the layering the rest of the codebase uses and the one #1127 records those two as
missing.

Encryption and decryption happen only here, and only where the result needs a
plaintext. Nothing below this module sees one and nothing above it does either: a
response carries the names of the stored secrets and never a value, and the log
lines carry names and counts.
"""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Final

from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.exceptions.guardrail_credentials import (
    GuardrailCredentialExistsError,
    MissingGuardrailParameterError,
    UnknownGuardrailError,
    UnknownGuardrailParameterError,
    UnstorableGuardrailParameterError,
)
from gateway.log_config import logger
from gateway.models.guardrails import GuardrailCredential
from gateway.models.secret_fields import REDACTED_VALUE, restore_redacted_values
from gateway.repositories import guardrail_credentials_repository as repository
from gateway.services.guardrail_catalog import (
    BuiltInGuardrailSpec,
    GuardrailParameterSpec,
    builtin_guardrail_spec,
)
from gateway.services.secret_box import (
    SecretBoxUnavailableError,
    SecretDecryptionError,
    decrypt_secret,
    encrypt_secret,
)
from gateway.types.guardrail_definition import GuardrailDefinition


class _Unset:
    """Sentinel type: 'this field was not provided', distinct from an explicit None."""


# A field left at UNSET keeps its stored value. The siblings in
# ``search_tool_store_service`` and ``provider_store_service`` use the same one.
UNSET: Final = _Unset()


def _require_spec(guardrail_name: str) -> BuiltInGuardrailSpec:
    spec = builtin_guardrail_spec(guardrail_name)
    if spec is None:
        raise UnknownGuardrailError(guardrail_name)
    return spec


def _storable_secret_names(parameters: list[GuardrailParameterSpec]) -> list[str]:
    """The credential arguments among ``parameters`` that a row can actually hold.

    Taken from one stage's parameters rather than the whole spec, so the
    alternatives a refusal offers belong to the stage the refused argument was
    sent for.
    """
    return [parameter.name for parameter in parameters if parameter.secret and parameter.storable]


def _check_stage(
    spec: BuiltInGuardrailSpec,
    stage: str,
    parameters: list[GuardrailParameterSpec],
    submitted: dict[str, Any],
) -> None:
    """Every submitted name is one the guardrail takes, and one a row can hold."""
    by_name = {parameter.name: parameter for parameter in parameters}
    for name in submitted:
        parameter = by_name.get(name)
        if parameter is None:
            raise UnknownGuardrailParameterError(spec.guardrail_name, name, stage)
        if not parameter.storable:
            alternatives = _storable_secret_names(parameters)
            raise UnstorableGuardrailParameterError(
                spec.guardrail_name,
                name,
                " and ".join(f"'{other}'" for other in alternatives) if alternatives else "a value it can store",
            )


def _supplied(create_kwargs: dict[str, Any], name: str) -> bool:
    """Whether ``name`` carries a value, treating an explicit null as absent.

    A key alone is not a value: ``{"api_key": null}`` would otherwise satisfy a
    required argument and store a definition that cannot build. Zero and false
    are values, so the test is against ``None`` rather than falsiness.
    """
    return create_kwargs.get(name) is not None


def _check_required(spec: BuiltInGuardrailSpec, create_kwargs: dict[str, Any]) -> None:
    """A required constructor argument is supplied, unless something else supplies it.

    The carve-out is ``env_var``. ``GuardrailParameterSpec.required`` folds in
    upstream's effectively-required flag, so an argument a deployment sets
    through the environment still reads required; demanding it here would refuse
    a row that would have worked. Whether the variable is actually set is not
    consulted, because that is a property of the process that builds the
    guardrail rather than of the one writing the row.
    """
    for parameter in spec.create_parameters:
        if not parameter.required or parameter.env_var or _supplied(create_kwargs, parameter.name):
            continue
        raise MissingGuardrailParameterError(
            spec.guardrail_name, f"'{parameter.name}' is required and nothing else supplies it."
        )


def _check_requirement_groups(spec: BuiltInGuardrailSpec, create_kwargs: dict[str, Any]) -> None:
    """One-of constraints no single argument's required flag can express.

    Skipped where the group names environment variables, for the reason
    :func:`_check_required` gives. Every group the catalog carries today names
    one, so this refuses nothing yet; it is here so a future group that names
    none is caught at write time rather than at build time.
    """
    for group in spec.requirement_groups:
        if group.env_vars or any(_supplied(create_kwargs, name) for name in group.parameters):
            continue
        raise MissingGuardrailParameterError(spec.guardrail_name, group.description)


def validate_guardrail_kwargs(
    guardrail_name: str,
    *,
    create_kwargs: dict[str, Any],
    validate_kwargs: dict[str, Any],
) -> None:
    """Hold a definition to what the catalog says its guardrail accepts.

    Raises the matching :mod:`gateway.exceptions.guardrail_credentials` member,
    each of which the route answers with a 400 naming the rule that was broken.
    """
    spec = _require_spec(guardrail_name)
    _check_stage(spec, "create", spec.create_parameters, create_kwargs)
    _check_stage(spec, "validate", spec.validate_parameters, validate_kwargs)
    _check_required(spec, create_kwargs)
    _check_requirement_groups(spec, create_kwargs)


def split_create_kwargs(guardrail_name: str, submitted: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Divide constructor arguments into the plain half and the secret half.

    By the catalog's ``secret`` flag alone, so a guardrail upstream adds splits
    correctly without an edit here. Validate first: an argument no spec declares
    has no flag to be classified by, and would otherwise land in the plain half.
    """
    spec = _require_spec(guardrail_name)
    secret_names = {parameter.name for parameter in spec.create_parameters if parameter.secret}
    plain = {name: value for name, value in submitted.items() if name not in secret_names}
    secrets = {name: value for name, value in submitted.items() if name in secret_names}
    return plain, secrets


def decrypt_create_secrets(row: GuardrailCredential) -> dict[str, Any]:
    """The row's secret arguments in clear, or ``{}`` when it stores none.

    Raises ``SecretBoxUnavailableError`` or ``SecretDecryptionError`` when a
    stored map cannot be read; the caller decides whether that is fatal.
    """
    if not row.encrypted_create_secrets:
        return {}
    loaded = json.loads(decrypt_secret(row.encrypted_create_secrets))
    return loaded if isinstance(loaded, dict) else {}


def definition_from_row(row: GuardrailCredential) -> GuardrailDefinition:
    """The row as the runner takes it, with its secrets put back where they came from.

    The two halves are one constructor argument map again. They were split on the
    way in only so the credentials could be encrypted, and the guardrail being
    built knows nothing about that split.

    Raises what :func:`decrypt_create_secrets` raises. Refusing is the point: a
    definition built from the plain half alone would be a client with no API key,
    which fails later and less clearly.
    """
    return GuardrailDefinition(
        guardrail_name=row.guardrail_name,
        create_kwargs={**row.create_kwargs, **decrypt_create_secrets(row)},
        validate_kwargs=dict(row.validate_kwargs),
    )


def stored_secret_names(row: GuardrailCredential) -> tuple[frozenset[str], bool]:
    """Which secrets the row holds, and whether they could be read at all.

    A row encrypted under a key the deployment no longer has reports no names
    and ``False``, so a listing shows it as needing attention instead of
    failing. That is the ``_is_decryptable`` posture of the provider store, with
    the names the read needed anyway.
    """
    try:
        return frozenset(decrypt_create_secrets(row)), True
    except (SecretBoxUnavailableError, SecretDecryptionError, ValueError):
        return frozenset(), False


def _encrypted(secrets: dict[str, Any]) -> str | None:
    """One encrypted string for the whole map, or None when there is nothing to hold.

    ``sort_keys`` so the ciphertext of an unchanged map does not depend on the
    order the caller sent its arguments in.
    """
    if not secrets:
        return None
    return encrypt_secret(json.dumps(secrets, sort_keys=True))


@asynccontextmanager
async def _write(db: AsyncSession) -> AsyncIterator[None]:
    """Roll back whatever the block staged when the database refuses it.

    Wraps the whole write rather than the commit alone, because a repository
    call flushes: the insert can fail before a commit is ever reached, and the
    session has to be left usable either way. The rollback belongs here because
    this is the layer that owns the transaction; the route's part is turning
    what comes out into a status, the way the sibling stores do.
    """
    try:
        yield
    except SQLAlchemyError:
        await db.rollback()
        raise


async def list_guardrail_credentials(db: AsyncSession) -> list[GuardrailCredential]:
    """Every stored guardrail, ordered by name."""
    return await repository.list_guardrail_credentials(db)


async def get_guardrail_credential(db: AsyncSession, name: str) -> GuardrailCredential | None:
    """The stored guardrail called ``name``, or ``None``."""
    return await repository.get_guardrail_credential(db, name)


async def get_guardrail_credential_for_update(db: AsyncSession, name: str) -> GuardrailCredential | None:
    """The stored guardrail called ``name``, locked for the write that follows."""
    return await repository.get_guardrail_credential_for_update(db, name)


async def create_guardrail_credential(
    db: AsyncSession,
    *,
    name: str,
    guardrail_name: str,
    create_kwargs: dict[str, Any],
    validate_kwargs: dict[str, Any],
    enabled: bool = True,
) -> GuardrailCredential:
    """Store a new guardrail definition.

    Validation and encryption both run before anything is staged, so a refused
    definition and a deployment with no ``OTARI_SECRET_KEY`` each leave the
    session untouched.
    """
    validate_guardrail_kwargs(guardrail_name, create_kwargs=create_kwargs, validate_kwargs=validate_kwargs)
    plain, secrets = split_create_kwargs(guardrail_name, create_kwargs)
    row = GuardrailCredential(
        name=name,
        guardrail_name=guardrail_name,
        create_kwargs=plain,
        validate_kwargs=dict(validate_kwargs),
        encrypted_create_secrets=_encrypted(secrets),
        enabled=enabled,
    )

    try:
        async with _write(db):
            await repository.add_guardrail_credential(db, row)
            await db.commit()
    except IntegrityError:
        # The route's pre-check races the insert; the primary key is what
        # actually decides, as it does in the sibling credential stores. The
        # flush raises it before the commit, which is inside the block above, so
        # the session is already clean by the time this runs.
        raise GuardrailCredentialExistsError(name) from None

    await db.refresh(row)
    logger.info("Stored guardrail '%s' (%s) with %d secret(s)", name, guardrail_name, len(secrets))
    return row


def _merged_create_kwargs(
    row: GuardrailCredential,
    target_guardrail: str,
    create_kwargs: dict[str, Any] | _Unset,
) -> dict[str, Any] | None:
    """The constructor arguments to store, or ``None`` to keep the stored ones.

    The stored map is decrypted only where the answer depends on it. A
    replacement carrying no ``***`` stands on its own, and an update that moves
    neither the arguments nor the guardrail leaves the ciphertext untouched. So a
    deployment whose key no longer reads a row can still disable it, edit its
    per-call arguments, and repair it by sending the credentials again, none of
    which needs the value it cannot read.

    Raises ``SecretDecryptionError`` in the two cases that do need it: a ``***``
    has nothing to stand for without the stored value, and re-splitting under a
    new guardrail needs the whole definition, where dropping the half that will
    not decrypt would turn a key problem into silent data loss.
    """
    if isinstance(create_kwargs, _Unset):
        if target_guardrail == row.guardrail_name:
            return None
        return {**row.create_kwargs, **decrypt_create_secrets(row)}
    if REDACTED_VALUE not in create_kwargs.values():
        return dict(create_kwargs)
    return restore_redacted_values(create_kwargs, decrypt_create_secrets(row)) or {}


async def update_guardrail_credential(
    db: AsyncSession,
    *,
    row: GuardrailCredential,
    guardrail_name: str | _Unset = UNSET,
    create_kwargs: dict[str, Any] | _Unset = UNSET,
    validate_kwargs: dict[str, Any] | _Unset = UNSET,
    enabled: bool | _Unset = UNSET,
) -> GuardrailCredential:
    """Update a stored definition. A field left at ``UNSET`` keeps its stored value.

    ``create_kwargs`` when sent replaces the whole map, against the stored
    arguments merged back together: a value of ``***`` keeps the stored secret,
    a new value rotates it, and a secret the caller left out is cleared. That is
    what makes an editor that loads a row, changes the endpoint and submits the
    whole object safe, since it was never shown the key it is echoing back.

    Changing ``guardrail_name`` without sending ``create_kwargs`` re-splits the
    stored arguments under the new class, so the plain and secret halves can
    never be left classified by a guardrail the row no longer names.
    """
    target_guardrail = row.guardrail_name if isinstance(guardrail_name, _Unset) else guardrail_name
    spec = _require_spec(target_guardrail)

    if isinstance(validate_kwargs, _Unset):
        target_validate = dict(row.validate_kwargs or {})
    else:
        target_validate = restore_redacted_values(validate_kwargs, row.validate_kwargs) or {}
    _check_stage(spec, "validate", spec.validate_parameters, target_validate)

    merged = _merged_create_kwargs(row, target_guardrail, create_kwargs)
    secrets: dict[str, Any] | None = None
    if merged is not None:
        _check_stage(spec, "create", spec.create_parameters, merged)
        _check_required(spec, merged)
        _check_requirement_groups(spec, merged)
        row.create_kwargs, secrets = split_create_kwargs(target_guardrail, merged)
        row.encrypted_create_secrets = _encrypted(secrets)

    row.guardrail_name = target_guardrail
    row.validate_kwargs = target_validate
    if not isinstance(enabled, _Unset):
        row.enabled = enabled

    async with _write(db):
        await db.commit()
    await db.refresh(row)
    logger.info(
        "Updated stored guardrail '%s' (%s), %s",
        row.name,
        target_guardrail,
        "credentials untouched" if secrets is None else f"{len(secrets)} secret(s)",
    )
    return row


async def delete_guardrail_credential(db: AsyncSession, name: str) -> bool:
    """Delete a stored guardrail. Returns whether it existed."""
    row = await repository.get_guardrail_credential(db, name)
    if row is None:
        return False
    async with _write(db):
        await repository.delete_guardrail_credential(db, row)
        await db.commit()
    logger.info("Deleted stored guardrail '%s'", name)
    return True


async def reencrypt_guardrail_credentials(db: AsyncSession) -> tuple[int, int]:
    """Re-encrypt stored secret maps with the current primary OTARI_SECRET_KEY.

    Returns ``(reencrypted, unreadable)``. The guardrail half of the rotation
    procedure; run it alongside the provider and search-tool endpoints. A map
    that cannot be decrypted with the configured key set is left untouched and
    counted, so the operator can recover it by re-entering that guardrail's
    credentials.
    """
    reencrypted = 0
    unreadable = 0
    async with _write(db):
        for row in await repository.list_encrypted_guardrail_credentials(db):
            if row.encrypted_create_secrets is None:
                continue
            try:
                plaintext = decrypt_secret(row.encrypted_create_secrets)
            except SecretDecryptionError:
                unreadable += 1
                continue
            row.encrypted_create_secrets = encrypt_secret(plaintext)
            reencrypted += 1
        await db.commit()
    return reencrypted, unreadable
