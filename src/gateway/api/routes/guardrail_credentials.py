"""Stored guardrail definitions for the dashboard (``/api/v1/guardrail-credentials``).

The write target for the picker at ``GET /api/v1/tool-settings/guardrails/catalog``.
That endpoint lists the guardrails this gateway can run itself and, for each
one, the constructor and per-call arguments it takes; these endpoints store one
of those choices together with the values filled in beside it, so a guardrail is
defined in Otari rather than in a sidecar's YAML.

Nothing on the request path reads these rows yet.

Deliberately the same shape as ``/api/v1/search-tools`` and
``/api/v1/provider-credentials``: rows keyed by name, the credentials encrypted
at rest and never returned, a tri-state PATCH, an optimistic-concurrency check
under a row lock, and a re-encryption endpoint for ``OTARI_SECRET_KEY``
rotation. Operator-gated and never mounted in hybrid, as those two are.

A response carries the *names* of the stored secrets and never a value, which is
where this parts company with its siblings' ``last4``: a guardrail may hold
several credentials, so which ones are set is the useful answer and the last
four characters of a map are not one.
"""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_db, require_deployment_operator
from gateway.exceptions.guardrail_credentials import (
    GuardrailCredentialError,
    GuardrailCredentialExistsError,
    GuardrailCredentialNotFoundError,
)
from gateway.models.guardrails import GuardrailCredential
from gateway.services.guardrail_credential_service import (
    UNSET,
    create_guardrail_credential,
    delete_guardrail_credential,
    get_guardrail_credential,
    get_guardrail_credential_for_update,
    list_guardrail_credentials,
    reencrypt_guardrail_credentials,
    stored_secret_names,
    update_guardrail_credential,
)
from gateway.services.secret_box import SecretBoxUnavailableError, SecretDecryptionError

router = APIRouter(
    prefix="/guardrail-credentials",
    tags=["guardrail-credentials"],
    dependencies=[Depends(require_deployment_operator)],
)


class StoredGuardrailSchema(BaseModel):
    """A stored guardrail definition. Credentials are never returned, only their names."""

    name: str
    guardrail_name: str
    create_kwargs: dict[str, Any] = Field(
        default_factory=dict, description="The non-secret constructor arguments, as stored."
    )
    create_secrets: dict[str, str] = Field(
        default_factory=dict,
        description="The stored credentials by name, each masked. Empty when the stored map cannot be read.",
    )
    validate_kwargs: dict[str, Any] = Field(
        default_factory=dict, description="The per-call arguments, with credential-shaped entries masked."
    )
    enabled: bool
    created_at: str | None = None
    updated_at: str | None = None
    decryptable: bool = Field(
        default=True,
        description=(
            "False when the stored credentials cannot be read with the current OTARI_SECRET_KEY. "
            "The definition is intact; re-enter its credentials or restore the key that wrote them."
        ),
    )

    @classmethod
    def from_model(cls, row: GuardrailCredential) -> "StoredGuardrailSchema":
        names, decryptable = stored_secret_names(row)
        return cls(**row.to_public_dict(secret_names=names), decryptable=decryptable)


class CreateGuardrailCredentialRequest(BaseModel):
    """Store a guardrail definition. Secret arguments are encrypted and never returned."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "prompt-injection",
                "guardrail_name": "lakera_guard",
                "create_kwargs": {"api_key": "lak-...", "endpoint": "https://api.lakera.ai/v2/guard"},
            }
        }
    )

    name: str = Field(
        min_length=1,
        max_length=128,
        # One path segment: the row is addressed as /{name}, and neither a bare
        # slash nor an encoded one reaches that route, so a name carrying either
        # would store a row no read, update or delete could ever name again.
        pattern=r"^[^/]+$",
        description="The profile name a caller sends. One path segment, so it cannot contain '/'.",
    )
    guardrail_name: str = Field(
        description="The guardrail to build, as listed by GET /tool-settings/guardrails/catalog."
    )
    create_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Constructor arguments, secret and plain together. They are split by the catalog's own "
            "secret flag; the secret half is encrypted before it is stored."
        ),
    )
    validate_kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Per-call arguments sent with the text on every check."
    )
    enabled: bool = Field(default=True, description="A disabled definition is kept but does not run.")


class UpdateGuardrailCredentialRequest(BaseModel):
    """Update a stored definition. Omitted fields keep their stored value."""

    model_config = ConfigDict(
        json_schema_extra={
            # A partial update, since every field here is optional and a derived
            # example would send placeholders the endpoint refuses. The '***'
            # stands for the stored credential the editor was never shown.
            "example": {
                "create_kwargs": {"api_key": "***", "endpoint": "https://api.lakera.ai/v2/guard"},
                "enabled": False,
            }
        }
    )

    guardrail_name: str | None = None
    create_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Replaces the whole map when sent. A value of '***' keeps the stored credential of that "
            "name, a new value rotates it, and a credential left out is cleared."
        ),
    )
    validate_kwargs: dict[str, Any] | None = None
    enabled: bool | None = None
    expected_updated_at: str | None = Field(
        default=None,
        description="Optimistic concurrency: if set, the update 412s unless it matches the stored updated_at.",
    )


class ReencryptGuardrailCredentialsResponse(BaseModel):
    """Result of re-encrypting stored guardrail credentials with the primary secret key."""

    reencrypted: int = Field(description="Number of stored credential maps re-encrypted.")
    unreadable: int = Field(
        description="Number of maps left untouched because they could not be decrypted."
    )


def _bad_request(exc: Exception) -> HTTPException:
    """Every validation refusal answers 400 with the rule that was broken.

    The messages come from the service and name a parameter, a guardrail or a
    requirement, never a submitted value: this router is operator-only, but the
    error-detail boundary holds here as it does everywhere.
    """
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


def _not_found(name: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND, detail=str(GuardrailCredentialNotFoundError(name))
    )


def _database_error() -> HTTPException:
    """A refused write is a 500 saying only that, as in the sibling stores.

    The service has already rolled back by the time this is raised, so nothing
    here touches the session.
    """
    return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database error")


@router.get("")
async def list_stored_guardrails(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[StoredGuardrailSchema]:
    """List every stored guardrail definition.

    Credentials are never returned. Each row reports which credentials it holds
    by name and whether they can still be read with the current
    ``OTARI_SECRET_KEY``; a row that cannot be read is listed rather than
    hidden, because the operator is the person who can fix it.
    """
    return [StoredGuardrailSchema.from_model(row) for row in await list_guardrail_credentials(db)]


@router.post("/reencrypt")
async def reencrypt_stored_guardrail_credentials(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ReencryptGuardrailCredentialsResponse:
    """Re-encrypt stored guardrail credentials with the primary OTARI_SECRET_KEY.

    The guardrail half of the key-rotation procedure; run it alongside
    ``POST /api/v1/provider-credentials/reencrypt`` and
    ``POST /api/v1/search-tools/reencrypt``. Maps that cannot be decrypted are
    left untouched and must be recovered by re-entering that guardrail's
    credentials.
    """
    try:
        reencrypted, unreadable = await reencrypt_guardrail_credentials(db)
    except SecretBoxUnavailableError as exc:
        await db.rollback()
        raise _bad_request(exc) from None
    except SQLAlchemyError:
        raise _database_error() from None
    return ReencryptGuardrailCredentialsResponse(reencrypted=reencrypted, unreadable=unreadable)


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_stored_guardrail(
    request: CreateGuardrailCredentialRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> StoredGuardrailSchema:
    """Store a guardrail definition.

    The definition is held to what the catalog says its guardrail accepts: an
    unknown guardrail, an argument it does not take, a live object that cannot
    be written down, and a required argument nothing else supplies are each a
    400 naming the rule. Storing a credential requires ``OTARI_SECRET_KEY``.
    """
    name = request.name.strip()
    if not name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="A guardrail name cannot be blank.")
    if await get_guardrail_credential(db, name) is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(GuardrailCredentialExistsError(name)))

    try:
        row = await create_guardrail_credential(
            db,
            name=name,
            guardrail_name=request.guardrail_name,
            create_kwargs=request.create_kwargs,
            validate_kwargs=request.validate_kwargs,
            enabled=request.enabled,
        )
    except GuardrailCredentialExistsError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from None
    except (GuardrailCredentialError, SecretBoxUnavailableError) as exc:
        await db.rollback()
        raise _bad_request(exc) from None
    except SQLAlchemyError:
        raise _database_error() from None

    return StoredGuardrailSchema.from_model(row)


@router.get("/{name}")
async def get_stored_guardrail(
    name: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> StoredGuardrailSchema:
    """Read one stored guardrail definition. Credentials are returned by name only."""
    row = await get_guardrail_credential(db, name)
    if row is None:
        raise _not_found(name)
    return StoredGuardrailSchema.from_model(row)


@router.patch("/{name}")
async def update_stored_guardrail(
    name: str,
    request: UpdateGuardrailCredentialRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> StoredGuardrailSchema:
    """Update a stored guardrail definition. Omitted fields are left as they are.

    The row is locked ``FOR UPDATE`` so the ``expected_updated_at`` check and
    the write it guards are atomic. The definition as it will be *after* the
    update is validated, so a change that would leave it unbuildable (clearing a
    required argument, or moving to a guardrail that does not take an argument
    the row carries) is refused rather than stored.
    """
    row = await get_guardrail_credential_for_update(db, name)
    if row is None:
        raise _not_found(name)
    if request.expected_updated_at is not None:
        current = row.updated_at.isoformat() if row.updated_at else None
        if current != request.expected_updated_at:
            raise HTTPException(
                status_code=status.HTTP_412_PRECONDITION_FAILED,
                detail="This guardrail was modified since you loaded it; reload and retry.",
            )

    sent = request.model_fields_set

    def supplied(field: str) -> Any:
        """The value the caller sent, or UNSET when they sent nothing for this field.

        An omitted field and an explicit null both keep the stored value. None of
        these four is nullable, so there is no third state for a null to mean.
        """
        value = getattr(request, field)
        return value if field in sent and value is not None else UNSET

    try:
        updated = await update_guardrail_credential(
            db,
            row=row,
            guardrail_name=supplied("guardrail_name"),
            create_kwargs=supplied("create_kwargs"),
            validate_kwargs=supplied("validate_kwargs"),
            enabled=supplied("enabled"),
        )
    except (GuardrailCredentialError, SecretBoxUnavailableError, SecretDecryptionError) as exc:
        await db.rollback()
        raise _bad_request(exc) from None
    except SQLAlchemyError:
        raise _database_error() from None

    return StoredGuardrailSchema.from_model(updated)


@router.delete("/{name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_stored_guardrail(
    name: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Delete a stored guardrail definition, credentials and all."""
    try:
        deleted = await delete_guardrail_credential(db, name)
    except SQLAlchemyError:
        raise _database_error() from None
    if not deleted:
        raise _not_found(name)
