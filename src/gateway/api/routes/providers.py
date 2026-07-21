"""Provider metadata and runtime provider-credential management for the dashboard.

The ``/v1/providers`` endpoint reports static, network-free metadata for every
configured provider. The ``/v1/provider-credentials`` endpoints manage the
``provider_credentials`` table: providers an operator adds at runtime through the
dashboard, encrypted at rest and merged over config.yml providers. All routes are
master-key gated and standalone-mode only (the router is not mounted in hybrid).
"""

from typing import Annotated, Any

from any_llm import LLMProvider
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, verify_master_key
from gateway.core.config import PROVIDER_TYPE_ALIASES, GatewayConfig
from gateway.log_config import logger
from gateway.models.entities import ProviderCredential
from gateway.services.model_discovery_service import (
    discover_provider_models,
    get_model_cache,
    test_provider_credentials,
)
from gateway.services.provider_metadata_service import (
    KnownProvider,
    ProviderInfo,
    list_known_providers,
    list_provider_info,
)
from gateway.services.provider_store_service import (
    UNSET,
    delete_credential,
    get_credential,
    get_credential_for_update,
    list_credentials,
    refresh_provider_cache,
    save_credential,
)
from gateway.services.secret_box import (
    SecretBoxUnavailableError,
    SecretDecryptionError,
    decrypt_secret,
)

router = APIRouter(prefix="/v1", tags=["providers"])


class ProviderCapabilitiesSchema(BaseModel):
    """Curated capability flags for a provider."""

    streaming: bool
    reasoning: bool
    vision: bool
    pdf: bool
    embeddings: bool
    image_generation: bool
    audio: bool
    rerank: bool
    responses_api: bool
    moderation: bool
    list_models: bool


class ProviderInfoSchema(BaseModel):
    """Static, network-free metadata for one configured provider instance."""

    instance: str = Field(description="Configured provider key (may differ from the type).")
    provider_type: str = Field(description="Underlying any-llm provider type.")
    name: str = Field(description="Human-friendly provider name.")
    doc_url: str | None = None
    description: str | None = None
    env_key: str | None = Field(default=None, description="Env var the credential is read from.")
    pricing_urls: list[str] = Field(default_factory=list)
    capabilities: ProviderCapabilitiesSchema


class ProvidersResponse(BaseModel):
    """Metadata for every configured provider."""

    providers: list[ProviderInfoSchema]


def _to_schema(info: ProviderInfo) -> ProviderInfoSchema:
    caps = info.capabilities
    return ProviderInfoSchema(
        instance=info.instance,
        provider_type=info.provider_type,
        name=info.name,
        doc_url=info.doc_url,
        description=info.description,
        env_key=info.env_key,
        pricing_urls=info.pricing_urls,
        capabilities=ProviderCapabilitiesSchema(
            streaming=caps.streaming,
            reasoning=caps.reasoning,
            vision=caps.vision,
            pdf=caps.pdf,
            embeddings=caps.embeddings,
            image_generation=caps.image_generation,
            audio=caps.audio,
            rerank=caps.rerank,
            responses_api=caps.responses_api,
            moderation=caps.moderation,
            list_models=caps.list_models,
        ),
    )


@router.get("/providers", dependencies=[Depends(verify_master_key)])
async def list_providers(
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> ProvidersResponse:
    """List static metadata for every configured provider.

    Operator-facing: reports each provider's capabilities, documentation and
    pricing links, and display name from the bundled any-llm and genai-prices
    datasets. No provider is contacted, so this is cheap and always available.
    Master-key gated because it describes the gateway's own configuration.
    """
    return ProvidersResponse(providers=[_to_schema(info) for info in list_provider_info(config)])


class KnownProviderSchema(BaseModel):
    """A provider the add-provider picker can offer, with autofill hints."""

    id: str = Field(description="any-llm provider id, used as the default instance name.")
    name: str = Field(description="Human-friendly display name.")
    env_key: str | None = Field(default=None, description="Env var the SDK reads this provider's key from.")
    default_api_base: str | None = Field(default=None, description="Built-in endpoint; blank means the SDK's default.")
    requires_api_key: bool = Field(description="False for keyless local backends (Ollama, llama.cpp).")


def _to_known_schema(provider: KnownProvider) -> KnownProviderSchema:
    return KnownProviderSchema(
        id=provider.id,
        name=provider.name,
        env_key=provider.env_key,
        default_api_base=provider.default_api_base,
        requires_api_key=provider.requires_api_key,
    )


@router.get("/providers/catalog", dependencies=[Depends(verify_master_key)])
async def provider_catalog() -> list[KnownProviderSchema]:
    """List every known provider for the add-provider picker.

    Network-free and config-independent: the full any-llm provider set with each
    one's display name, credential env var, default endpoint, and whether it
    needs a key. Master-key gated because it is operator-facing dashboard data.
    """
    return [_to_known_schema(provider) for provider in list_known_providers()]


# --------------------------------------------------------------------------- #
# Runtime provider-credential management (/v1/provider-credentials)
# --------------------------------------------------------------------------- #


class StoredProviderResponse(BaseModel):
    """A runtime-stored provider. The API key is never returned, only ``last4``."""

    instance: str
    provider_type: str | None = None
    api_base: str | None = None
    last4: str | None = None
    client_args: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None
    # False when the stored key cannot be decrypted with the current
    # OTARI_SECRET_KEY (e.g. the key was rotated or lost). Such a provider is
    # skipped at runtime, so the dashboard flags it for the operator to fix.
    decryptable: bool = True

    @classmethod
    def from_model(cls, row: ProviderCredential, *, decryptable: bool = True) -> "StoredProviderResponse":
        return cls(**row.to_public_dict(), decryptable=decryptable)


def _is_decryptable(row: ProviderCredential) -> bool:
    """Whether the row's stored key can be read with the current OTARI_SECRET_KEY."""
    if not row.encrypted_api_key:
        return True
    try:
        decrypt_secret(row.encrypted_api_key)
    except (SecretBoxUnavailableError, SecretDecryptionError):
        return False
    return True


class CreateStoredProviderRequest(BaseModel):
    """Create a stored provider. ``api_key`` is write-only and requires OTARI_SECRET_KEY."""

    instance: str = Field(min_length=1, description="Routing key, e.g. 'openai' or a named instance like 'home_lab'.")
    provider_type: str | None = Field(
        default=None,
        description="any-llm implementation when the instance name is not itself one.",
    )
    api_base: str | None = None
    api_key: str | None = Field(default=None, description="Provider API key. Stored encrypted; never returned.")
    client_args: dict[str, Any] | None = None


class UpdateStoredProviderRequest(BaseModel):
    """Update a stored provider. Omitted fields are unchanged; ``api_key`` rotates in place."""

    provider_type: str | None = None
    api_base: str | None = None
    api_key: str | None = Field(default=None, description="New API key. Omit to keep the existing one. Never returned.")
    client_args: dict[str, Any] | None = None
    expected_updated_at: str | None = Field(
        default=None,
        description="Optimistic concurrency: if set, the update 412s unless it matches the stored updated_at.",
    )


class TestProviderResponse(BaseModel):
    """Result of a live provider connection test."""

    ok: bool
    model_count: int
    error: str | None = None


class TestProviderRequest(BaseModel):
    """Credentials to test before saving (from the add-provider form)."""

    instance: str | None = Field(default=None, description="Provider/instance name; the impl when no provider_type.")
    provider_type: str | None = None
    api_base: str | None = None
    api_key: str | None = None
    client_args: dict[str, Any] | None = None


def _validate_instance(instance: str, provider_type: str | None) -> None:
    """Reject an unroutable instance name or an unknown provider_type, as a 400."""
    if not instance.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provider instance name must not be blank.",
        )
    if ":" in instance or "/" in instance:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provider instance name must not contain ':' or '/'.",
        )
    if provider_type:
        impl = PROVIDER_TYPE_ALIASES.get(provider_type, provider_type)
        try:
            LLMProvider(impl)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"provider_type '{provider_type}' is not a known provider implementation.",
            ) from exc


async def _commit(db: AsyncSession, *, conflict_detail: str | None = None) -> None:
    try:
        await db.commit()
    except IntegrityError:
        # A concurrent create can slip past the pre-check and collide on the
        # primary key here; surface that as the intended 409, not a 500.
        await db.rollback()
        if conflict_detail is not None:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=conflict_detail) from None
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database error") from None
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None


async def _apply_write(db: AsyncSession, config: GatewayConfig, instance: str) -> None:
    """Make a committed credential change take effect on this worker.

    Clears the model-discovery cache for the instance (a stale listing would
    otherwise survive a key change) and re-merges the overlay. The write is
    already committed, so a refresh failure is logged, not surfaced as a 500;
    other workers converge within the TTL.
    """
    get_model_cache().clear(instance)
    try:
        await refresh_provider_cache(db, config)
    except SQLAlchemyError:
        logger.warning("Provider overlay refresh failed after writing '%s'; converges within TTL", instance)


@router.post("/provider-credentials/test", dependencies=[Depends(verify_master_key)])
async def test_provider_connection(
    request: TestProviderRequest,
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> TestProviderResponse:
    """Test provider credentials without storing them (for the add/edit form).

    Resolves the implementation from ``provider_type`` (honoring the
    ``*-compatible`` aliases) or the ``instance`` name, then lists the provider's
    models with the supplied credentials. Nothing is persisted and the key is
    never echoed.
    """
    impl = (
        PROVIDER_TYPE_ALIASES.get(request.provider_type, request.provider_type)
        if request.provider_type
        else (request.instance or "")
    )
    if not impl:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide a provider or a provider type to test.",
        )
    result = await test_provider_credentials(
        impl,
        api_key=request.api_key,
        api_base=request.api_base,
        client_args=request.client_args,
        timeout=config.model_discovery_timeout_seconds,
    )
    return TestProviderResponse(ok=result.error is None, model_count=len(result.models), error=result.error)


@router.get("/provider-credentials", dependencies=[Depends(verify_master_key)])
async def list_stored_providers(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[StoredProviderResponse]:
    """List runtime-stored providers. Keys are never returned, only ``last4``."""
    return [
        StoredProviderResponse.from_model(row, decryptable=_is_decryptable(row)) for row in await list_credentials(db)
    ]


@router.post("/provider-credentials", status_code=status.HTTP_201_CREATED, dependencies=[Depends(verify_master_key)])
async def create_stored_provider(
    request: CreateStoredProviderRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> StoredProviderResponse:
    """Add a provider at runtime. Storing a key requires OTARI_SECRET_KEY."""
    _validate_instance(request.instance, request.provider_type)
    if await get_credential(db, request.instance) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A stored provider '{request.instance}' already exists; use PATCH to update it.",
        )
    try:
        row = await save_credential(
            db,
            instance=request.instance,
            provider_type=request.provider_type,
            api_base=request.api_base,
            api_key=request.api_key,
            client_args=request.client_args,
        )
    except SecretBoxUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from None

    await _commit(
        db,
        conflict_detail=f"A stored provider '{request.instance}' already exists; use PATCH to update it.",
    )
    if request.instance in (config._provider_baseline or {}):
        logger.warning(
            "Stored provider '%s' shadows the config.yml provider of the same name; the stored credential now wins.",
            request.instance,
        )
    await _apply_write(db, config, request.instance)
    await db.refresh(row)
    return StoredProviderResponse.from_model(row)


@router.patch("/provider-credentials/{instance}", dependencies=[Depends(verify_master_key)])
async def update_stored_provider(
    instance: str,
    request: UpdateStoredProviderRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> StoredProviderResponse:
    """Update a stored provider. Omitted fields are left as-is; an explicit ``null`` clears them.

    ``api_key`` follows the same rule: omit it to keep the stored key, send a new
    one to rotate, or send ``null`` to clear it. The row is locked ``FOR UPDATE``
    so the ``expected_updated_at`` check and the write it guards are atomic.
    """
    existing = await get_credential_for_update(db, instance)
    if existing is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No stored provider '{instance}'.")
    if request.expected_updated_at is not None:
        current = existing.updated_at.isoformat() if existing.updated_at else None
        if current != request.expected_updated_at:
            raise HTTPException(
                status_code=status.HTTP_412_PRECONDITION_FAILED,
                detail="This provider was modified since you loaded it; reload and retry.",
            )
    _validate_instance(instance, request.provider_type)
    # Distinguish "field omitted" (keep) from "field set to null" (clear).
    sent = request.model_fields_set
    try:
        row = await save_credential(
            db,
            instance=instance,
            provider_type=request.provider_type if "provider_type" in sent else UNSET,
            api_base=request.api_base if "api_base" in sent else UNSET,
            api_key=request.api_key if "api_key" in sent else UNSET,
            client_args=request.client_args if "client_args" in sent else UNSET,
        )
    except SecretBoxUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from None

    await _commit(db)
    await _apply_write(db, config, instance)
    await db.refresh(row)
    return StoredProviderResponse.from_model(row)


@router.delete(
    "/provider-credentials/{instance}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(verify_master_key)],
)
async def delete_stored_provider(
    instance: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> None:
    """Delete a stored provider. A config.yml provider cannot be deleted here."""
    if not await delete_credential(db, instance):
        detail = f"No stored provider '{instance}'."
        if instance in (config._provider_baseline or {}):
            detail = f"Provider '{instance}' is defined in config.yml and cannot be deleted through the API."
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)
    await _commit(db)
    await _apply_write(db, config, instance)


@router.post("/provider-credentials/{instance}/test", dependencies=[Depends(verify_master_key)])
async def test_stored_provider(
    instance: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> TestProviderResponse:
    """Verify a stored provider's key by listing its models, without exposing the key."""
    if await get_credential(db, instance) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No stored provider '{instance}'.")
    # Make sure the just-stored credential is merged, then force a live check
    # rather than trusting a cached listing from before a key change.
    await _apply_write(db, config, instance)
    result = await discover_provider_models(config, instance)
    return TestProviderResponse(ok=result.error is None, model_count=len(result.models), error=result.error)
