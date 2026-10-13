"""Runtime search and fetch instance management for the dashboard (``/api/v1/search-tools``).

``POST /api/v1/search`` dispatches against the ``search_tools`` map. That map used to
come only from a config file, so a deployment configured entirely through the
dashboard and environment variables could not use the endpoint at all (issue
#601). These endpoints are the missing route in, and they are deliberately the
same shape as ``/api/v1/provider-credentials``: rows in ``search_tool_credentials``,
the API key encrypted at rest and never returned, merged over the config-file
tools with the stored row winning on a name collision. A row is a search or a
fetch instance, as its ``kind`` says, and fetch instances share these routes, so
one prefix serves both.

Config-file tools stay honored and stay read-only here; they are reported by the
list endpoint so the dashboard can show every tool a request could name, not just
the editable ones.

Operator-gated and standalone-only (the router is not mounted in hybrid). URL
validation is structural, matching ``/api/v1/tool-settings`` rather than the provider
SSRF gate: the backend this most often points at is a SearXNG sidecar on a
private address, which a deny-private gate would refuse.

``/api/v1/search-tools/providers`` is the exception, on ``catalog_router``: it lists
the providers any-search and any-fetch serve, which is a property of the build
rather than of this deployment, so any catalog reader may see it. What it says
about this deployment, its tools and their inherited endpoints, only an operator
sees.
"""

from collections.abc import Mapping
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import (
    catalog_reader_operates_deployment,
    get_config,
    get_db,
    require_deployment_operator,
    verify_catalog_reader,
)
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.core.database import DATABASE_ERRORS, release_session
from gateway.core.settings.tools import (
    BUILTIN_FETCH,
    BUILTIN_FETCH_PROVIDER,
    ToolInstance,
    ToolKind,
)
from gateway.inflight import track_request
from gateway.log_config import logger
from gateway.models.tools import SearchToolCredential
from gateway.schemas.tools import (
    ConfigSearchToolSchema,
    CreatedSearchToolSchema,
    CreateSearchToolRequest,
    ReencryptSearchToolsResponse,
    SearchProviderSchema,
    SearchToolsResponse,
    SearchToolTestRequest,
    SearchToolTestResponse,
    StoredSearchToolSchema,
    StoredSearchToolTestRequest,
    UpdateSearchToolRequest,
)
from gateway.services.search_tool_store_service import (
    UNSET,
    config_file_search_tools,
    config_file_tools,
    delete_search_tool,
    get_search_tool,
    get_search_tool_for_update,
    list_search_tools,
    reencrypt_search_tools,
    refresh_search_tool_cache,
    refresh_tool_instances,
    save_search_tool,
    stored_tool_names,
)
from gateway.services.secret_box import (
    SecretBoxUnavailableError,
    SecretDecryptionError,
    decrypt_secret,
)
from gateway.services.tool_settings_service import (
    WEB_SEARCH_DEFAULT_TOOL,
    apply_override,
    stage_override,
)
from gateway.services.tools import (
    check_kind_unchanged,
    check_test_input,
    check_tool_entry,
    check_tool_name_is_free,
    check_tool_write,
    default_to_pin,
    instance_to_test,
    run_connection_test,
    search_provider_catalog,
    unsaved_instance_to_test,
)

router = APIRouter(
    prefix="/search-tools",
    tags=["search-tools"],
    dependencies=[Depends(require_deployment_operator)],
)
# The provider catalog, which describes the installed libraries, and only to an
# operator anything this deployment configured. A tenant reaches it:
# organization search keys name a provider too, and it is owners and admins who
# fill that form, never an operator. Gated like the other catalog reads so
# admitting a session is spelled at the router (see
# ``deps.verify_catalog_reader``).
catalog_router = APIRouter(
    prefix="/search-tools",
    tags=["search-tools"],
    dependencies=[Depends(verify_catalog_reader)],
)


def _is_decryptable(row: SearchToolCredential) -> bool:
    """Whether the row's stored key can be read with the current OTARI_SECRET_KEY."""
    if not row.encrypted_api_key:
        return True
    try:
        decrypt_secret(row.encrypted_api_key)
    except (SecretBoxUnavailableError, SecretDecryptionError):
        return False
    return True


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
    except DATABASE_ERRORS:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database error") from None


async def _apply_write(db: AsyncSession, config: GatewayConfig, name: str) -> None:
    """Make a committed search-tool change take effect on this worker.

    The write is already committed, so a refresh failure is logged, not surfaced
    as a 500; other workers converge within the TTL.
    """
    try:
        await refresh_search_tool_cache(db, config)
    except DATABASE_ERRORS:
        logger.warning("Search tool overlay refresh failed after writing '%s'; converges within TTL", name)


@catalog_router.get("/providers")
async def list_search_providers(
    config: Annotated[GatewayConfig, Depends(get_config)],
    operates: Annotated[bool, Depends(catalog_reader_operates_deployment)],
    kind: Annotated[
        ToolKind, Query(description="Which providers to list: search providers (the default) or fetch providers.")
    ] = "search",
) -> list[SearchProviderSchema]:
    """List the providers a search or fetch tool may name, for the add-tool form.

    The list comes from the metadata any-search and any-fetch publish, so a
    provider either library adds appears with no change to the gateway. Reports
    per provider whether an API key is required, what endpoint a tool inherits
    when it declares none, and the native options a tool may set. Providers that
    exist only for tests are left out, and so is the fetch provider ``builtin``,
    which only the implicit ``builtin_fetch`` tool uses.

    What belongs to this deployment rather than to the libraries, its own tools
    on each provider and an endpoint a tool inherits from its settings, is
    shown only to a caller who operates the deployment: the tool settings
    reader withholds the same from anyone else.
    """
    return search_provider_catalog(config, kind, operates=operates)


@router.get("")
async def list_all_search_tools(
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    kind: Annotated[ToolKind, Query(description="Which instances to list: 'search' (the default) or 'fetch'.")] = (
        "search"
    ),
) -> SearchToolsResponse:
    """List every search instance ``POST /api/v1/search`` can name, or with ``?kind=fetch`` every fetch instance.

    ``stored`` are the editable rows written through this API; ``config`` are the
    config-file entries, which are still honored and are reported so the operator
    can see the whole set, ``builtin_fetch`` first among the fetch instances. Keys
    are never returned, only ``last4``.
    """
    from_config = config_file_tools(config, kind)
    stored = await list_search_tools(db, kind)
    stored_names = {row.name for row in stored}
    builtin = [
        ConfigSearchToolSchema(name=BUILTIN_FETCH, kind="fetch", provider=BUILTIN_FETCH_PROVIDER, has_api_key=False)
    ]
    return SearchToolsResponse(
        stored=[
            StoredSearchToolSchema.from_model(
                row,
                decryptable=_is_decryptable(row),
                shadows_config=row.name in from_config,
            )
            for row in stored
        ],
        config=(builtin if kind == "fetch" else [])
        + [
            ConfigSearchToolSchema(
                name=name,
                kind=kind,
                provider=str(entry.get("provider") or name),
                fetch_tool=entry.get("fetch_tool") if kind == "search" else None,
                api_base=entry.get("api_base"),
                has_api_key=bool(entry.get("api_key")),
                shadowed=name in stored_names,
            )
            for name, entry in sorted(from_config.items())
        ],
    )


@router.post("/reencrypt")
async def reencrypt_stored_search_tool_keys(
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> ReencryptSearchToolsResponse:
    """Re-encrypt stored search-tool keys with the primary OTARI_SECRET_KEY.

    The search-tool half of the ``OTARI_SECRET_KEY`` rotation procedure; run it
    alongside ``POST /api/v1/provider-credentials/reencrypt``. Rows that cannot be
    decrypted are left untouched and must be recovered by replacing the affected
    tool's key.
    """
    try:
        reencrypted, unreadable, skipped = await reencrypt_search_tools(db)
        await db.commit()
    except SecretBoxUnavailableError as exc:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from None
    except DATABASE_ERRORS:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database error") from None
    try:
        await refresh_search_tool_cache(db, config)
    except DATABASE_ERRORS:
        logger.warning("Search tool overlay refresh failed after re-encrypting keys; converges within TTL")
    return ReencryptSearchToolsResponse(reencrypted=reencrypted, unreadable=unreadable, skipped=skipped)


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_search_tool(
    request: CreateSearchToolRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> CreatedSearchToolSchema:
    """Add a search or fetch instance at runtime. Storing an API key requires OTARI_SECRET_KEY.

    Creating a second search instance, while the first is the in-loop default
    only because it is the only one, first sets ``web_search_default_tool`` to
    the first in the same commit, and says so in the response, so that adding
    an instance never turns in-loop search off. That runtime value wins over the
    configuration file until it is cleared.
    """
    name = request.name.strip()
    kind = request.kind
    entry = request.to_entry()
    check_tool_entry(name, entry, kind=kind, inherited_api_base=config.web_search_url)
    try:
        # Another replica may have stored an instance this one would clash with or
        # name, or the default setting the pin below reads.
        await refresh_tool_instances(db, config)
    except DATABASE_ERRORS:
        await db.rollback()
        logger.exception("Failed to reload stored tools before creating '%s'", name)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database error") from None
    check_tool_write(
        config,
        name,
        entry,
        kind=kind,
        sets_name=True,
        sets_options=request.options is not None,
        sets_fetch_tool=True,
    )
    check_tool_name_is_free(config, name, kind)
    conflict = f"A stored {kind} tool '{name}' already exists; use PATCH to update it."
    if await get_search_tool(db, name) is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=conflict)
    pinned = default_to_pin(config, name, stored_names=stored_tool_names()) if kind == "search" else None
    try:
        row = await save_search_tool(
            db,
            name=name,
            kind=kind,
            provider=request.provider,
            fetch_tool=entry["fetch_tool"],
            api_base=request.api_base,
            api_key=request.api_key,
            timeout=request.timeout,
            options=request.options,
        )
        if pinned is not None:
            await stage_override(db, WEB_SEARCH_DEFAULT_TOOL, pinned.name)
    except SecretBoxUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from None

    if pinned is not None:
        # The default's row can collide too, with another create pinning at the same time.
        conflict = (
            f"'{name}' was not stored: a stored tool of that name exists, or another create changed the "
            "search instances at the same time. Reload and retry."
        )
    await _commit(db, conflict_detail=conflict)
    if pinned is not None:
        apply_override(config, WEB_SEARCH_DEFAULT_TOOL, pinned.name)
        logger.info("Set web_search_default_tool to '%s' before adding a second search instance", pinned.name)
    from_config = config_file_tools(config, kind)
    shadows_config = name in from_config
    if shadows_config:
        logger.warning(
            "Stored %s tool '%s' shadows the config.yml entry of the same name; the stored entry now wins.",
            kind,
            name,
        )
    await _apply_write(db, config, name)
    await db.refresh(row)
    return CreatedSearchToolSchema(
        **row.to_public_dict(),
        shadows_config=shadows_config,
        pinned_web_search_default_tool=pinned.name if pinned is not None else None,
        notice=pinned.notice if pinned is not None else None,
    )


@router.patch("/{name}")
async def update_search_tool(
    name: str,
    request: UpdateSearchToolRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> StoredSearchToolSchema:
    """Update a stored search or fetch instance. Omitted fields are left as-is; an explicit ``null`` clears them.

    ``api_key`` follows the same rule: omit it to keep the stored key, send a new
    one to rotate, or send ``null`` to clear it (a keyless SearXNG backend). The
    row is locked ``FOR UPDATE`` so the ``expected_updated_at`` check and the
    write it guards are atomic. The tool as it will be after the update is
    validated, so a change that would leave it unusable (clearing the key of a
    provider that needs one) is refused rather than stored. Its options are
    checked against the provider's when the update sets them or changes the
    provider, and ``fetch_tool`` when the update sets it, so rotating the key of
    an instance stored before those rules never trips on them. ``kind`` cannot
    change, so an instance never moves between the search and fetch maps.
    """
    existing = await get_search_tool_for_update(db, name)
    if existing is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No stored search tool '{name}'.")
    kind: ToolKind = "fetch" if existing.kind == "fetch" else "search"
    check_kind_unchanged(name, kind, request.kind)
    if request.expected_updated_at is not None:
        current = existing.updated_at.isoformat() if existing.updated_at else None
        if current != request.expected_updated_at:
            raise HTTPException(
                status_code=status.HTTP_412_PRECONDITION_FAILED,
                detail="This search tool was modified since you loaded it; reload and retry.",
            )

    # Distinguish "field omitted" (keep) from "field set to null" (clear), then
    # validate the resulting tool rather than the patch in isolation.
    sent = request.model_fields_set
    fetch_tool = request.fetch_tool.strip() if request.fetch_tool is not None else None
    merged: dict[str, Any] = {
        "provider": request.provider if "provider" in sent and request.provider else existing.provider,
        "fetch_tool": fetch_tool if "fetch_tool" in sent else existing.fetch_tool,
        "api_base": request.api_base if "api_base" in sent else existing.api_base,
        "timeout": request.timeout if "timeout" in sent else existing.timeout_seconds,
        "options": request.options if "options" in sent else existing.options,
        # Only presence matters to the validator, and the stored key is never
        # decrypted here just to re-validate it.
        "api_key": request.api_key if "api_key" in sent else existing.encrypted_api_key,
    }
    check_tool_entry(name, merged, kind=kind, inherited_api_base=config.web_search_url)
    sets_fetch_tool = "fetch_tool" in sent
    if sets_fetch_tool and merged["fetch_tool"] is not None:
        try:
            # The fetch instance it names may have been stored through another replica.
            await refresh_search_tool_cache(db, config)
        except DATABASE_ERRORS:
            await db.rollback()
            logger.exception("Failed to reload stored tools before updating '%s'", name)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database error") from None
    check_tool_write(
        config,
        name,
        merged,
        kind=kind,
        sets_name=False,
        sets_options="options" in sent or merged["provider"] != existing.provider,
        sets_fetch_tool=sets_fetch_tool,
    )

    try:
        row = await save_search_tool(
            db,
            name=name,
            # An explicit null provider is meaningless (the column is non-null),
            # so it is treated as "unchanged" rather than rejected.
            provider=request.provider if "provider" in sent and request.provider else UNSET,
            fetch_tool=merged["fetch_tool"] if sets_fetch_tool else UNSET,
            api_base=request.api_base if "api_base" in sent else UNSET,
            api_key=request.api_key if "api_key" in sent else UNSET,
            timeout=request.timeout if "timeout" in sent else UNSET,
            options=request.options if "options" in sent else UNSET,
        )
    except SecretBoxUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from None

    await _commit(db)
    await _apply_write(db, config, name)
    await db.refresh(row)
    from_config = config_file_tools(config, kind)
    return StoredSearchToolSchema.from_model(row, shadows_config=name in from_config)


@router.delete("/{name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_stored_search_tool(
    name: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> None:
    """Delete a stored search or fetch instance. A config-file one, or ``builtin_fetch``, cannot be deleted here."""
    if not await delete_search_tool(db, name):
        detail = f"No stored search tool '{name}'."
        if name in config_file_search_tools(config):
            detail = f"Search tool '{name}' is defined in the config file and cannot be deleted through the API."
        elif name in config_file_tools(config, "fetch"):
            detail = f"Fetch tool '{name}' is defined in the config file and cannot be deleted through the API."
        elif name == BUILTIN_FETCH:
            detail = f"{BUILTIN_FETCH}, the built-in fetcher, always exists and cannot be deleted."
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)
    await _commit(db)
    await _apply_write(db, config, name)


# The in-flight registry's name for a connection test, named or not.
TEST_ENDPOINT = f"{API_ROOT}/search-tools/test"


async def _dispatch_test(
    raw_request: Request,
    db: AsyncSession,
    config: GatewayConfig,
    instance: ToolInstance,
    entry: Mapping[str, Any],
    *,
    query: str | None,
    url: str | None,
) -> SearchToolTestResponse:
    """Run a connection test as a provider call: no pooled connection held across it, and seen in flight."""
    check_test_input(instance, query=query, url=url)
    await release_session(db)
    track_request(raw_request, endpoint=TEST_ENDPOINT, model=instance.name, provider=instance.provider)
    return await run_connection_test(config, instance, entry, query=query, url=url)


@router.post("/test")
async def test_unsaved_search_tool(
    request: SearchToolTestRequest,
    raw_request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> SearchToolTestResponse:
    """Test an instance before saving it, with one search or one fetch.

    Takes the create request's fields, held to the create's checks against the
    instances this worker has loaded, plus ``query`` for a search instance or
    ``url`` for a fetch instance. Answers whether the provider answered without
    an error, the error's tag when it did not, and how many hits or characters
    came back, never the results or the page.
    """
    name = request.name.strip()
    entry = request.to_entry()
    instance = unsaved_instance_to_test(config, name, entry, kind=request.kind)
    return await _dispatch_test(raw_request, db, config, instance, entry, query=request.query, url=request.url)


@router.post("/{name}/test")
async def test_search_tool(
    name: str,
    request: StoredSearchToolTestRequest,
    raw_request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> SearchToolTestResponse:
    """Test a configured or stored instance with one search or one fetch.

    Takes ``query`` for a search instance or ``url`` for a fetch instance, and
    answers as ``POST /search-tools/test`` does. ``builtin_fetch`` has no test
    yet, and answers a 400.
    """
    try:
        # The instance may have been stored through another replica.
        await refresh_search_tool_cache(db, config)
    except DATABASE_ERRORS:
        logger.warning("Search tool overlay refresh failed before testing '%s'; testing the loaded one", name)
    instance, entry = instance_to_test(config, name)
    return await _dispatch_test(raw_request, db, config, instance, entry, query=request.query, url=request.url)
