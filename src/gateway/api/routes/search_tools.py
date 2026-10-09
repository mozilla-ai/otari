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

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

import any_fetch
import any_search
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
    RESERVED_INSTANCE_NAMES,
    SEARCH_PROVIDERS_REQUIRING_API_BASE,
    SEARCH_PROVIDERS_WITHOUT_ADAPTER,
    ToolInstance,
    ToolKind,
    effective_fetch_instances,
    effective_search_instances,
    instance_from_entry,
    instance_name_problems,
    option_problems,
    search_default_to_pin,
    validate_fetch_tool_entry,
    validate_search_tool_entry,
    validate_search_tool_transport,
)
from gateway.inflight import track_request
from gateway.log_config import logger
from gateway.models.tools import SearchToolCredential
from gateway.schemas.tools import SearchProviderSchema
from gateway.services.search_backend import (
    SearchProviderError,
    SearchQuery,
    SearchToolError,
    resolve_search_tool,
    run_search,
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
    validate_url,
)
from gateway.services.tools import search_provider_catalog

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


class StoredSearchToolSchema(BaseModel):
    """A runtime-stored search or fetch instance. The API key is never returned, only ``last4``."""

    name: str
    kind: ToolKind = Field(default="search", description="Whether this is a search or a fetch instance.")
    provider: str
    fetch_tool: str | None = Field(
        default=None,
        description="A search instance's enrichment fetch instance. Null means the fetch default enriches it.",
    )
    api_base: str | None = None
    last4: str | None = None
    timeout: float | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None
    # False when the stored key cannot be decrypted with the current
    # OTARI_SECRET_KEY. Such a tool is skipped at runtime, so the dashboard flags
    # it for the operator to fix.
    decryptable: bool = True
    shadows_config: bool = Field(
        default=False,
        description="True when a config-file search tool of the same name exists; the stored one is in effect.",
    )

    @classmethod
    def from_model(
        cls,
        row: SearchToolCredential,
        *,
        decryptable: bool = True,
        shadows_config: bool = False,
    ) -> "StoredSearchToolSchema":
        return cls(**row.to_public_dict(), decryptable=decryptable, shadows_config=shadows_config)


class CreatedSearchToolSchema(StoredSearchToolSchema):
    """A stored instance as created, with the default setting the create also stored, if any."""

    pinned_web_search_default_tool: str | None = Field(
        default=None,
        description=(
            "Set when this create also stored web_search_default_tool, naming the search instance that was "
            "the in-loop default because it was the only one, so that adding a second does not turn in-loop "
            "search off. This runtime value wins over the configuration file until it is cleared."
        ),
    )
    notice: str | None = Field(default=None, description="What else the create changed, for the operator to read.")


class ConfigSearchToolSchema(BaseModel):
    """A search or fetch instance declared in the config file, or ``builtin_fetch``. Read-only here."""

    name: str
    kind: ToolKind = Field(default="search", description="Whether this is a search or a fetch instance.")
    provider: str
    fetch_tool: str | None = Field(
        default=None,
        description="A search instance's enrichment fetch instance. Null means the fetch default enriches it.",
    )
    api_base: str | None = None
    has_api_key: bool = Field(description="Whether the config entry carries an API key. The key itself is not shown.")
    shadowed: bool = Field(
        default=False,
        description="True when a stored instance of the same name and kind overrides this entry.",
    )


class SearchToolsResponse(BaseModel):
    """Every search instance ``POST /api/v1/search`` can name, or every fetch instance, by where it came from."""

    stored: list[StoredSearchToolSchema]
    config: list[ConfigSearchToolSchema]


class CreateSearchToolRequest(BaseModel):
    """Create a stored search or fetch instance. ``api_key`` is write-only and requires OTARI_SECRET_KEY."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"name": "local", "provider": "searxng", "api_base": "http://searxng:8080"}}
    )

    name: str = Field(
        min_length=1,
        pattern=r"^[^/:]+$",
        description=(
            "Name callers pass as 'search_tool_name' or in /api/v1/search/{tool}, or that names a fetch "
            "instance. It contains no '/' or ':', is not builtin_fetch or none in any case, and is unique "
            "across search and fetch instances."
        ),
    )
    kind: ToolKind = Field(default="search", description="'search' or 'fetch'. It cannot change once created.")
    provider: str = Field(
        description=(
            "Provider id. GET /api/v1/search-tools/providers lists the search providers, and with "
            "?kind=fetch the fetch providers."
        )
    )
    fetch_tool: str | None = Field(
        default=None,
        description=(
            "For a search instance: the fetch instance that enriches its results, a configured or stored one "
            "or builtin_fetch. Omit it for the fetch default."
        ),
    )
    api_base: str | None = Field(
        default=None,
        description="Backend endpoint. Omit to inherit the provider's default (searxng inherits web_search_url).",
    )
    api_key: str | None = Field(default=None, description="Provider API key. Stored encrypted; never returned.")
    timeout: float | None = Field(default=None, gt=0, description="Per-request timeout in seconds.")
    options: dict[str, Any] | None = Field(
        default=None,
        description="Provider-native request fields used as defaults (e.g. exa's 'type', searxng's 'engines').",
    )


class UpdateSearchToolRequest(BaseModel):
    """Update a stored search or fetch instance. Omitted fields are unchanged; ``api_key`` rotates in place."""

    kind: ToolKind | None = Field(
        default=None, description="Accepted only when it matches the stored kind, which cannot change."
    )
    provider: str | None = None
    fetch_tool: str | None = Field(
        default=None,
        description="For a search instance: the fetch instance that enriches its results. Null clears it.",
    )
    api_base: str | None = None
    api_key: str | None = Field(default=None, description="New API key. Omit to keep the existing one. Never returned.")
    timeout: float | None = Field(default=None, gt=0)
    options: dict[str, Any] | None = None
    expected_updated_at: str | None = Field(
        default=None,
        description="Optimistic concurrency: if set, the update 412s unless it matches the stored updated_at.",
    )


class ReencryptSearchToolsResponse(BaseModel):
    """Result of re-encrypting stored search-tool keys with the primary secret key."""

    reencrypted: int = Field(description="Number of stored search-tool keys re-encrypted.")
    unreadable: int = Field(description="Number of encrypted keys left untouched because they could not be decrypted.")
    skipped: int = Field(
        default=0,
        description=(
            "Number of rows whose stored key changed between the read and the write, so the "
            "re-encryption was not applied. They already hold whoever wrote them last."
        ),
    )


class SearchToolTestRequest(CreateSearchToolRequest):
    """An unsaved search or fetch instance to test, and what to test it with."""

    query: str | None = Field(default=None, min_length=1, description="For a search instance: the query to run.")
    url: str | None = Field(default=None, min_length=1, description="For a fetch instance: the page to fetch.")


class StoredSearchToolTestRequest(BaseModel):
    """What to test a configured or stored instance with."""

    query: str | None = Field(default=None, min_length=1, description="For a search instance: the query to run.")
    url: str | None = Field(default=None, min_length=1, description="For a fetch instance: the page to fetch.")


class SearchToolTestResponse(BaseModel):
    """How one search or one fetch went. Never the results or the page."""

    ok: bool = Field(description="Whether the provider answered the call without an error.")
    error: str | None = Field(
        default=None,
        description=(
            "When not ok, the error's tag: timeout, network, http_error, invalid_response, or the provider's own."
        ),
    )
    hits: int | None = Field(default=None, description="For a search that worked: how many hits came back.")
    characters: int | None = Field(
        default=None, description="For a fetch that worked: how many characters of page text came back."
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


def _unprocessable(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=detail)


def _section(kind: ToolKind) -> str:
    return "search_tools" if kind == "search" else "fetch_tools"


def _validate_entry(name: str, entry: dict[str, Any], *, kind: ToolKind, inherited_api_base: str | None = None) -> None:
    """Hold a dashboard-written instance to the rules the config file is held to, as a 422.

    The same validation startup runs on, per kind, so an instance saved here can
    never be one that would refuse to boot from a config file.
    """
    try:
        if kind == "search":
            validate_search_tool_entry(name, entry)
        else:
            validate_fetch_tool_entry(name, entry)
    except ValueError as exc:
        raise _unprocessable(str(exc)) from None
    provider = str(entry.get("provider") or name)
    api_base = entry.get("api_base")
    if not api_base and kind == "search" and provider in SEARCH_PROVIDERS_REQUIRING_API_BASE:
        api_base = inherited_api_base
    if api_base:
        try:
            validate_url(str(api_base))
            validate_search_tool_transport(name, api_base, entry.get("api_key"))
        except ValueError as exc:
            raise _unprocessable(str(exc)) from None


def _check_write_rules(
    config: GatewayConfig,
    name: str,
    entry: dict[str, Any],
    *,
    kind: ToolKind,
    sets_name: bool,
    sets_options: bool,
    sets_fetch_tool: bool,
) -> None:
    """Refuse, as a 422, what the instance rules let load but not be written.

    Only in what the write sets: a search instance stored before the rules keeps
    loading with its name and options, so a change that leaves them alone, such
    as rotating its key, does not trip on them. A fetch instance's name is held
    to the rules by its entry's own validation.
    """
    provider = str(entry.get("provider") or name)
    problems = instance_name_problems(name) if sets_name and kind == "search" else []
    if sets_options:
        problems += option_problems(kind, provider, entry.get("options") or {}).values()
    if problems:
        raise _unprocessable(f"{_section(kind)}.{name} is refused: {'; '.join(problems)}.")
    fetch_tool = entry.get("fetch_tool")
    if not sets_fetch_tool or fetch_tool is None:
        return
    if kind == "fetch":
        raise _unprocessable(f"fetch_tools.{name}.fetch_tool is refused: only a search instance has one.")
    if fetch_tool not in effective_fetch_instances(config):
        raise _unprocessable(
            f"search_tools.{name}.fetch_tool must name a fetch instance, or {BUILTIN_FETCH}; "
            f"there is no '{fetch_tool}'."
        )


def _check_name_is_free(config: GatewayConfig, name: str, kind: ToolKind) -> None:
    """Refuse, as a 422, a name an instance of the other kind already has.

    Names are unique across search and fetch instances, because the pricing key
    ``<provider>:<instance>`` carries no capability. A stored instance may still
    take the name of a config-file one of its own kind, which it then overrides.
    """
    other: ToolKind = "fetch" if kind == "search" else "search"
    if name in (config.fetch_tools if kind == "search" else config.search_tools):
        raise _unprocessable(
            f"A {other} instance named '{name}' exists; names are unique across search and fetch instances."
        )


def _pin_refusal(pinned: str, *, stored: bool) -> str:
    way_out = (
        f"delete '{pinned}' and create it again under another name, since a stored name cannot change"
        if stored
        else f"rename '{pinned}' in the configuration file"
    )
    return (
        f"Adding a second search instance would leave the in-loop tool with no default, so this create first "
        f"sets web_search_default_tool to '{pinned}', the only search instance until now. That name is reserved "
        f"and cannot be the default: {way_out}, then add this one."
    )


def _strip(value: str | None) -> str | None:
    return value.strip() if isinstance(value, str) else value


def _entry(request: CreateSearchToolRequest) -> dict[str, Any]:
    """The request as a config-file entry, the shape the checks and the overlay read."""
    return {
        "provider": request.provider,
        "fetch_tool": _strip(request.fetch_tool),
        "api_base": request.api_base,
        "api_key": request.api_key,
        "timeout": request.timeout,
        "options": request.options,
    }


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
    entry = _entry(request)
    _validate_entry(name, entry, kind=kind, inherited_api_base=config.web_search_url)
    try:
        # Another replica may have stored an instance this one would clash with or
        # name, or the default setting the pin below reads.
        await refresh_tool_instances(db, config)
    except DATABASE_ERRORS:
        await db.rollback()
        logger.exception("Failed to reload stored tools before creating '%s'", name)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database error") from None
    _check_write_rules(
        config,
        name,
        entry,
        kind=kind,
        sets_name=True,
        sets_options=request.options is not None,
        sets_fetch_tool=True,
    )
    _check_name_is_free(config, name, kind)
    conflict = f"A stored {kind} tool '{name}' already exists; use PATCH to update it."
    if await get_search_tool(db, name) is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=conflict)
    pinned = search_default_to_pin(config, name) if kind == "search" else None
    if pinned is not None and pinned.name.lower() in RESERVED_INSTANCE_NAMES:
        stored = await get_search_tool(db, pinned.name) is not None
        raise _unprocessable(_pin_refusal(pinned.name, stored=stored))
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
    notice = None
    if pinned is not None:
        apply_override(config, WEB_SEARCH_DEFAULT_TOOL, pinned.name)
        notice = (
            f"web_search_default_tool is now '{pinned.name}'. The in-loop tool searched with it as the only search "
            f"instance, and adding '{name}' would otherwise have turned in-loop search off. This runtime value "
            "wins over the configuration file until it is cleared."
        )
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
        notice=notice,
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
    if request.kind is not None and request.kind != kind:
        raise _unprocessable(
            f"'{name}' is a {kind} instance, and an instance's kind cannot change: delete it and create it "
            f"again as a {request.kind} instance."
        )
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
    merged: dict[str, Any] = {
        "provider": request.provider if "provider" in sent and request.provider else existing.provider,
        "fetch_tool": _strip(request.fetch_tool) if "fetch_tool" in sent else existing.fetch_tool,
        "api_base": request.api_base if "api_base" in sent else existing.api_base,
        "timeout": request.timeout if "timeout" in sent else existing.timeout_seconds,
        "options": request.options if "options" in sent else existing.options,
        # Only presence matters to the validator, and the stored key is never
        # decrypted here just to re-validate it.
        "api_key": request.api_key if "api_key" in sent else existing.encrypted_api_key,
    }
    _validate_entry(name, merged, kind=kind, inherited_api_base=config.web_search_url)
    sets_fetch_tool = "fetch_tool" in sent
    if sets_fetch_tool and merged["fetch_tool"] is not None:
        try:
            # The fetch instance it names may have been stored through another replica.
            await refresh_search_tool_cache(db, config)
        except DATABASE_ERRORS:
            await db.rollback()
            logger.exception("Failed to reload stored tools before updating '%s'", name)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database error") from None
    _check_write_rules(
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


def _library_call(instance: ToolInstance) -> tuple[dict[str, Any], dict[str, Any]]:
    """What a library provider is built and called with: the instance's settings and options as the resolver sends them.

    The instance's own key and base URL, never the environment's: the libraries
    read a provider's variable when none is passed, and an empty one counts as
    passed. Options carry otari's defaults below them, and leave out the ones
    the rules drop.
    """
    default_timeout = any_search.DEFAULT_TIMEOUT if instance.kind == "search" else any_fetch.DEFAULT_TIMEOUT
    settings = {
        "api_key": instance.api_key or "",
        "api_base": instance.api_base or "",
        "timeout": instance.timeout or default_timeout,
    }
    kept = {key: value for key, value in instance.options.items() if key not in instance.dropped_options}
    return settings, {**instance.provider_defaults, **kept}


async def _test_with_library(instance: ToolInstance, *, query: str | None, url: str | None) -> SearchToolTestResponse:
    settings, options = _library_call(instance)
    try:
        if instance.kind == "search":
            async with any_search.AnySearch.create(instance.provider, **settings) as engine:
                result = await engine.search(str(query), **options)
            if result.error is not None:
                return SearchToolTestResponse(ok=False, error=result.error.tag)
            return SearchToolTestResponse(ok=True, hits=len(result.hits))
        async with any_fetch.AnyFetch.create(instance.provider, **settings) as fetcher:
            page = await fetcher.fetch(str(url), **options)
        if page.error is not None:
            return SearchToolTestResponse(ok=False, error=page.error.tag)
        return SearchToolTestResponse(ok=True, characters=len(page.text))
    except (any_search.ProviderError, any_fetch.ProviderError) as exc:
        return SearchToolTestResponse(ok=False, error=exc.tag)
    except (any_search.AnySearchError, any_fetch.AnyFetchError) as exc:
        # A key or option the provider cannot run with: the instance's to fix.
        raise _unprocessable(str(exc)) from None


async def _test_with_old_client(
    config: GatewayConfig, name: str, entry: dict[str, Any], query: str
) -> SearchToolTestResponse:
    """One search through the direct endpoint's own client, which serves SearXNG until any-search does."""
    try:
        # Resolved as the direct endpoint resolves it, against this entry alone,
        # so an unsaved one inherits web_search_url and the engines the same way.
        tool = resolve_search_tool(config.model_copy(update={"search_tools": {name: entry}}), name)
    except SearchToolError as exc:
        raise _unprocessable(str(exc)) from None
    try:
        outcome = await run_search(tool, SearchQuery(query=query))
    except SearchProviderError as exc:
        return SearchToolTestResponse(ok=False, error=exc.tag)
    return SearchToolTestResponse(ok=True, hits=len(outcome.results))


# The in-flight registry's name for a connection test, named or not.
TEST_ENDPOINT = f"{API_ROOT}/search-tools/test"


async def _run_test(
    raw_request: Request,
    db: AsyncSession,
    config: GatewayConfig,
    instance: ToolInstance,
    entry: dict[str, Any],
    *,
    query: str | None,
    url: str | None,
) -> SearchToolTestResponse:
    if instance.kind == "search" and not query:
        raise _unprocessable("A search instance is tested with a 'query'.")
    if instance.kind == "fetch" and not url:
        raise _unprocessable("A fetch instance is tested with a 'url'.")
    # Not held across the provider call, which can take the instance's whole timeout.
    await release_session(db)
    track_request(raw_request, endpoint=TEST_ENDPOINT, model=instance.name, provider=instance.provider)
    if instance.kind == "search" and instance.provider in SEARCH_PROVIDERS_WITHOUT_ADAPTER:
        response = await _test_with_old_client(config, instance.name, entry, str(query))
    else:
        response = await _test_with_library(instance, query=query, url=url)
    logger.info(
        "Connection test of %s instance '%s': %s",
        instance.kind,
        instance.name,
        "ok" if response.ok else f"failed ({response.error})",
    )
    return response


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
    entry = _entry(request)
    _validate_entry(name, entry, kind=request.kind, inherited_api_base=config.web_search_url)
    _check_write_rules(
        config,
        name,
        entry,
        kind=request.kind,
        sets_name=True,
        sets_options=request.options is not None,
        sets_fetch_tool=True,
    )
    _check_name_is_free(config, name, request.kind)
    instance = instance_from_entry(request.kind, name, entry)
    return await _run_test(raw_request, db, config, instance, entry, query=request.query, url=request.url)


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
    if (search := effective_search_instances(config).get(name)) is not None:
        entry = config.search_tools[name]
        return await _run_test(raw_request, db, config, search, entry, query=request.query, url=request.url)
    if name == BUILTIN_FETCH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"{BUILTIN_FETCH}, the built-in fetcher, has no connection test yet: it arrives when web fetch "
                "runs on the fetch instances."
            ),
        )
    if (fetch := effective_fetch_instances(config).get(name)) is not None:
        entry = config.fetch_tools[name]
        return await _run_test(raw_request, db, config, fetch, entry, query=request.query, url=request.url)
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No search or fetch instance '{name}'.")
