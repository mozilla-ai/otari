"""Runtime search and fetch instance management for the dashboard (``/api/v1/search-tools``).

``POST /api/v1/search`` dispatches against the ``search_tools`` map. That map used to
come only from a config file, so a deployment configured entirely through the
dashboard and environment variables could not use the endpoint at all (issue
#601). These endpoints are the missing route in: rows in ``search_tool_credentials``,
the API key encrypted at rest and never returned, merged over the config-file
tools with the stored row winning on a name collision. A row is a search or a
fetch instance, as its ``kind`` says, and fetch instances share these routes, so
one prefix serves both.

Config-file tools stay honored and stay read-only here; they are reported by the
list endpoint so the dashboard can show every tool a request could name, not just
the editable ones.

Operator-gated. URL
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

import httpx
from fastapi import APIRouter, Depends, Query, Request, status

from gateway.api.deps import (
    SearchToolServiceDep,
    SearchToolTesterDep,
    WebProviderClientDep,
    catalog_reader_operates_deployment,
    get_config,
    require_deployment_operator,
    verify_catalog_reader,
)
from gateway.api.routes._helpers import database_error
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.core.database import DATABASE_ERRORS
from gateway.core.settings.tools import ToolInstance, ToolKind
from gateway.inflight import track_request
from gateway.schemas.tools import (
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
from gateway.services.tools import check_test_input, run_connection_test, search_provider_catalog

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
    service: SearchToolServiceDep,
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
    try:
        return await service.list_tools(kind)
    except DATABASE_ERRORS as exc:
        raise database_error("list stored search tools", exc) from None


@router.post("/reencrypt")
async def reencrypt_stored_search_tool_keys(service: SearchToolServiceDep) -> ReencryptSearchToolsResponse:
    """Re-encrypt stored search-tool keys with the primary OTARI_SECRET_KEY.

    The search-tool half of the ``OTARI_SECRET_KEY`` rotation procedure; run it
    alongside ``POST /api/v1/hosted-providers/reencrypt``. Rows that cannot be
    decrypted are left untouched and must be recovered by replacing the affected
    tool's key.
    """
    try:
        return await service.reencrypt()
    except DATABASE_ERRORS as exc:
        raise database_error("reencrypt stored search tool keys", exc) from None


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_search_tool(
    request: CreateSearchToolRequest, service: SearchToolServiceDep
) -> CreatedSearchToolSchema:
    """Add a search or fetch instance at runtime. Storing an API key requires OTARI_SECRET_KEY.

    Creating a second search instance, while the first is the in-loop default
    only because it is the only one, first sets ``web_search_default_tool`` to
    the first in the same commit, and says so in the response, so that adding
    an instance never turns in-loop search off. That runtime value wins over the
    configuration file until it is cleared.
    """
    try:
        return await service.create(request)
    except DATABASE_ERRORS as exc:
        raise database_error("create search tool", exc) from None


@router.patch("/{name}")
async def update_search_tool(
    name: str, request: UpdateSearchToolRequest, service: SearchToolServiceDep
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
    try:
        return await service.update(name, request)
    except DATABASE_ERRORS as exc:
        raise database_error("update search tool", exc) from None


@router.delete("/{name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_stored_search_tool(name: str, service: SearchToolServiceDep) -> None:
    """Delete a stored search or fetch instance. A config-file one, or ``builtin_fetch``, cannot be deleted here."""
    try:
        await service.delete(name)
    except DATABASE_ERRORS as exc:
        raise database_error("delete stored search tool", exc) from None


# The in-flight registry's name for a connection test, named or not.
TEST_ENDPOINT = f"{API_ROOT}/search-tools/test"


async def _dispatch_test(
    raw_request: Request,
    config: GatewayConfig,
    instance: ToolInstance,
    entry: Mapping[str, Any],
    *,
    query: str | None,
    url: str | None,
    client: httpx.AsyncClient,
) -> SearchToolTestResponse:
    """Run a connection test as a provider call, seen in flight."""
    check_test_input(instance, query=query, url=url)
    track_request(raw_request, endpoint=TEST_ENDPOINT, model=instance.name, provider=instance.provider)
    return await run_connection_test(config, instance, entry, query=query, url=url, client=client)


@router.post("/test")
async def test_unsaved_search_tool(
    request: SearchToolTestRequest,
    raw_request: Request,
    service: SearchToolTesterDep,
    config: Annotated[GatewayConfig, Depends(get_config)],
    web_provider_client: WebProviderClientDep,
) -> SearchToolTestResponse:
    """Test an instance before saving it, with one search or one fetch.

    Takes the create request's fields, held to the create's checks against the
    instances this worker has loaded, plus ``query`` for a search instance or
    ``url`` for a fetch instance. Answers whether the provider answered without
    an error, the error's tag when it did not, and how many hits or characters
    came back, never the results or the page.
    """
    instance, entry = service.unsaved_instance_to_test(request)
    return await _dispatch_test(
        raw_request, config, instance, entry, query=request.query, url=request.url, client=web_provider_client
    )


@router.post("/{name}/test")
async def test_search_tool(
    name: str,
    request: StoredSearchToolTestRequest,
    raw_request: Request,
    service: SearchToolTesterDep,
    config: Annotated[GatewayConfig, Depends(get_config)],
    web_provider_client: WebProviderClientDep,
) -> SearchToolTestResponse:
    """Test a configured or stored instance with one search or one fetch.

    Takes ``query`` for a search instance or ``url`` for a fetch instance, and
    answers as ``POST /search-tools/test`` does. ``builtin_fetch``, the built-in
    fetcher, fetches as a fetch tool call does, with no domain policy.
    """
    instance, entry = await service.instance_to_test(name)
    return await _dispatch_test(
        raw_request, config, instance, entry, query=request.query, url=request.url, client=web_provider_client
    )
