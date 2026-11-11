"""Agent memory over MCP, a core feature switched by ``memory_enabled``.

Listed in ``CORE_FEATURES`` as ``FEATURE`` below. ``POST /memory/mcp`` speaks
MCP's Streamable HTTP transport, stateless and with JSON responses only: each
request is one JSON-RPC message, and a tool call is answered in the response to
the POST that made it. There is no server-to-client stream, so ``GET`` is
refused, as the transport allows.

The caller's workspace and member come from its API key (``get_memory_caller``),
never from a tool argument, and every result is returned as data with its
provenance and trust level, for the agent to weigh rather than obey.
"""

import functools
import json
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request, Response, status
from fastapi.responses import JSONResponse

from gateway.api.deps import (
    CurrentIdentity,
    MemoryCallerDep,
    MemoryServiceDep,
    build_hosted_provider_service,
    build_memory_service,
    get_config,
    memory_keys_for,
    verify_api_key,
    verify_master_key,
)
from gateway.core.config import GatewayConfig
from gateway.core.feature import CoreFeature
from gateway.core.surface import Surface
from gateway.ports.content_key_port import ContentKeyPort
from gateway.schemas.memory import (
    OrganizationMemoryPublic,
    OrganizationMemoryUpdate,
    WorkspaceMemoryErased,
    WorkspaceMemoryPublic,
    WorkspaceMemoryUpdate,
)
from gateway.services.memory import (
    MEMORY_INSTRUCTIONS,
    MemoryCaller,
    MemoryService,
    SwitchChange,
    ToolOutcome,
    WorkspaceMemory,
    call_memory_tool,
    mcp_tool_definitions,
    run_consolidator,
)
from gateway.version import __version__

# Agents reach memory with an API key, and its workspace and member are the ones that key names.
router = APIRouter(prefix="/memory", tags=["memory"], dependencies=[Depends(verify_api_key)])

MAX_BODY_BYTES = 64 * 1024
LATEST_PROTOCOL_VERSION = "2025-11-25"
SUPPORTED_PROTOCOL_VERSIONS = frozenset({"2025-03-26", "2025-06-18", LATEST_PROTOCOL_VERSION})

_PARSE_ERROR = -32700
_INVALID_REQUEST = -32600
_METHOD_NOT_FOUND = -32601
_INVALID_PARAMS = -32602


def _rpc_result(message_id: object, result: dict[str, Any]) -> JSONResponse:
    return JSONResponse({"jsonrpc": "2.0", "id": message_id, "result": result})


def _rpc_error(message_id: object, code: int, message: str) -> JSONResponse:
    return JSONResponse({"jsonrpc": "2.0", "id": message_id, "error": {"code": code, "message": message}})


def _tool_result(outcome: ToolOutcome) -> dict[str, Any]:
    if outcome.error is not None:
        return {"content": [{"type": "text", "text": outcome.error}], "isError": True}
    return {"content": [{"type": "text", "text": outcome.text}], "structuredContent": outcome.data, "isError": False}


async def _call_tool(service: MemoryService, caller: MemoryCaller, params: object) -> dict[str, Any] | None:
    """The tool's result, or None when the call names no tool this server has."""
    if not isinstance(params, dict) or not isinstance(params.get("name"), str):
        return None
    outcome = await call_memory_tool(service, caller, params["name"], params.get("arguments"))
    return _tool_result(outcome) if outcome is not None else None


@router.post(
    "/mcp",
    responses={
        202: {"description": "A notification, which has no response."},
        403: {"description": "The key names no member of its workspace."},
        413: {"description": "The message exceeds 64 KiB."},
        415: {"description": "Only application/json is accepted."},
    },
)
async def memory_mcp(request: Request, service: MemoryServiceDep, caller: MemoryCallerDep) -> Response:
    """Answer one MCP JSON-RPC message: initialize, ping, tools/list or tools/call."""
    if request.headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
        return Response(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)
    body = await request.body()
    if len(body) > MAX_BODY_BYTES:
        return Response(status_code=status.HTTP_413_CONTENT_TOO_LARGE)
    try:
        message = json.loads(body)
    except ValueError:
        return _rpc_error(None, _PARSE_ERROR, "Parse error")
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or not isinstance(message.get("method"), str):
        return _rpc_error(None, _INVALID_REQUEST, "Send one JSON-RPC 2.0 message; batches are not supported.")
    if "id" not in message:
        return Response(status_code=status.HTTP_202_ACCEPTED)
    message_id, method, params = message["id"], message["method"], message.get("params")
    if method == "initialize":
        requested = params.get("protocolVersion") if isinstance(params, dict) else None
        return _rpc_result(
            message_id,
            {
                "protocolVersion": requested if requested in SUPPORTED_PROTOCOL_VERSIONS else LATEST_PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "otari-memory", "version": __version__},
                "instructions": MEMORY_INSTRUCTIONS,
            },
        )
    if method == "ping":
        return _rpc_result(message_id, {})
    if method == "tools/list":
        return _rpc_result(message_id, {"tools": mcp_tool_definitions()})
    if method == "tools/call":
        result = await _call_tool(service, caller, params)
        if result is None:
            return _rpc_error(message_id, _INVALID_PARAMS, "Unknown tool.")
        return _rpc_result(message_id, result)
    return _rpc_error(message_id, _METHOD_NOT_FOUND, "Method not found.")


@router.get("/mcp", status_code=status.HTTP_405_METHOD_NOT_ALLOWED, responses={405: {"description": "No stream."}})
async def memory_mcp_stream() -> Response:
    """No server-to-client stream: every answer comes back on the POST that asked."""
    return Response(status_code=status.HTTP_405_METHOD_NOT_ALLOWED, headers={"Allow": "POST"})


# The dashboard's and the operator's: a workspace's managers turn memory on and off, and erase it.
workspace_router = APIRouter(
    prefix="/workspaces/{workspace_id}/memory", tags=["memory"], dependencies=[Depends(verify_master_key)]
)


def _workspace_public(memory: WorkspaceMemory, config: GatewayConfig) -> WorkspaceMemoryPublic:
    return WorkspaceMemoryPublic(
        workspace_id=memory.workspace_id,
        deployment_enabled=config.memory_enabled,
        organization_enabled=memory.organization_enabled,
        enabled=memory.enabled,
        workspace_scope_enabled=memory.workspace_scope_enabled,
        write_policy=memory.write_policy,
        offer_to_every_request=memory.offer_to_every_request,
        hide_external=memory.hide_external,
        retention_days=memory.retention_days,
        deployment_retention_days=config.memory_retention_days,
        entries=memory.entries,
        proposals_waiting=memory.proposals_waiting,
    )


@workspace_router.get("")
async def get_workspace_memory(
    service: MemoryServiceDep,
    current_identity: CurrentIdentity,
    config: Annotated[GatewayConfig, Depends(get_config)],
    workspace_id: uuid.UUID,
) -> WorkspaceMemoryPublic:
    """A workspace's memory switches and how much it holds. An organization owner/admin, or this workspace's."""
    return _workspace_public(await service.workspace_memory(current_identity, workspace_id), config)


@workspace_router.put("")
async def set_workspace_memory(
    service: MemoryServiceDep,
    current_identity: CurrentIdentity,
    config: Annotated[GatewayConfig, Depends(get_config)],
    workspace_id: uuid.UUID,
    body: WorkspaceMemoryUpdate,
) -> WorkspaceMemoryPublic:
    """Set a workspace's memory switches. Turning memory off stops recall at once and deletes nothing.

    An organization owner/admin, or an owner/admin of this workspace, may.
    """
    change = SwitchChange(
        enabled=body.enabled,
        workspace_scope_enabled=body.workspace_scope_enabled,
        write_policy=body.write_policy,
        offer_to_every_request=body.offer_to_every_request,
        hide_external=body.hide_external,
        retention_days=body.retention_days,
    )
    return _workspace_public(await service.change_switches(current_identity, workspace_id, change), config)


@workspace_router.delete("")
async def erase_workspace_memory(
    service: MemoryServiceDep, current_identity: CurrentIdentity, workspace_id: uuid.UUID
) -> WorkspaceMemoryErased:
    """Delete everything a workspace's memory holds, and the data keys it was sealed under.

    With the keys gone, nothing deleted opens from a backup either. The switches stay as they were.
    """
    erased = await service.erase(current_identity, workspace_id)
    return WorkspaceMemoryErased(
        workspace_id=workspace_id,
        entries=erased.entries,
        proposals=erased.proposals,
        reads=erased.reads,
        keys=erased.keys,
    )


# The caller's active organization: its owners and admins turn memory on or off for every workspace of it.
organization_router = APIRouter(
    prefix="/organizations/me/memory", tags=["memory"], dependencies=[Depends(verify_master_key)]
)


@organization_router.get("")
async def get_organization_memory(
    service: MemoryServiceDep,
    current_identity: CurrentIdentity,
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> OrganizationMemoryPublic:
    """The caller's organization's memory switch. Its owners and admins only."""
    memory = await service.organization_memory(current_identity)
    return OrganizationMemoryPublic(
        organization_id=memory.organization_id, deployment_enabled=config.memory_enabled, enabled=memory.enabled
    )


@organization_router.put("")
async def set_organization_memory(
    service: MemoryServiceDep,
    current_identity: CurrentIdentity,
    config: Annotated[GatewayConfig, Depends(get_config)],
    body: OrganizationMemoryUpdate,
) -> OrganizationMemoryPublic:
    """Turn memory on or off for every workspace of the caller's organization. Off deletes nothing.

    On lets each workspace's managers turn it on for their workspace; off stops recall in all of them at once.
    """
    memory = await service.change_organization_switch(current_identity, enabled=body.enabled)
    return OrganizationMemoryPublic(
        organization_id=memory.organization_id, deployment_enabled=config.memory_enabled, enabled=memory.enabled
    )


async def _consolidate(config: GatewayConfig) -> None:
    # A worker is handed the config alone, so it binds its own ports the way the app does,
    # overlay included: memory is sealed and opened under the same key source either way.
    from gateway.container import build_container  # noqa: PLC0415 - the composition root, needed by this worker only

    container = build_container(
        config.bootstrap,
        config=config,
        hosted_provider_service=functools.partial(build_hosted_provider_service, config=config),
    )
    keys = memory_keys_for(config, container.resolve(ContentKeyPort, None))
    try:
        await run_consolidator(lambda uow: build_memory_service(uow, config, keys, container))
    finally:
        keys.clear()


FEATURE = CoreFeature(
    name="memory",
    # Published while the deployment offers memory, so the dashboard shows its switches only then.
    surface=Surface("memory"),
    enabled=lambda config: config.memory_enabled,
    routers=lambda config: (router, workspace_router, organization_router),
    worker=_consolidate,
)
