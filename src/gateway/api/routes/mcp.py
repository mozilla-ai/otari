"""Stateless execution of one caller-approved MCP tool call."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from mcp.types import CallToolResult
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db_if_needed, get_session_identity, verify_api_key_or_master_key
from gateway.api.routes._platform import (
    _extract_platform_user_token,
    _resolve_platform_mcp_servers,
)
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.mcp import McpServerConfig
from gateway.services.mcp_client import MCPClientPool
from gateway.services.url_safety import UnsafeURLError, validate_mcp_url

router = APIRouter(prefix="/v1/mcp", tags=["mcp"])

MCP_TOOL_NOT_ALLOWED_DETAIL = "The requested tool is not allowed by the MCP server configuration"
MCP_TOOL_NOT_FOUND_DETAIL = "The requested tool was not found on the MCP server"
MCP_EXECUTION_FAILED_DETAIL = "MCP server request failed"


class McpExecuteRequest(BaseModel):
    """One MCP server and the exact tool call the client approved."""

    server: McpServerConfig
    tool_name: str = Field(
        min_length=1,
        max_length=256,
        description="The MCP tool the caller has approved for this one execution.",
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="The JSON-object arguments to pass to the approved tool.",
    )


async def _authenticate(
    raw_request: Request,
    db: AsyncSession | None,
    config: GatewayConfig,
) -> None:
    """Authenticate through the mode's existing data-plane path."""
    if config.is_hybrid_mode:
        user_token = _extract_platform_user_token(raw_request)
        # Stateless execution has no stored ids to resolve. The empty resolve
        # deliberately reuses the platform's MCP authorization boundary rather
        # than inventing a second authentication call for this endpoint.
        await _resolve_platform_mcp_servers(config, user_token, [])
        return

    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication temporarily unavailable, please retry",
        )
    session_identity = await get_session_identity(raw_request, db, config)
    await verify_api_key_or_master_key(raw_request, db, config, session_identity)


@router.post(
    "/execute",
    response_model=CallToolResult,
    response_model_exclude_none=True,
)
async def execute_mcp_tool(
    raw_request: Request,
    request: McpExecuteRequest,
    db: Annotated[AsyncSession | None, Depends(get_db_if_needed)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> CallToolResult:
    """Execute one caller-approved tool against one inline MCP server.

    The server is connected only for this request. Otari validates its URL,
    applies its configured tool allowlist, confirms the tool through live MCP
    discovery, and then executes exactly the named call. A server-returned
    ``isError`` remains a typed MCP result; transport and protocol failures are
    returned as a sanitized gateway error.
    """
    await _authenticate(raw_request, db, config)

    server = request.server
    if server.allowed_tools is not None and request.tool_name not in server.allowed_tools:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=MCP_TOOL_NOT_ALLOWED_DETAIL,
        )

    try:
        await validate_mcp_url(
            server.url,
            has_authorization_token=bool(server.authorization_token),
        )
    except UnsafeURLError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    result: CallToolResult | None = None
    try:
        async with MCPClientPool([server]) as pool:
            if not pool.owns_tool(request.tool_name):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=MCP_TOOL_NOT_FOUND_DETAIL,
                )
            result = await pool.call_tool_result(request.tool_name, request.arguments)
    except HTTPException:
        raise
    except Exception as exc:
        if result is not None:
            # The tool already returned a definitive result. A transport close
            # failure must not invite the caller to retry a mutating operation.
            logger.warning("Stateless MCP cleanup failed error_class=%s", type(exc).__name__)
        else:
            logger.warning("Stateless MCP request failed error_class=%s", type(exc).__name__)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=MCP_EXECUTION_FAILED_DETAIL,
            ) from exc

    if result is None:
        raise RuntimeError("MCP tool call completed without a result")
    return result
