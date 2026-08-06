"""Discovery for the tools Otari runs itself.

``GET /v1/tools`` answers "what can I put in ``tools[]`` and have the gateway
execute?". Without it the contract is undiscoverable: a client has to read the
docs to learn that ``otari_web_search`` exists, and cannot tell whether this
deployment has a backend wired up for it.

Each entry reports the declaration forms this deployment actually honours right
now, so the answer changes with configuration: interception adds the
provider-named web-search keywords, and a tool with no backend URL is listed as
unavailable rather than hidden (a client seeing ``available: false`` learns the
tool exists and the operator has not configured it, which is the actionable
distinction).

Registered in both runtime modes: hybrid mode serves the completion endpoints
too. Note that hybrid mode additionally enforces a per-workspace policy the
gateway cannot see from here, so an advertised tool may still be refused with a
403 at request time.
"""

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from gateway.api.deps import get_config, verify_api_key_or_master_key
from gateway.api.routes._tools import Tool, web_search_declaration_forms
from gateway.core.config import GatewayConfig
from gateway.core.env import otari_env
from gateway.services.sandbox_backend import code_execution_tool_definition
from gateway.services.web_search_backend import web_search_tool_definition

router = APIRouter(prefix="/v1", tags=["tools"])


class ManagedTool(BaseModel):
    """One tool the gateway can run itself."""

    id: str = Field(description="The canonical tool type to put in `tools[]`.")
    object: Literal["tool"] = "tool"
    description: str = Field(description="What the tool does, as the model is told.")
    available: bool = Field(
        description=(
            "Whether this deployment has a backend configured for the tool. A request "
            "declaring an unavailable tool is rejected with 400."
        )
    )
    accepted_types: list[str] = Field(
        description=(
            "Every `tools[].type` this deployment currently routes to the tool. Always "
            "includes the canonical `otari_*` type; for web search it also includes the "
            "provider-named keywords when interception is enabled."
        )
    )
    input_schema: dict[str, Any] = Field(
        description="JSON Schema for the arguments the model supplies, as the model sees it."
    )
    example: dict[str, Any] = Field(description="A ready-to-use `tools[]` entry.")


class ToolsResponse(BaseModel):
    """The gateway-run tools this deployment exposes."""

    object: Literal["list"] = "list"
    data: list[ManagedTool]


def _managed_tools(config: GatewayConfig) -> list[ManagedTool]:
    web_search = web_search_tool_definition()["function"]
    code_execution = code_execution_tool_definition()["function"]
    # Same resolution the request path uses: the effective config value, falling back
    # to the env var so a pure-env deployment reports accurately.
    sandbox_configured = bool(config.sandbox_url or otari_env("SANDBOX_URL"))
    web_search_configured = bool(config.web_search_url or otari_env("WEB_SEARCH_URL"))
    return [
        ManagedTool(
            id=Tool.WEB_SEARCH,
            description=web_search["description"],
            available=web_search_configured,
            accepted_types=web_search_declaration_forms(config),
            input_schema=web_search["parameters"],
            example={"type": Tool.WEB_SEARCH},
        ),
        ManagedTool(
            id=Tool.CODE_EXECUTION,
            description=code_execution["description"],
            available=sandbox_configured,
            accepted_types=[str(Tool.CODE_EXECUTION)],
            input_schema=code_execution["parameters"],
            example={"type": Tool.CODE_EXECUTION},
        ),
    ]


@router.get("/tools", dependencies=[Depends(verify_api_key_or_master_key)])
async def list_tools(config: Annotated[GatewayConfig, Depends(get_config)]) -> ToolsResponse:
    """List the tools Otari runs itself, with the declaration forms it accepts.

    Every other `tools[]` entry, including provider-native keywords not listed
    here, is forwarded to the upstream provider untouched.
    """
    return ToolsResponse(data=_managed_tools(config))
