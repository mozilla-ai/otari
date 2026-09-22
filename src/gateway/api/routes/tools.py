"""Discovery for the tools Otari runs itself.

``GET /api/v1/tools`` answers "what can I put in ``tools[]`` and have the gateway
execute?". Without it the contract is undiscoverable: a client has to read the
docs to learn that ``otari_web_search`` exists, and cannot tell whether this
deployment has a backend wired up for it.

Each entry reports the declaration forms this deployment actually honors right
now, so the answer changes with configuration: interception adds the
provider-named web-search keywords, and a tool with no backend URL is listed as
unavailable rather than hidden (a client seeing ``available: false`` learns the
tool exists and the operator has not configured it, which is the actionable
distinction).

Standalone-only, like every other non-completion router. Hybrid mode could not
answer this honestly: there the platform owns the per-workspace tool policy
(``_resolve_platform_web_search``), so a tool this gateway has configured may
still be refused with a 403 for the caller asking. Reporting ``available: true``
from local config alone would be worse than not answering.
"""

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from gateway.api.deps import get_config, verify_catalog_reader
from gateway.api.routes._tools import Tool, code_execution_declaration_forms, web_search_declaration_forms
from gateway.core.config import GatewayConfig
from gateway.core.surface import Surface
from gateway.services.sandbox_backend import code_execution_tool_definition
from gateway.services.web_retrieval_backend import web_fetch_tool_definition, web_search_tool_definition

router = APIRouter(
    tags=["tools"],
    dependencies=[Depends(verify_catalog_reader)],
)

SURFACE = Surface("tools")


class ManagedTool(BaseModel):
    """One tool the gateway can run itself."""

    id: str = Field(description="The canonical tool type to put in `tools[]`.")
    object: Literal["tool"] = "tool"
    description: str = Field(description="What the tool does, as the model is told.")
    available: bool = Field(
        description=(
            "Whether this deployment has enabled and configured the tool. A request "
            "declaring an unavailable tool is rejected with 400."
        )
    )
    accepted_types: list[str] = Field(
        description=(
            "Every `tools[].type` this deployment currently routes to the tool. Always "
            "includes the canonical `otari_*` type; for web search it also includes the "
            "provider-named keywords when interception is enabled, and for code execution "
            "the provider-named keywords unless the deployment's executor is `provider`."
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
    web_fetch = web_fetch_tool_definition()["function"]
    code_execution = code_execution_tool_definition()["function"]
    # The same question the request path asks, asked the same way: a hosted
    # provider needs no URL, so deriving this from sandbox_url would publish
    # "unavailable" for a deployment that runs code perfectly well.
    sandbox_configured = config.sandbox_configured()
    can_web_search = config.web_search_configured()
    return [
        ManagedTool(
            id=Tool.WEB_FETCH,
            description=web_fetch["description"],
            available=config.web_fetch_enabled,
            accepted_types=[str(Tool.WEB_FETCH)],
            input_schema=web_fetch["parameters"],
            example={"type": Tool.WEB_FETCH},
        ),
        ManagedTool(
            id=Tool.WEB_SEARCH,
            description=web_search["description"],
            available=can_web_search,
            accepted_types=web_search_declaration_forms(config),
            input_schema=web_search["parameters"],
            example={"type": Tool.WEB_SEARCH},
        ),
        ManagedTool(
            id=Tool.CODE_EXECUTION,
            description=code_execution["description"],
            available=sandbox_configured,
            accepted_types=(
                code_execution_declaration_forms(config) if sandbox_configured else [str(Tool.CODE_EXECUTION)]
            ),
            input_schema=code_execution["parameters"],
            example={"type": Tool.CODE_EXECUTION},
        ),
    ]


@router.get("/tools")
async def list_tools(config: Annotated[GatewayConfig, Depends(get_config)]) -> ToolsResponse:
    """List the tools Otari runs itself, with the declaration forms it accepts.

    Every other `tools[]` entry, including provider-native keywords not listed
    here, is forwarded to the upstream provider untouched.
    """
    return ToolsResponse(data=_managed_tools(config))
