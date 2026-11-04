"""Discovery for the tools Otari runs itself.

``GET /api/v1/tools`` answers "what can I put in ``tools[]`` and have the gateway
execute?". Without it the contract is undiscoverable: a client has to read the
docs to learn that ``otari_web_search`` exists, and cannot tell whether this
deployment has a backend wired up for it.

Each entry reports the declaration forms this deployment actually honors right
now, so the answer changes with configuration: interception adds the
provider-named web-search keywords, and a tool with no backend URL is listed as
unavailable rather than hidden, with a ``reason`` saying who can change that.

An API key's answer also takes in its workspace's web search and code execution
switches, read the way the request path reads them, so a tool the workspace has
turned off (or never turned on, where the deployment makes it opt-in) is not
advertised to a key whose requests would be refused. A master key or a dashboard
session belongs to no single workspace and gets the deployment's answer.
"""

from dataclasses import dataclass
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from gateway.api.deps import (
    WorkspaceCodeExecutionPoliciesDep,
    WorkspaceWebSearchPoliciesDep,
    get_config,
    verify_catalog_reader,
)
from gateway.core.config import GatewayConfig
from gateway.core.surface import Surface
from gateway.models.api_keys import APIKey
from gateway.models.tools import ResolvedWebSearchConfig
from gateway.services.sandbox_backend import code_execution_tool_definition
from gateway.services.tools import (
    Tool,
    code_execution_declaration_forms,
    fetch_resolves,
    search_resolves,
    web_fetch_tool_definition,
    web_search_configured,
    web_search_declaration_forms,
    web_search_tool_definition,
)

router = APIRouter(
    tags=["tools"],
    dependencies=[Depends(verify_catalog_reader)],
)

SURFACE = Surface("tools")

ToolUnavailableReason = Literal["not_configured", "workspace_disabled"]


class ManagedTool(BaseModel):
    """One tool the gateway can run itself."""

    id: str = Field(description="The canonical tool type to put in `tools[]`.")
    object: Literal["tool"] = "tool"
    description: str = Field(description="What the tool does, as the model is told.")
    available: bool = Field(
        description=(
            "Whether a request from this caller may declare the tool: the deployment has "
            "enabled and configured it and, for an API key, the key's workspace allows it. "
            "A master key or a dashboard session gets the deployment's answer alone. A "
            "request declaring an unavailable tool is refused."
        )
    )
    reason: ToolUnavailableReason | None = Field(
        default=None,
        description=(
            "Why the tool is unavailable, `null` when it is available. `not_configured`: "
            "nothing is set up to run it (for web search, neither the deployment nor the API "
            "key's organization has a search provider). `workspace_disabled`: the API key's "
            "workspace does not allow it, which a workspace admin changes under Tools in the "
            "dashboard."
        ),
    )
    accepted_types: list[str] = Field(
        description=(
            "Every `tools[].type` this deployment currently routes to the tool. Always "
            "includes the canonical `otari_*` type; for web search it also includes the "
            "provider-named keywords when interception is enabled and the deployment has a "
            "search of its own, and for code execution "
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


@dataclass(frozen=True)
class _WorkspaceSwitches:
    """Whether the caller's workspace allows the web tools and code execution."""

    web: bool = True
    code_execution: bool = True
    # What the web tools resolve on, with the workspace's own keys; ``None`` for the deployment's alone.
    search_policy: ResolvedWebSearchConfig | None = None


async def _workspace_switches(
    config: GatewayConfig,
    api_key: APIKey | None,
    web_policies: WorkspaceWebSearchPoliciesDep,
    code_execution_policies: WorkspaceCodeExecutionPoliciesDep,
) -> _WorkspaceSwitches:
    """The switches the request path applies to ``api_key``'s workspace, all on without a key."""
    if api_key is None:
        return _WorkspaceSwitches()
    web, search_policy = await web_policies.access(api_key.workspace_id)
    return _WorkspaceSwitches(
        web=web,
        code_execution=await code_execution_policies.allows_sandbox(
            api_key.workspace_id, pinnable_images=config.pinnable_sandbox_images()
        ),
        search_policy=search_policy,
    )


def _reason(configured: bool, workspace_allows: bool) -> ToolUnavailableReason | None:
    if not configured:
        return "not_configured"
    return None if workspace_allows else "workspace_disabled"


def _managed_tools(config: GatewayConfig, switches: _WorkspaceSwitches) -> list[ManagedTool]:
    web_search = web_search_tool_definition()["function"]
    web_fetch = web_fetch_tool_definition()["function"]
    code_execution = code_execution_tool_definition()["function"]
    # The same question the request path asks, asked the same way: a hosted
    # provider needs no URL, so deriving this from sandbox_url would publish
    # "unavailable" for a deployment that runs code perfectly well.
    sandbox_configured = config.sandbox_configured()
    deployment_searches = web_search_configured(config)
    policy = switches.search_policy
    can_web_fetch = config.web_fetch_enabled and (policy is None or fetch_resolves(config, policy))
    can_web_search = deployment_searches if policy is None else search_resolves(config, policy)
    return [
        ManagedTool(
            id=Tool.WEB_FETCH,
            description=web_fetch["description"],
            available=can_web_fetch and switches.web,
            reason=_reason(can_web_fetch, switches.web),
            accepted_types=[str(Tool.WEB_FETCH)],
            input_schema=web_fetch["parameters"],
            example={"type": Tool.WEB_FETCH},
        ),
        ManagedTool(
            id=Tool.WEB_SEARCH,
            description=web_search["description"],
            available=can_web_search and switches.web,
            reason=_reason(can_web_search, switches.web),
            # Admission claims a provider's own search keywords only for a deployment that searches itself.
            accepted_types=web_search_declaration_forms(config) if deployment_searches else [str(Tool.WEB_SEARCH)],
            input_schema=web_search["parameters"],
            example={"type": Tool.WEB_SEARCH},
        ),
        ManagedTool(
            id=Tool.CODE_EXECUTION,
            description=code_execution["description"],
            available=sandbox_configured and switches.code_execution,
            reason=_reason(sandbox_configured, switches.code_execution),
            accepted_types=(
                code_execution_declaration_forms(config) if sandbox_configured else [str(Tool.CODE_EXECUTION)]
            ),
            input_schema=code_execution["parameters"],
            example={"type": Tool.CODE_EXECUTION},
        ),
    ]


@router.get("/tools")
async def list_tools(
    config: Annotated[GatewayConfig, Depends(get_config)],
    auth: Annotated[tuple[APIKey | None, bool], Depends(verify_catalog_reader)],
    web_policies: WorkspaceWebSearchPoliciesDep,
    code_execution_policies: WorkspaceCodeExecutionPoliciesDep,
) -> ToolsResponse:
    """List the tools Otari runs itself, with the declaration forms it accepts.

    Called with an API key, a tool its workspace does not allow is listed as
    unavailable. Every other `tools[]` entry, including provider-native keywords
    not listed here, is forwarded to the upstream provider untouched.
    """
    api_key, _ = auth
    switches = await _workspace_switches(config, api_key, web_policies, code_execution_policies)
    return ToolsResponse(data=_managed_tools(config, switches))
