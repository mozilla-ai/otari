"""Storing the versions of agent guardrails that coding agents' hooks report checks against.

A route of its own rather than one of the Hook Server's: that router evaluates
what a caller sends and stores nothing, so it is mounted everywhere. This one
writes to the trace store, so it is mounted only where a control plane, and the
store, exists.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from gateway.api.deps import TraceServiceDep, verify_api_key_or_master_key
from gateway.models.api_keys import APIKey
from gateway.schemas.traces import GuardrailDefinitionRequest, GuardrailDefinitionResponse
from otari_agent.domain.definitions import gates_from_body

router = APIRouter(prefix="/hooks", tags=["hooks"], dependencies=[Depends(verify_api_key_or_master_key)])


@router.post(
    "/guardrails",
    responses={
        status.HTTP_409_CONFLICT: {
            "description": "The workspace already keeps the most guardrail versions it may, and this one is new."
        }
    },
)
async def record_guardrail_definition(
    request: GuardrailDefinitionRequest,
    auth: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    traces: TraceServiceDep,
) -> GuardrailDefinitionResponse:
    """Keep one version of an agent guardrail, so the gate spans that name it can say what each gate is for.

    `otari hook` sends this once per version, the first time it reports a check
    against it. The fingerprint is recomputed from the gates, and one that does
    not match the one sent is refused, so a version cannot be stored under
    another's name. Sending a version again changes nothing. A workspace keeps at
    most 1,000 versions, and a new one past that is refused with a 409. It is stored in the
    API key's workspace, beside that key's traces, and takes the same key a trace
    export does: a budget-exempt API key, never the master key.
    """
    api_key = auth[0]
    if api_key is None or not api_key.exclude_from_budget:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, "A guardrail definition is stored with a budget-exempt API key's traces."
        )
    gates = gates_from_body(gate.model_dump() for gate in request.gates)
    digest = await traces.record_definition(api_key.workspace_id, gates, request.digest)
    return GuardrailDefinitionResponse(digest=digest)
