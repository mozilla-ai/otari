"""Decisions endpoint: typed questions about a state, answered with probabilities.

``POST /api/v1/decisions`` takes the body TypeSafe's ``/v1/systemone`` defined, which
OpenRouter's alpha Decisions API and llama-server's ``/v1/systemone`` adopted, and
forwards it to the provider the ``model`` prefix names under ``decision_providers``.
``POST /api/v1/systemone`` serves the same request at the path TypeSafe's SDK and
llama-server use, so a client written for either works with only its base URL changed.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response

from gateway.api.deps import verify_api_key_or_master_key
from gateway.api.routes._passthrough import DecisionScaffold, get_decision_scaffold
from gateway.models.api_keys import APIKey
from gateway.schemas.inference import DecisionRequest, DecisionResponse

router = APIRouter(tags=["decisions"])


@router.post("/decisions")
async def create_decision(
    raw_request: Request,
    response: Response,
    request: DecisionRequest,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    scaffold: Annotated[DecisionScaffold, Depends(get_decision_scaffold)],
) -> DecisionResponse:
    """Answer typed questions (noul, choice, score) about a state.

    ``model`` is ``<provider>:<model>``, where the provider is a ``decision_providers``
    entry, for example ``typesafe:jev-latest`` or ``openrouter:typesafe/jev-1.13``.

    Authentication modes:
    - Master key: the ``user`` field is required and may name any existing user.
    - API key: usage and spend bind to the key's own user; a ``user`` naming a
      different user is rejected with 403 unless mismatch rejection is off.
    """
    return await scaffold.run(raw_request=raw_request, response=response, request=request, auth_result=auth_result)


@router.post("/systemone")
async def create_systemone_decision(
    raw_request: Request,
    response: Response,
    request: DecisionRequest,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    scaffold: Annotated[DecisionScaffold, Depends(get_decision_scaffold)],
) -> DecisionResponse:
    """Answer typed questions at the path TypeSafe's SDK and llama-server use.

    Identical to ``POST /api/v1/decisions``; point the client's base URL at the gateway's ``/api``.
    """
    return await scaffold.run(raw_request=raw_request, response=response, request=request, auth_result=auth_result)
