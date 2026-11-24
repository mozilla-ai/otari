"""Unit tests for the stored-file scope check's ordering in ``resolve_request_context``.

The normalizer refuses a key that attaches a stored file it may not read, but it
runs after the reservation and the pricing gate, so the refusal used to lose to
a 402. ``_require_stored_file_scope`` asks the same question beside the
declared-tool scopes, ahead of both, and these tests pin that order: the refusal
fires before any budget work where the request attaches a stored upload, and
stays silent where it does not.
"""

import uuid
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi import HTTPException, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

import gateway.api.routes._pipeline as pipeline
from _unavailable_model_support import reporter_with_null_listener
from gateway.api.routes import chat
from gateway.container import build_container
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.ports.model_provider_port import ModelProviderPort
from gateway.services.tools import WorkspaceCodeExecutionPolicies
from gateway.types.key_scope import KeyScope, KeyScopeSet

_NO_FILES_READ = KeyScopeSet.checked([KeyScope.INFERENCE_RUN], is_service_key=False)


def _key(scopes: KeyScopeSet) -> Any:
    return SimpleNamespace(
        id="key-1",
        user_id="user-1",
        allowed_models=None,
        exclude_from_budget=False,
        reject_user_mismatch=None,
        is_service_key=False,
        workspace_id=uuid.uuid4(),
        scopes=scopes,
    )


def test_missing_scope_refused_when_file_attached() -> None:
    with pytest.raises(HTTPException) as exc_info:
        pipeline._require_stored_file_scope(chat._ADAPTER, _key(_NO_FILES_READ), True)
    assert exc_info.value.status_code == 403


def test_scope_holder_and_no_attachment_pass() -> None:
    holder = KeyScopeSet.checked([KeyScope.INFERENCE_RUN, KeyScope.FILES_READ], is_service_key=False)
    pipeline._require_stored_file_scope(chat._ADAPTER, _key(holder), True)
    pipeline._require_stored_file_scope(chat._ADAPTER, _key(_NO_FILES_READ), False)
    pipeline._require_stored_file_scope(chat._ADAPTER, None, True)  # master key is unscoped


class _OrderProbe:
    """Fails the test the moment budget work runs after a refused scope check."""

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_verify(*args: Any, **kwargs: Any) -> tuple[Any, bool]:
            return _key(_NO_FILES_READ), False

        async def fake_reserve(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("reserve_budget ran: the stored-file refusal did not come first")

        monkeypatch.setattr(pipeline, "verify_api_key_or_master_key", fake_verify)
        monkeypatch.setattr(pipeline, "reserve_budget", fake_reserve)


@pytest.mark.asyncio
async def test_attach_refusal_precedes_the_reservation(monkeypatch: pytest.MonkeyPatch) -> None:
    _OrderProbe().install(monkeypatch)
    config = GatewayConfig(require_pricing=False)
    app = SimpleNamespace(
        state=SimpleNamespace(
            container=build_container(config=GatewayConfig()), unavailable_model_reporter=reporter_with_null_listener()
        )
    )
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": f"{API_ROOT}/chat/completions",
            "headers": [],
            "app": app,
        }
    )
    with pytest.raises(HTTPException) as exc_info:
        await pipeline.resolve_request_context(
            adapter=chat._ADAPTER,
            raw_request=request,
            response=Response(),
            db=cast(AsyncSession, object()),
            uow=cast(UnitOfWork, object()),
            config=config,
            log_writer=cast(Any, object()),
            model="openai:gpt-4",
            user_id_from_request=None,
            estimate_prompt_chars=10,
            estimate_max_output_tokens=None,
            master_key_user_required_detail="user required",
            user_forbidden_detail="forbidden",
            code_execution_policies=cast(WorkspaceCodeExecutionPolicies, object()),
            model_provider=cast(ModelProviderPort, object()),
            declared=pipeline.DeclaredTools(
                guardrails=None,
                guardrail_targets=tuple,
                tools=None,
                mcp_servers=None,
                mcp_server_ids=None,
                max_tool_iterations=None,
                tools_header=None,
            ),
            attaches_stored_file=True,
        )
    assert exc_info.value.status_code == 403
