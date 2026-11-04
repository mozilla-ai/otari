"""The workspace's hosted provider rule, as dispatch and the catalog read it from the cache."""

import uuid

import pytest

from gateway.core.config import GatewayConfig
from gateway.services.model_access import organization_key_refusal
from gateway.services.providers import WorkspaceHostedAccessDeniedError, narrow_hosted_models
from gateway.services.providers import _hosted_name_index as hosted_name_index
from gateway.services.providers import _workspace_hosted_access as access
from gateway.services.providers._workspace_hosted_access import WorkspaceHostedRule, ensure_workspace_may_use_hosted

WORKSPACE = uuid.uuid4()
OTHER = uuid.uuid4()


@pytest.fixture(autouse=True)
def _rules(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        access,
        "_rules",
        {
            WORKSPACE: {
                "openai": WorkspaceHostedRule(enabled=True, models=frozenset({"gpt-4o"})),
                "mistral": WorkspaceHostedRule(enabled=False, models=None),
                "gemini": WorkspaceHostedRule(enabled=True, models=frozenset({"gemini-2.5-pro"})),
            }
        },
    )


def test_a_workspace_with_no_rule_inherits_everything() -> None:
    ensure_workspace_may_use_hosted(OTHER, "mistral", "mistral-large")
    ensure_workspace_may_use_hosted(None, "mistral", "mistral-large")


def test_a_provider_switched_off_is_refused_even_for_a_call_naming_no_model() -> None:
    with pytest.raises(WorkspaceHostedAccessDeniedError) as refused:
        ensure_workspace_may_use_hosted(WORKSPACE, "mistral", None)

    assert refused.value.workspace_id == WORKSPACE


def test_a_narrowed_provider_serves_only_its_models() -> None:
    ensure_workspace_may_use_hosted(WORKSPACE, "openai", "gpt-4o")
    ensure_workspace_may_use_hosted(WORKSPACE, "openai", None)
    with pytest.raises(WorkspaceHostedAccessDeniedError):
        ensure_workspace_may_use_hosted(WORKSPACE, "openai", "gpt-4o-mini")


def test_the_catalog_is_narrowed_the_way_dispatch_is() -> None:
    hosted = {
        "openai": frozenset({"gpt-4o", "gpt-4o-mini"}),
        "mistral": frozenset({"mistral-large"}),
        "gemini": None,
        "anthropic": None,
    }

    assert narrow_hosted_models(hosted, WORKSPACE) == {
        "openai": frozenset({"gpt-4o"}),
        # No roster of its own, so the workspace's list is the roster.
        "gemini": frozenset({"gemini-2.5-pro"}),
        "anthropic": None,
    }
    assert narrow_hosted_models(hosted, OTHER) == hosted
    assert narrow_hosted_models(hosted, None) == hosted


def test_requiring_zero_data_retention_refuses_a_configured_instance_and_an_undeclared_hosted_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured instance cannot be declared, so a workspace requiring it is refused it outright."""
    monkeypatch.setattr(access, "_rules", {})
    monkeypatch.setattr(access, "_zero_retention_workspaces", frozenset({WORKSPACE}))
    monkeypatch.setattr(hosted_name_index, "_zero_retention", frozenset({"gemini"}))
    config = GatewayConfig(master_key="test", providers={"local": {"provider": "openai", "api_key": "sk-local"}})

    def refusal(workspace_id: uuid.UUID) -> object:
        return organization_key_refusal(
            config, workspace_id, "local", "openai", "gpt-4o", selector="local:gpt-4o", through_alias=False
        )

    refused = refusal(WORKSPACE)
    assert refused is not None
    assert getattr(refused, "status_code", None) == 403
    assert refusal(OTHER) is None
    ensure_workspace_may_use_hosted(WORKSPACE, "gemini", "gemini-2.5-pro")
    with pytest.raises(WorkspaceHostedAccessDeniedError):
        ensure_workspace_may_use_hosted(WORKSPACE, "openai", "gpt-4o")
    assert narrow_hosted_models({"gemini": None, "openai": None}, WORKSPACE) == {"gemini": None}
