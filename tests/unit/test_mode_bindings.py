"""The code execution rules that differ by mode are chosen where the app is wired, not read by a service."""

from typing import cast

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_playground_tool_offer, get_workspace_listener
from gateway.core.config import GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.services.playground_service import ForwardedToolOffer, LocalToolOffer
from gateway.services.tenancy.workspace_listener import NullWorkspaceListener
from gateway.services.tools import CodeExecutionWorkspaceDefaults

# Stands in for a request's session; nothing here queries it.
A_SESSION = cast(AsyncSession, object())


def _hosted() -> GatewayConfig:
    return GatewayConfig(mode="hosted")


def test_a_hosted_control_plane_starts_each_new_workspace_with_code_execution_on() -> None:
    assert isinstance(get_workspace_listener(UnitOfWork(A_SESSION), _hosted()), CodeExecutionWorkspaceDefaults)


def test_a_standalone_deployment_stages_nothing_for_a_new_workspace() -> None:
    assert isinstance(get_workspace_listener(UnitOfWork(A_SESSION), GatewayConfig()), NullWorkspaceListener)


def test_a_hosted_control_plane_offers_the_playground_its_data_plane_s_tools() -> None:
    assert isinstance(get_playground_tool_offer(_hosted()), ForwardedToolOffer)


def test_a_standalone_deployment_offers_the_playground_its_own_tools() -> None:
    assert isinstance(get_playground_tool_offer(GatewayConfig()), LocalToolOffer)


@pytest.mark.parametrize(
    ("workspace_enabled", "enabled", "reason"),
    [
        (None, False, "Not turned on for this workspace."),
        (False, False, "Turned off for this workspace."),
        (True, True, None),
    ],
)
def test_a_forwarded_offer_reads_no_policy_as_off_whatever_the_local_sandbox(
    workspace_enabled: bool | None, enabled: bool, reason: str | None
) -> None:
    availability = ForwardedToolOffer(_hosted()).code_execution(workspace_enabled)

    assert (availability.configured, availability.enabled, availability.reason) == (True, enabled, reason)


def test_a_local_offer_reads_no_policy_as_no_narrowing() -> None:
    config = GatewayConfig(sandbox_url="http://127.0.0.1:9999/sandbox")

    assert LocalToolOffer(config).code_execution(None).enabled is True


def test_a_forwarded_offer_refuses_uploads_its_data_plane_could_not_read() -> None:
    files = ForwardedToolOffer(_hosted()).files()

    assert (files.configured, files.enabled) == (False, False)
