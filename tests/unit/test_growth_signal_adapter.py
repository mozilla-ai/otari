"""The core ``GrowthSignalPort`` adapter: a build with no vendor schedules nothing."""

import uuid
from typing import cast

import pytest
from fastapi import BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.adapters.growth_signal_adapter import NullGrowthSignalAdapter
from gateway.ports.growth_signal_port import GrowthAction

NO_SESSION = cast(AsyncSession, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("action", list(GrowthAction))
async def test_an_action_schedules_nothing(action: GrowthAction) -> None:
    background_tasks = BackgroundTasks()

    await NullGrowthSignalAdapter(NO_SESSION).record_action(
        background_tasks=background_tasks,
        action=action,
        user_id=uuid.uuid4(),
        email=None,
        organization_id=uuid.uuid4(),
        details={"provider": "openai"},
    )

    assert background_tasks.tasks == []


def test_the_action_names_are_the_ones_an_adapter_maps() -> None:
    # Pinned as strings: an overlay keys its vendor's event names on these, so a rename splits its history.
    assert [action.value for action in GrowthAction] == [
        "api_key_created",
        "guardrail_created",
        "member_invited",
        "provider_key_added",
        "mcp_server_created",
        "web_search_key_added",
        "routing_policy_created",
        "budget_created",
    ]
