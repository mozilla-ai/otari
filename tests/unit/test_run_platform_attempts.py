"""Unit tests for the ``run_platform_attempts`` runner.

The runner is exercised end-to-end through the platform-mode integration
tests; this file covers narrow defensive paths that are awkward to provoke
through a full request, such as the empty-attempts guard.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException

from gateway.api.routes._platform import ResolvedRoute, run_platform_attempts


@pytest.mark.asyncio
async def test_empty_attempts_raises_500_with_explicit_diagnostic() -> None:
    """A caller that hands the runner an empty ``attempts`` list is in a
    programming-error state — the route handler should have raised a 502
    "no resolvable provider" before reaching the runner. The runner surfaces
    the bug as a 500 with a clear message rather than falling through to the
    terminal "all upstream providers failed" path (which would carry a
    misleading ``last_exc=None``).
    """
    route = ResolvedRoute(request_id="test", fallback_enabled=False, attempts=[])

    async def _never_called(_kwargs: dict[str, Any], _on_first_response: Any) -> Any:
        raise AssertionError("run_attempt must not be called when attempts is empty")

    with pytest.raises(HTTPException) as ei:
        await run_platform_attempts(
            route=route,
            attempts=[],
            base_request_fields={},
            run_attempt=_never_called,
            extract_usage=lambda _r: None,
            classify_error=lambda _exc: (False, "unknown"),
            report_attempt_outcome=lambda *_args: None,
            on_success=lambda _attempt: None,
            max_tool_iterations=1,
        )
    assert ei.value.status_code == 500
    assert "empty attempts list" in ei.value.detail
