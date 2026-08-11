"""The shared gate for tests that need the built dashboard bundle.

The bundle is built, not committed, so these tests have nothing to run against in
`otari-tests.yml` and skip themselves there. The only job that builds a bundle
selects them by marker (`-m dashboard_bundle` in `otari-dashboard-serving.yml`),
so the skip and the marker have to travel together: a test carrying only the skip
would run nowhere at all, skipped in one workflow and unselected in the other,
and report green while testing nothing. Applying both from one decorator makes
that pairing impossible to get half right, wherever the test lives.
"""

from collections.abc import Callable
from typing import TypeVar, cast

import pytest

from gateway.dashboard import get_dashboard_dir

F = TypeVar("F", bound=Callable[..., object])

_skip_without_bundle = pytest.mark.skipif(
    get_dashboard_dir() is None,
    reason="dashboard bundle not built (run: make dashboard)",
)


def requires_dashboard_bundle(test: F) -> F:
    """Mark a test as needing the built bundle, and skip it when there is none."""
    return cast(F, _skip_without_bundle(pytest.mark.dashboard_bundle(test)))
