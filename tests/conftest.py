import sys
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if "gateway" in sys.modules:
    del sys.modules["gateway"]


@pytest.fixture(autouse=True)
def _no_background_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop the app lifespan from dialing providers and models.dev for real.

    Every ``TestClient(app)`` runs the lifespan, which starts the discovery and
    catalog refreshers; their first act is to prime the cache from a live dial.
    That is right in production and wrong in a test: it makes real outbound calls
    from every app boot, and it races a test that patches the dial *after*
    startup, so the read then serves whatever the unpatched prime cached.

    Suppressing the refreshers leaves the cache empty, so a read takes the
    cold-provider path and dials once, under whatever the test has patched.
    A test that wants the warm-cache read path seeds the cache itself.

    Lives in the root conftest, not the integration one, because the unit suite
    builds apps too (``tests/unit/test_gateway_root_page.py``,
    ``test_tools_endpoint.py``, ``test_settings_endpoint.py`` and others all use
    ``TestClient(create_app(...))``). Scoping this to ``tests/integration`` left
    every one of those making live models.dev fetches on CI, which is both wrong
    on its own terms and what surfaced the unbounded-shutdown bug that
    ``_stop_refresher`` now guards against. A test that wants the real refresher
    calls it directly rather than through the lifespan.
    """

    async def _noop(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr("gateway.main.run_discovery_refresher", _noop)
    monkeypatch.setattr("gateway.main.run_catalog_refresher", _noop)


@pytest.fixture(autouse=True)
def _reset_default_pricing() -> Generator[None, None, None]:
    """Restore process-wide pricing state to its default before each test.

    ``configure_default_pricing`` is set at app startup, so a test that builds an
    app with a different ``default_pricing`` would otherwise leak that state into
    later tests that call ``find_model_pricing`` directly. Reset to off, matching
    the config field's opt-in default; tests that need defaults enable explicitly.

    Also clear the memoized genai-prices resolutions so a real price cached by one
    test cannot mask another test that patches ``calc_price`` to fail.
    """
    from gateway.services.pricing_refresh_service import reset_price_refresh_state
    from gateway.services.pricing_service import configure_default_pricing, configure_provider_types

    configure_default_pricing(False)
    configure_provider_types(None)
    reset_price_refresh_state()
    yield
    configure_default_pricing(False)
    configure_provider_types(None)
    reset_price_refresh_state()
