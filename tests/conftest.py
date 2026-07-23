import sys
from collections.abc import Generator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if "gateway" in sys.modules:
    del sys.modules["gateway"]


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
    from gateway.services.pricing_service import configure_default_pricing

    configure_default_pricing(False)
    reset_price_refresh_state()
    yield
    configure_default_pricing(False)
    reset_price_refresh_state()
