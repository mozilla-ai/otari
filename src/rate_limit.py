"""Backward-compatible alias for legacy `rate_limit` imports in tests."""

import sys

from gateway import rate_limit as _rate_limit

sys.modules[__name__] = _rate_limit
