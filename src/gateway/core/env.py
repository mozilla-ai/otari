"""Environment-variable helpers.

User-facing environment variables use the ``OTARI_`` prefix. The ``GATEWAY_``
prefix is the legacy name, kept working for backward compatibility (it predates
the rename to Otari). When both are set, ``OTARI_`` takes precedence.
"""

import os
from typing import overload

ENV_PREFIX = "OTARI_"
LEGACY_ENV_PREFIX = "GATEWAY_"


@overload
def otari_env(suffix: str, default: str) -> str: ...


@overload
def otari_env(suffix: str, default: None = None) -> str | None: ...


def otari_env(suffix: str, default: str | None = None) -> str | None:
    """Read ``OTARI_<suffix>``, falling back to the legacy ``GATEWAY_<suffix>``.

    Returns ``default`` when neither is set. ``OTARI_`` wins when both are set.
    """
    value = os.environ.get(f"{ENV_PREFIX}{suffix}")
    if value is not None:
        return value
    value = os.environ.get(f"{LEGACY_ENV_PREFIX}{suffix}")
    if value is not None:
        return value
    return default
