"""Environment-variable helpers.

User-facing environment variables use the ``OTARI_`` prefix.
"""

import os
from typing import overload

ENV_PREFIX = "OTARI_"


@overload
def otari_env(suffix: str, default: str) -> str: ...


@overload
def otari_env(suffix: str, default: None = None) -> str | None: ...


def otari_env(suffix: str, default: str | None = None) -> str | None:
    """Read ``OTARI_<suffix>``, returning ``default`` when it is not set."""
    value = os.environ.get(f"{ENV_PREFIX}{suffix}")
    if value is not None:
        return value
    return default
