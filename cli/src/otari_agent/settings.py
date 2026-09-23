"""The three settings `otari hook` needs, read without the gateway's config stack.

`gateway.core.config.load_config` imports any-llm, pydantic-settings and the
SQLModel tables, most of the dependency tree this distribution exists to leave
behind. The hook reads host, port and master_key and nothing else, so this
mirrors only the slice of load_config that produces them: the `.env` order of
`_load_dotenv` (the config file's directory first, then the current directory,
neither overriding a variable already in the environment), the top-level YAML
keys, and the `OTARI_<FIELD>` override, where a non-empty variable beats YAML
and an empty one counts as unset.

Not mirrored, on purpose: OTARI_CONFIG_YAML and OTARI_CONFIG_B64, `${VAR}`
interpolation inside YAML values, and mode validation. A deployment that relies
on any of them passes --url and --api-key (or OTARI_URL and OTARI_API_KEY) to
the hook instead.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Mirror gateway.core.config.API_KEY_HEADER and API_ROOT. Literals rather than
# imports: this process talks to a gateway over HTTP that may be a different
# build. tests/unit/test_otari_agent_cli.py pins them equal.
API_KEY_HEADER = "Otari-Key"
API_ROOT = "/api/v1"

_ENV_PREFIX = "OTARI_"
# The gateway's bind default; the hook dials localhost when it sees it.
_DEFAULT_HOST = "0.0.0.0"  # noqa: S104
_DEFAULT_PORT = 8000
_FIELDS = ("host", "port", "master_key")


@dataclass(frozen=True)
class HookSettings:
    """Where a gateway listens and what authenticates a hook check against it."""

    host: str = _DEFAULT_HOST
    port: int = _DEFAULT_PORT
    master_key: str | None = None


def _load_dotenv(config_path: str | None) -> None:
    candidates = [Path.cwd() / ".env"]
    if config_path:
        candidates.insert(0, Path(config_path).resolve().parent / ".env")
    seen: set[Path] = set()
    for dotenv_path in candidates:
        if dotenv_path in seen or not dotenv_path.exists():
            continue
        seen.add(dotenv_path)
        load_dotenv(dotenv_path=dotenv_path, override=False)


def load_settings(config_path: str | None = None) -> HookSettings:
    """Resolve host, port and master_key the way `otari serve` would for the same inputs.

    Raises ValueError for a config file that cannot be read or is not a mapping,
    and for a port that is not an integer; `otari hook` reports either and fails open.
    """
    _load_dotenv(config_path)
    values: dict[str, object] = {}
    if config_path:
        try:
            loaded = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError(f"could not read {config_path}: {exc}") from exc
        if loaded is not None and not isinstance(loaded, dict):
            raise ValueError(f"{config_path} must hold a YAML mapping at the top level")
        values = {field: loaded[field] for field in _FIELDS if loaded and field in loaded}
    for field in _FIELDS:
        env_value = os.environ.get(f"{_ENV_PREFIX}{field.upper()}")
        if env_value:
            values[field] = env_value

    port = values.get("port")
    try:
        resolved_port = int(str(port)) if port is not None else _DEFAULT_PORT
    except ValueError as exc:
        raise ValueError(f"port must be an integer, got {port!r}") from exc
    host = values.get("host")
    master_key = values.get("master_key")
    return HookSettings(
        host=str(host) if host else _DEFAULT_HOST,
        port=resolved_port,
        master_key=str(master_key) if master_key else None,
    )
