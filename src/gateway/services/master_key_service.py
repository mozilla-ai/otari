"""First-run master-key bootstrap.

When no master key is configured (no ``OTARI_MASTER_KEY`` / config value), the
gateway generates one on first launch, stores only its SHA-256 hash, and shows
it once, inside a setup link, so the operator can open the dashboard and set a
password. This makes
"launch with almost nothing" real without leaving the management API
unauthenticated: the generated key gates every management route exactly like an
operator-set one, and there is no unauthenticated setup route to race.

An operator-provided key always wins and is never generated over. The hash is
kept in the ``runtime_settings`` key/value table (the flag logic ignores keys it
does not know), so no new table is needed. The plaintext is never stored or
logged after that one-time notice.
"""

import hashlib
import secrets
from typing import Any, cast

from sqlalchemy import update
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.platform import RuntimeSetting

# Stored in runtime_settings; ignored by runtime_settings_service (not a SETTABLE_KEY).
MASTER_KEY_HASH_KEY = "master_key_hash"
_MASTER_KEY_PREFIX = "otari-mk-"


class MasterKeyRotationConflictError(RuntimeError):
    """Raised when another request rotates the generated master key first."""


def generate_master_key() -> str:
    """Return a fresh, high-entropy master key with a recognizable prefix."""
    return f"{_MASTER_KEY_PREFIX}{secrets.token_urlsafe(32)}"


def is_generated_master_key(token: str) -> bool:
    """Whether a token has the format reserved for generated master keys."""
    return token.startswith(_MASTER_KEY_PREFIX)


def hash_master_key(token: str) -> str:
    """SHA-256 hex of a master key.

    Deliberately not ``auth.hash_key``: a master key is a different credential
    with its own shape and its own storage, so the two hashes stay independent.
    """
    return hashlib.sha256(token.encode()).hexdigest()


async def load_master_key_hash(session: AsyncSession) -> str | None:
    """Load the generated master-key hash persisted for all gateway replicas."""
    row = await session.get(RuntimeSetting, MASTER_KEY_HASH_KEY)
    return row.value if row else None


async def ensure_master_key(config: GatewayConfig, session: AsyncSession) -> str | None:
    """Make sure a master key exists, and return the key when this call generated it.

    - Operator-set key (``config.master_key``): used as-is, nothing generated.
    - Previously generated key: its hash is loaded so the stored key keeps working
      across restarts (idempotent, nothing returned).
    - Neither: generate one, persist its hash, enable the hash-compare auth
      path, and return the key for the caller to show once.

    A persistence failure is logged, not raised: the gateway still serves
    inference; only the management API stays locked until a key is available.
    """
    if config.master_key:
        return None
    try:
        existing = await load_master_key_hash(session)
        if existing:
            config._master_key_hash = existing
            return None
        token = generate_master_key()
        hashed = hash_master_key(token)
        session.add(RuntimeSetting(key=MASTER_KEY_HASH_KEY, value=hashed))
        await session.commit()
        config._master_key_hash = hashed
        return token
    except SQLAlchemyError:
        await session.rollback()
        # With several workers/replicas on a shared DB, all of them see no hash
        # on first run and race to INSERT; the losers land here on the duplicate
        # primary key. Adopt the winner's hash instead of returning with the
        # management API locked at 503 (and never show a second, dead key).
        existing = await load_master_key_hash(session)
        if existing:
            config._master_key_hash = existing
            return None
        logger.warning("Could not persist a generated master key; the management API stays locked until one is set.")
        return None


async def stage_generated_master_key_rotation(session: AsyncSession, current_hash: str) -> tuple[str, str]:
    """Stage a new generated master-key hash and return ``(token, hash)``.

    The plaintext token is returned exactly once for the dashboard to reveal.
    Callers must commit the session before updating the in-memory config hash so
    a failed write never invalidates the current key in this worker. The update
    compares against the authenticated key's hash, so two concurrent rotations
    cannot both return a key.
    """
    token = generate_master_key()
    hashed = hash_master_key(token)
    result = cast(
        CursorResult[Any],
        await session.execute(
            update(RuntimeSetting)
            .where(RuntimeSetting.key == MASTER_KEY_HASH_KEY, RuntimeSetting.value == current_hash)
            .values(value=hashed)
        ),
    )
    if result.rowcount != 1:
        raise MasterKeyRotationConflictError("The master key was already rotated. Reload and try again.")
    return token, hashed


def interface_url(config: GatewayConfig) -> str:
    """Where a browser on the operator's machine reaches this deployment, with no trailing slash."""
    if config.effective_ui_base_url:
        return config.ui_link("")
    host = "localhost" if config.host in ("0.0.0.0", "::", "") else config.host
    return f"http://{host}:{config.port}"


def setup_link(config: GatewayConfig, token: str) -> str:
    """The dashboard link that carries a generated master key to its setup page.

    The key rides in the fragment, which a browser keeps to itself: it reaches
    no server, access log or ``Referer`` header. The page takes it from there and
    asks only for the password that claims the deployment.
    """
    path = f"/#/setup?key={token}"
    if config.effective_ui_base_url:
        return config.ui_link(path)
    return f"{interface_url(config)}{path}"


def startup_notice(config: GatewayConfig, version: str, generated_key: str | None) -> list[str]:
    """What a start says once it is ready: where it runs, and how to get in.

    A start that generated a master key gives the setup link, on a line of its
    own so it copies cleanly; it is the only place that key is ever shown. Any
    other start says where to sign in.
    """
    running = f"Otari {version} is running at {interface_url(config)}"
    if generated_key is None:
        return [running, f"Sign in at {interface_url(config)}/"]
    return [
        "",
        running,
        "",
        "Open this link to choose your password and finish setup:",
        "",
        setup_link(config, generated_key),
        "",
        "It is shown only once and carries your master key, so treat it like one.",
        "",
    ]
