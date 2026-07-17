"""First-run master-key bootstrap.

When no master key is configured (no ``OTARI_MASTER_KEY`` / config value), the
gateway generates one on first launch, stores only its SHA-256 hash, and prints
the plaintext once so the operator can sign in to the dashboard. This makes
"launch with almost nothing" real without leaving the management API
unauthenticated: the generated key gates every management route exactly like an
operator-set one, and there is no unauthenticated setup route to race.

An operator-provided key always wins and is never generated over. The hash is
kept in the ``runtime_settings`` key/value table (the flag logic ignores keys it
does not know), so no new table is needed. The plaintext is never stored or
logged after the one-time banner.
"""

import hashlib
import secrets

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import GatewayConfig
from gateway.log_config import log_secret, logger
from gateway.models.entities import RuntimeSetting

# Stored in runtime_settings; ignored by runtime_settings_service (not a SETTABLE_KEY).
MASTER_KEY_HASH_KEY = "master_key_hash"
_MASTER_KEY_PREFIX = "otari-mk-"


def generate_master_key() -> str:
    """Return a fresh, high-entropy master key with a recognizable prefix."""
    return f"{_MASTER_KEY_PREFIX}{secrets.token_urlsafe(32)}"


def hash_master_key(token: str) -> str:
    """SHA-256 hex of a master key.

    Deliberately not ``auth.hash_key``, which validates the ``gw-`` API-key
    format; a master key has its own shape.
    """
    return hashlib.sha256(token.encode()).hexdigest()


async def _load_hash(session: AsyncSession) -> str | None:
    row = await session.get(RuntimeSetting, MASTER_KEY_HASH_KEY)
    return row.value if row else None


async def ensure_master_key(config: GatewayConfig, session: AsyncSession) -> None:
    """Make sure a master key exists, generating and printing one on first run.

    - Operator-set key (``config.master_key``): used as-is, nothing generated.
    - Previously generated key: its hash is loaded so the stored key keeps working
      across restarts (idempotent, no second banner).
    - Neither: generate one, persist its hash, print it once, and enable the
      hash-compare auth path.

    A persistence failure is logged, not raised: the gateway still serves
    inference; only the management API stays locked until a key is available.
    """
    if config.master_key:
        return
    try:
        existing = await _load_hash(session)
        if existing:
            config._master_key_hash = existing
            return
        token = generate_master_key()
        session.add(RuntimeSetting(key=MASTER_KEY_HASH_KEY, value=hash_master_key(token)))
        await session.commit()
        config._master_key_hash = hash_master_key(token)
        _print_banner(config, token)
    except SQLAlchemyError:
        await session.rollback()
        logger.warning("Could not persist a generated master key; the management API stays locked until one is set.")


def _print_banner(config: GatewayConfig, token: str) -> None:
    host = "localhost" if config.host in ("0.0.0.0", "::", "") else config.host
    url = f"http://{host}:{config.port}/"
    bar = "=" * 64
    logger.warning(bar)
    logger.warning("Otari first-run: no master key was set, so one was generated.")
    logger.warning("Save it now (it is shown only once). Sign in to the dashboard at %s", url)
    log_secret("Your master key:", token)
    logger.warning("To choose your own instead, set OTARI_MASTER_KEY and restart.")
    logger.warning(bar)
