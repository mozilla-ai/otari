"""Build every stored guardrail once, at startup, instead of on a request.

Lazy building was never a preference. A profile used to be a key in a sidecar's
YAML, so this process could not name the profiles a deployment had, and the first
request to use one was the only thing that could ask for it to be built. A
definition is enumerable now, so the request that names a profile does not have to
be the one that pays for constructing it.

The lifespan starts this as a background task rather than awaiting it on the boot
path: a vendor SDK slow to import must not hold the port closed, and a definition
that will not build must not stop the gateway. What that costs is a short window
after boot in which a profile is not yet available, which the fail-open and
fail-closed rule already governs.

One shot, not a refresher, unlike the provider and search-tool caches whose shape
this otherwise resembles. There is nothing to converge on a TTL: a definition
changes through a write, and the write rebuilds what it changed.
"""

from __future__ import annotations

import asyncio
import time

from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import GatewayConfig
from gateway.core.database import create_session
from gateway.log_config import logger
from gateway.models.guardrails import GuardrailCredential
from gateway.services.guardrail_credential_service import definition_from_row, list_guardrail_credentials
from gateway.services.guardrail_runner import get_guardrail_runner
from gateway.services.guardrails import GuardrailsNotReachableError
from gateway.services.secret_box import SecretBoxUnavailableError, SecretDecryptionError
from gateway.types.guardrail_definition import GuardrailDefinition

# When a pass is worth a second log line. Not a ceiling: the pass is sequential and
# each build carries its own deadline, so what bounds it is the number of stored
# definitions, and nothing here should cut that set short and leave the profiles it
# skipped looking undefined. Nine construct in about a second, so a pass past this
# means something is slow enough to want naming.
_SLOW_PASS_S = 60.0


async def stored_definitions(db: AsyncSession) -> dict[str, GuardrailDefinition]:
    """Every enabled stored guardrail, keyed by the profile it answers to.

    A row whose secrets no longer decrypt is skipped rather than raised on: the
    rest of the deployment's guardrails are not that row's to take down, and the
    store already reports it as ``decryptable: false``.
    """
    definitions: dict[str, GuardrailDefinition] = {}
    for row in await list_guardrail_credentials(db):
        if not row.enabled:
            continue
        try:
            definitions[row.name] = definition_from_row(row)
        except (SecretBoxUnavailableError, SecretDecryptionError):
            logger.warning("Stored guardrail %r was not built: its secrets cannot be decrypted", row.name)
    return definitions


async def apply_stored_guardrail(row: GuardrailCredential) -> None:
    """Make the runner agree with one stored row, after the write that changed it.

    The three ways a row does not become a built guardrail are the three the pass
    above already applies, and they are here rather than at the two call sites so
    a write cannot disagree with a restart about what a row means: disabled is
    skipped, undecryptable is skipped, and anything else is built.

    Raises nothing. The write is committed either way, so the worst outcome is a
    profile that is merely cold, and the log line is what an operator acts on.
    """
    runner = get_guardrail_runner()
    if not row.enabled:
        # Dropped rather than left alone: the profile may have been enabled a
        # moment ago, and a restart would not bring it back.
        runner.drop(row.name)
        return

    try:
        definition = definition_from_row(row)
    except (SecretBoxUnavailableError, SecretDecryptionError):
        runner.drop(row.name)
        logger.warning("Stored guardrail %r was written but its secrets cannot be decrypted", row.name)
        return

    try:
        await runner.load_one(row.name, definition)
    except GuardrailsNotReachableError as exc:
        logger.warning("Stored guardrail %r was written but did not build: %s", row.name, exc)


async def load_stored_guardrails(config: GatewayConfig) -> None:
    """Hand every stored definition to the runner, and log how the pass went.

    Raises nothing. A database not ready at boot is the likely way this fails as a
    whole, and a task that died is reported at shutdown as one more thing to worry
    about, which this is not: every profile it did not build is one an operator can
    rebuild by saving it again or by testing it.
    """
    if config.is_hybrid_mode:
        return

    started = time.monotonic()
    try:
        async with create_session() as db:
            definitions = await stored_definitions(db)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning("Stored guardrails were not built at startup", exc_info=True)
        return

    if not definitions:
        return

    outcome = await get_guardrail_runner().load(definitions)
    elapsed = time.monotonic() - started
    if elapsed > _SLOW_PASS_S:
        logger.warning("Building %d stored guardrails took %.0fs", len(definitions), elapsed)
    logger.info("Built %d of %d stored guardrails", outcome.built, len(definitions))
