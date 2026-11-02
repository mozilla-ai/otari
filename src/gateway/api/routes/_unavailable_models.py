"""Switch off a hosted model the moment its provider refuses to serve it, beside the request that met the refusal.

A provider can stop serving a model it still lists (Together moved models to
dedicated endpoints only, and its listing does not say so), and every request
for it then fails. The health checks service switches the model off and tells
the operators; this starts it without holding up the answer.
"""

import asyncio

from gateway.api.deps import build_health_check_service
from gateway.core.config import GatewayConfig
from gateway.core.database import DATABASE_ERRORS, describe_database_error
from gateway.core.unit_of_work import create_unit_of_work
from gateway.log_config import logger
from gateway.services.health_checks import refusal_switches_off
from gateway.services.inference import upstream_error_message

# Switch-offs under way, one per model, held so one outliving its request is not garbage collected half done.
_PENDING: dict[tuple[str, str], asyncio.Task[None]] = {}


def note_unavailable_model(
    config: GatewayConfig, instance: str, model: str, exc: BaseException, *, deployment_paid: bool
) -> None:
    """Switch ``instance:model`` off in the background when ``exc`` is its provider refusing the deployment's key.

    Returns at once: the request that met the refusal is answered as it would
    have been, and the switch-off happens beside it.
    """
    key = (instance, model)
    if key in _PENDING or not refusal_switches_off(exc, deployment_paid=deployment_paid):
        return
    refusal = upstream_error_message(exc) or type(exc).__name__
    task = asyncio.create_task(_switch_off(config, instance, model, refusal))
    _PENDING[key] = task
    task.add_done_callback(lambda _done: _PENDING.pop(key, None))


async def _switch_off(config: GatewayConfig, instance: str, model: str, refusal: str) -> None:
    selector = f"{instance}:{model}"
    try:
        async with create_unit_of_work() as uow:
            await build_health_check_service(uow, config).switch_off_unavailable(instance, model, refusal)
    except asyncio.CancelledError:
        raise
    except DATABASE_ERRORS as exc:
        # codeql[py/clear-text-logging-sensitive-data]
        logger.warning("Could not switch off unavailable model %s (%s)", selector, describe_database_error(exc))
    except Exception as exc:
        # A background task: nothing awaits it, so anything it raises ends here.
        # codeql[py/clear-text-logging-sensitive-data]
        logger.warning("Could not switch off unavailable model %s (%s)", selector, type(exc).__name__)
