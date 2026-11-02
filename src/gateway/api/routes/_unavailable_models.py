"""Switch off a hosted model the moment its provider refuses to serve it, and tell the operators.

A provider can stop serving a model it still lists (Together moved models to
dedicated endpoints only, and its listing does not say so), and every request
for it then fails. The first request to meet the refusal on the deployment's
own credential switches the model off, so it leaves GET /models and stops being
tried, and the Health page's alert list gets one email to review it. An
organization's own key never switches anything: what its account may reach
says nothing about the deployment's.
"""

import asyncio

from gateway.api.deps import build_health_check_service, build_hosted_provider_service
from gateway.core.config import GatewayConfig
from gateway.core.database import DATABASE_ERRORS, describe_database_error
from gateway.core.unit_of_work import create_unit_of_work
from gateway.log_config import logger
from gateway.services.inference import is_model_unavailable_error, upstream_error_message
from gateway.services.upstream_redaction import redact_upstream_message

# Mailed to the alert list, so capped like a health check's error.
_REFUSAL_LIMIT = 300

# Switch-offs under way, one per model, held so one outliving its request is not garbage collected half done.
_PENDING: dict[tuple[str, str], asyncio.Task[None]] = {}


def _refusal(exc: BaseException) -> str:
    text = redact_upstream_message(upstream_error_message(exc)) or type(exc).__name__
    return text if len(text) <= _REFUSAL_LIMIT else text[: _REFUSAL_LIMIT - 1] + "…"


def note_unavailable_model(
    config: GatewayConfig, instance: str, model: str, exc: BaseException, *, deployment_paid: bool
) -> None:
    """Switch ``instance:model`` off in the background when ``exc`` is its provider refusing to serve it.

    Returns at once: the request that met the refusal is answered as it would
    have been, and the switch-off happens beside it.
    """
    key = (instance, model)
    if not deployment_paid or key in _PENDING or not is_model_unavailable_error(exc):
        return
    task = asyncio.create_task(_switch_off(config, instance, model, _refusal(exc)))
    _PENDING[key] = task
    task.add_done_callback(lambda _done: _PENDING.pop(key, None))


async def _switch_off(config: GatewayConfig, instance: str, model: str, refusal: str) -> None:
    selector = f"{instance}:{model}"
    try:
        async with create_unit_of_work() as uow:
            switched = await build_hosted_provider_service(uow, config=config).switch_off_unavailable(instance, model)
        if not switched:
            return
        logger.warning("Switched off hosted model %s: its provider refused to serve it", selector)
        async with create_unit_of_work() as uow:
            await build_health_check_service(uow, config).alert_unavailable(selector, refusal)
    except asyncio.CancelledError:
        raise
    except DATABASE_ERRORS as exc:
        logger.warning("Could not switch off unavailable model %s (%s)", selector, describe_database_error(exc))
    except Exception as exc:
        # A background task: nothing awaits it, so anything it raises ends here.
        logger.warning("Could not switch off unavailable model %s (%s)", selector, type(exc).__name__)
