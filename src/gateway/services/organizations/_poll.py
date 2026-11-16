"""A background loop that clears a backlog of one database step and waits for the next one."""

import asyncio
from collections.abc import Awaitable, Callable

from gateway.core.database import DATABASE_ERRORS, describe_database_error
from gateway.core.unit_of_work import UnitOfWork, create_unit_of_work
from gateway.log_config import logger


async def poll(step: Callable[[UnitOfWork], Awaitable[bool]], *, interval: float, name: str) -> None:
    """Run ``step`` every ``interval`` seconds, on a new Unit of Work, until the task is canceled.

    A backlog already due is cleared before the next wait, rather than one item per wait.

    ``name`` identifies the loop in its log messages.
    An error is logged and the loop waits ``interval`` before trying again: a database error with its description.
    NOTE: Any other error is logged by its class only, because its message can hold user content.
    """
    while True:
        await asyncio.sleep(interval)
        try:
            again = True
            while again:
                async with create_unit_of_work() as uow:
                    again = await step(uow)
        except DATABASE_ERRORS as exc:
            logger.warning("%s poll failed: %s; retrying in %ss", name, describe_database_error(exc), interval)
        except Exception as exc:
            # NOTE: Only the class is logged, because the message of an error can hold user content.
            logger.warning("%s poll failed (%s); retrying in %ss", name, type(exc).__name__, interval)
