"""A model its provider refuses to serve, and who is told of it."""

import asyncio
from dataclasses import dataclass
from typing import Protocol, Self

from gateway.core.database import DATABASE_ERRORS, describe_database_error
from gateway.log_config import logger
from gateway.services.inference._upstream_errors import is_model_unavailable_error, upstream_error_message


@dataclass(frozen=True)
class UnavailableModel:
    """A model its provider refused to serve.

    ``refusal`` is the provider's own wording.
    ``uses_deployment_credential`` is true when the refused call used the deployment's own credential.
    """

    instance: str
    model: str
    refusal: str
    uses_deployment_credential: bool

    @classmethod
    def from_failure(
        cls, exc: BaseException, *, instance: str, model: str, uses_deployment_credential: bool
    ) -> Self | None:
        """The model ``exc`` says its provider refused to serve, or ``None`` when ``exc`` is any other failure."""
        if not is_model_unavailable_error(exc):
            return None
        refusal = upstream_error_message(exc) or type(exc).__name__
        return cls(instance, model, refusal, uses_deployment_credential)


class ProviderRefusalListener(Protocol):
    """Reacts to a provider refusing to serve a model."""

    async def model_unavailable(self, unavailable_model: UnavailableModel) -> None:
        """React to the ``unavailable_model``."""
        ...


class UnavailableModelReporter:
    """Tells its listener when a provider failure is the provider refusing to serve the model.

    The request that met the refusal does not wait for the listener, and a listener that fails does not reach it.
    """

    def __init__(self, listener: ProviderRefusalListener) -> None:
        self._listener = listener
        # Held so a task that outlives its request is not garbage collected half done.
        self._pending: dict[tuple[str, str, bool], asyncio.Task[None]] = {}

    def report(self, exc: BaseException, *, instance: str, model: str, uses_deployment_credential: bool) -> None:
        """Tell the listener when ``exc`` is the provider refusing to serve ``model``.

        Any other failure is ignored, and so is a refusal the listener is still being told of.
        """
        key = (instance, model, uses_deployment_credential)
        if key in self._pending:
            return
        unavailable_model = UnavailableModel.from_failure(
            exc, instance=instance, model=model, uses_deployment_credential=uses_deployment_credential
        )
        if unavailable_model is None:
            return
        task = asyncio.create_task(self._tell(unavailable_model))
        self._pending[key] = task
        task.add_done_callback(lambda _done: self._pending.pop(key, None))

    async def _tell(self, unavailable_model: UnavailableModel) -> None:
        selector = f"{unavailable_model.instance}:{unavailable_model.model}"
        try:
            await self._listener.model_unavailable(unavailable_model)
        except DATABASE_ERRORS as exc:
            # codeql[py/clear-text-logging-sensitive-data]
            logger.warning("Could not act on unavailable model %s (%s)", selector, describe_database_error(exc))
        except Exception as exc:
            # Nothing awaits this task, so anything it raises ends here.
            # codeql[py/clear-text-logging-sensitive-data]
            logger.warning("Could not act on unavailable model %s (%s)", selector, type(exc).__name__)
