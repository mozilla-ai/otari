"""Test doubles for the unavailable-model reporter and its listener."""

import asyncio

from gateway.services.inference import UnavailableModel, UnavailableModelReporter


class RecordingListener:
    def __init__(self) -> None:
        self.models: list[UnavailableModel] = []

    async def model_unavailable(self, unavailable_model: UnavailableModel) -> None:
        self.models.append(unavailable_model)


class _NullListener:
    async def model_unavailable(self, unavailable_model: UnavailableModel) -> None:
        pass


async def background_tasks_done() -> None:
    await asyncio.gather(*(asyncio.all_tasks() - {asyncio.current_task()}), return_exceptions=True)


def reporter_with_null_listener() -> UnavailableModelReporter:
    return UnavailableModelReporter(_NullListener())
