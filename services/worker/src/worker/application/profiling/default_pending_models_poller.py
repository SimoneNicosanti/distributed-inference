import asyncio
from collections.abc import AsyncGenerator
from contextlib import suppress
from typing import override

from lifecycle.async_lifecycle import AsyncLifecycle
from shared.model.model_variant import ModelVariantId
from shared.service.service import WorkerId
from worker.application.ports.outbound.profiling.model_execution.model_execution_profile_reader import (
    ModelExecutionProfileReader,
)
from worker.application.ports.outbound.profiling.model_execution.model_static_profile_reader import (
    ModelStaticProfileReader,
)
from worker.application.profiling.abc.pending_models_poller import (
    PendingModelsPoller,
)

POLLING_TIME_INTERVAL_S = 1 * 60  # 1 minute


class DefaultPendingModelsPoller(PendingModelsPoller, AsyncLifecycle):
    def __init__(
        self,
        worker_id: WorkerId,
        model_static_profile_reader: ModelStaticProfileReader,
        model_execution_profile_reader: ModelExecutionProfileReader,
    ) -> None:

        self._worker_id = worker_id
        self._model_static_profile_reader = model_static_profile_reader
        self._model_execution_profile_reader = model_execution_profile_reader

        self._poll_task: asyncio.Task[None] | None = None

        self._poll_queue: asyncio.Queue[ModelVariantId] = asyncio.Queue(maxsize=64)
        self._scheduled_model_ids: set[ModelVariantId] = set()

    @override
    async def pending_model_ids(self) -> AsyncGenerator[ModelVariantId]:
        while True:
            model_id = await self._poll_queue.get()

            try:
                # Queue contents can become stale.
                already_profiled = await self._model_execution_profile_reader.check_model_execution_profile_exists(
                    self._worker_id,
                    model_id,
                )

                if not already_profiled:
                    yield model_id
            finally:
                self._scheduled_model_ids.discard(model_id)
                self._poll_queue.task_done()

    async def _poll_once(self) -> None:
        available_ids, profiled_ids = await asyncio.gather(
            self._model_static_profile_reader.get_all_model_static_profile_ids(),
            self._model_execution_profile_reader.get_all_profiled_model_ids_by_worker_id(
                self._worker_id
            ),
        )

        pending_ids = set(available_ids) - set(profiled_ids) - self._scheduled_model_ids

        for model_variant_id in pending_ids:
            self._scheduled_model_ids.add(model_variant_id)

            try:
                await self._poll_queue.put(model_variant_id)
            except BaseException:
                self._scheduled_model_ids.discard(model_variant_id)
                raise

    async def _poll_loop(self) -> None:
        while True:
            await self._poll_once()
            await asyncio.sleep(POLLING_TIME_INTERVAL_S)

    @override
    async def start(self) -> None:
        if self._poll_task is not None:
            return
        self._poll_task = asyncio.create_task(self._poll_loop())

    @override
    async def stop(self) -> None:
        if self._poll_task is None:
            return
        self._poll_task.cancel()

        with suppress(asyncio.CancelledError):
            await self._poll_task
