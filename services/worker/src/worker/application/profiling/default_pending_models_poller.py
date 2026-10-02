import asyncio
from collections.abc import AsyncGenerator
from typing import override

from lifecycle.async_lifecycle import AsyncLifecycle
from shared.identifiers.identifiers import WorkerId
from shared.model.model_version import ModelVersionId
from worker.application.ports.outbound.profiling.model_execution.model_execution_profile_store import (
    ModelExecutionProfileStore,
)
from worker.application.ports.outbound.profiling.model_execution.model_metadata_store import (
    ModelMetadataStore,
)
from worker.application.profiling.contracts.pending_models_poller import (
    PendingModelsPoller,
)

POLLING_TIME_INTERVAL_S = 1 * 60  # 1 minute


class DefaultPendingModelsPoller(PendingModelsPoller, AsyncLifecycle):
    def __init__(
        self,
        worker_id: WorkerId,
        model_profile_store: ModelMetadataStore,
        model_execution_profile_store: ModelExecutionProfileStore,
    ) -> None:

        self._worker_id = worker_id
        self._model_profile_store = model_profile_store
        self._model_execution_profile_store = model_execution_profile_store

        self._poll_task: asyncio.Task[None] | None = None

        self._poll_queue: asyncio.Queue[ModelVersionId] = asyncio.Queue(maxsize=64)
        self._scheduled_model_ids: set[ModelVersionId] = set()

    @override
    async def pending_models(self) -> AsyncGenerator[ModelVersionId]:
        while True:
            model_id = await self._poll_queue.get()

            try:
                # Queue contents can become stale.
                already_profiled = await self._model_execution_profile_store.check_model_execution_profile_exists(
                    self._worker_id,
                    model_id,
                )

                if not already_profiled:
                    yield model_id
            finally:
                self._scheduled_model_ids.discard(model_id)
                self._poll_queue.task_done()

    async def _poll_once(self) -> None:
        all_ids, profiled_ids = await asyncio.gather(
            self._model_profile_store.get_all_model_version_ids(),
            self._model_execution_profile_store.get_all_profiled_model_ids_by_worker_id(
                self._worker_id
            ),
        )

        profiled = set(profiled_ids)

        for model_id in all_ids:
            if model_id in profiled:
                continue

            if model_id in self._scheduled_model_ids:
                continue

            self._scheduled_model_ids.add(model_id)

            try:
                await self._poll_queue.put(model_id)
            except BaseException:
                self._scheduled_model_ids.discard(model_id)
                raise

    async def _poll_loop(self) -> None:
        while True:
            await self._poll_once()
            await asyncio.sleep(POLLING_TIME_INTERVAL_S)

    @override
    async def start(self) -> None:
        async with asyncio.TaskGroup() as task_group:
            self._poll_task = task_group.create_task(self._poll_loop())

    @override
    async def stop(self) -> None:
        if self._poll_task is None:
            return
        self._poll_task.cancel()
