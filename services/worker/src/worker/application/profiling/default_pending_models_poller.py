import asyncio
from collections.abc import AsyncGenerator
from typing import override

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


class DefaultPendingModelsPoller(PendingModelsPoller):
    def __init__(
        self,
        worker_id: WorkerId,
        model_static_profile_reader: ModelStaticProfileReader,
        model_execution_profile_reader: ModelExecutionProfileReader,
    ) -> None:

        self._worker_id = worker_id
        self._model_static_profile_reader = model_static_profile_reader
        self._model_execution_profile_reader = model_execution_profile_reader

    @override
    async def pending_model_ids(self) -> AsyncGenerator[ModelVariantId]:
        while True:
            available_ids, profiled_ids = await asyncio.gather(
                self._model_static_profile_reader.get_all_model_static_profile_ids(),
                self._model_execution_profile_reader.get_all_profiled_model_ids_by_worker_id(
                    self._worker_id
                ),
            )
            pending_ids = set(available_ids) - set(profiled_ids)

            if not pending_ids:
                await asyncio.sleep(POLLING_TIME_INTERVAL_S)
                continue

            for model_id in pending_ids:
                already_profiled = await self._model_execution_profile_reader.check_model_execution_profile_exists(
                    self._worker_id, model_id
                )
                if not already_profiled:
                    yield model_id
