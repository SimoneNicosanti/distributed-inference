import asyncio
import logging
from contextlib import aclosing, suppress
from typing import override

from artifacts.storage.artifact_store import ArtifactStore
from blocks.lifecycle.async_lifecycle import AsyncLifecycle
from shared.model.model_variant import ModelVariantId
from shared.service.service import WorkerId
from worker.application.ports.outbound.profiling.model_execution.model_execution_profile_publisher import (
    ModelExecutionProfilePublisher,
)
from worker.application.ports.outbound.profiling.model_execution.model_static_profile_reader import (
    ModelStaticProfileReader,
)
from worker.application.ports.outbound.profiling.model_execution.pending_model_variant_consumer import (
    PendingModelVariantConsumer,
)
from worker.application.profiling.abc.model_execution_profiler import (
    ModelExecutionProfiler,
)
from worker.application.profiling.abc.model_execution_profiling_coordinator import (
    ModelExecutionProfilingCoordinator,
)
from worker.domain.profiling.model_execution.model_execution_profile import (
    ModelExecutionProfile,
)

logger = logging.getLogger(__name__)

EXECUTION_PROFILING_INTERVAL_S = 1 * 60


class DefaultModelExecutionProfilingCoordinator(
    ModelExecutionProfilingCoordinator, AsyncLifecycle
):
    def __init__(
        self,
        worker_id: WorkerId,
        pending_models_consumer: PendingModelVariantConsumer,
        model_static_profile_reader: ModelStaticProfileReader,
        model_execution_profiler: ModelExecutionProfiler,
        artifact_store: ArtifactStore,
        model_exec_profile_publisher: ModelExecutionProfilePublisher,
    ):
        self._worker_id = worker_id
        self._pending_models_consumer = pending_models_consumer
        self._model_static_profile_reader = model_static_profile_reader
        self._model_execution_profiler = model_execution_profiler
        self._artifact_store = artifact_store
        self._model_exec_profile_storage = model_exec_profile_publisher

        self._profile_loop_task: asyncio.Task[None] | None = None

        self._profiling_lock = asyncio.Lock()

    @override
    async def profile_model_execution(
        self, model_variant_id: ModelVariantId
    ) -> ModelExecutionProfile:
        model_static_profile = (
            await self._model_static_profile_reader.get_model_static_profile(
                model_variant_id=model_variant_id
            )
        )

        async with (
            self._profiling_lock,
            self._artifact_store.download_artifact(
                model_static_profile.artifact_ref
            ) as artifact_bundle,
        ):
            model_execution_profile = (
                await self._model_execution_profiler.profile_model_execution(
                    model_static_profile=model_static_profile,
                    artifact_bundle=artifact_bundle,
                )
            )

            return model_execution_profile

    async def _coordinate_model_execution_profiling(
        self, model_variant_id: ModelVariantId
    ) -> None:
        try:
            model_execution_profile = await self.profile_model_execution(
                model_variant_id=model_variant_id
            )
            await self._model_exec_profile_storage.puplish_model_execution_profile(
                worker_id=self._worker_id,
                model_execution_profile=model_execution_profile,
            )
        except Exception:
            logger.exception(
                "Model execution profiling or publication failed: worker=%s, model=%s",
                self._worker_id,
                model_variant_id,
            )
            return

        try:
            await self._pending_models_consumer.acknowledge(model_variant_id)
        except Exception:
            logger.exception(
                "Error while acknowledging success profile: worker=%s, model=%s",
                self._worker_id,
                model_variant_id,
            )

    async def _profile_loop(self) -> None:
        while True:
            try:
                async with aclosing(
                    self._pending_models_consumer.pending_model_variant_ids()
                ) as pending_model_ids:
                    async for pending_model_id in pending_model_ids:
                        await self._coordinate_model_execution_profiling(
                            model_variant_id=pending_model_id
                        )
            except Exception:
                logger.exception(
                    "Execution profiling round failed",
                )

            await asyncio.sleep(EXECUTION_PROFILING_INTERVAL_S)

    @override
    async def start(self) -> None:
        if self._profile_loop_task is not None:
            return
        self._profile_loop_task = asyncio.create_task(self._profile_loop())

    @override
    async def stop(self) -> None:
        if self._profile_loop_task is None:
            return
        self._profile_loop_task.cancel()

        with suppress(asyncio.CancelledError):
            await self._profile_loop_task

        self._profile_loop_task = None
