import asyncio
from contextlib import aclosing, suppress
from typing import override

from artifacts.storage.artifact_store import ArtifactStore
from lifecycle.async_lifecycle import AsyncLifecycle
from shared.model.model_variant import ModelVariantId
from shared.service.service import WorkerId
from worker.application.activity.abc.activity_request_factory import (
    ActivityRequestFactory,
)
from worker.application.ports.outbound.activity_manager.activity_manager import (
    ActivityManager,
)
from worker.application.ports.outbound.profiling.model_execution.model_execution_profile_publisher import (
    ModelExecutionProfilePublisher,
)
from worker.application.ports.outbound.profiling.model_execution.model_execution_profiler import (
    ModelExecutionProfiler,
)
from worker.application.ports.outbound.profiling.model_execution.model_static_profile_reader import (
    ModelStaticProfileReader,
)
from worker.application.profiling.abc.model_execution_profiling_coordinator import (
    ModelExecutionProfilingCoordinator,
)
from worker.application.profiling.abc.pending_models_poller import (
    PendingModelsPoller,
)
from worker.domain.activity.activity_request import ActivityType
from worker.domain.profiling.model_execution.model_execution_profile import (
    ModelExecutionProfile,
)


class DefaultModelExecutionProfileCoordinator(
    ModelExecutionProfilingCoordinator, AsyncLifecycle
):
    def __init__(
        self,
        worker_id: WorkerId,
        pending_models_poller: PendingModelsPoller,
        model_static_profile_reader: ModelStaticProfileReader,
        activity_request_factory: ActivityRequestFactory,
        activity_manager: ActivityManager,
        model_execution_profiler: ModelExecutionProfiler,
        artifact_store: ArtifactStore,
        model_exec_profile_publisher: ModelExecutionProfilePublisher,
    ):
        self._worker_id = worker_id
        self._pending_models_poller = pending_models_poller
        self._model_static_profile_reader = model_static_profile_reader
        self._activity_request_factory = activity_request_factory
        self._activity_manager = activity_manager
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
            activity_request = (
                self._activity_request_factory.create_request_for_activity_type(
                    ActivityType.EXECUTION_PROFILING
                )
            )
            activity_grant = await self._activity_manager.acquire_activity_grant(
                request=activity_request
            )
            async with activity_grant:
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
        model_execution_profile = await self.profile_model_execution(
            model_variant_id=model_variant_id
        )
        await self._model_exec_profile_storage.puplish_model_execution_profile(
            worker_id=self._worker_id, model_execution_profile=model_execution_profile
        )

    async def _profile_loop(self) -> None:
        async with aclosing(
            self._pending_models_poller.pending_model_ids()
        ) as pending_model_ids:
            async for pending_model_id in pending_model_ids:
                await self._coordinate_model_execution_profiling(
                    model_variant_id=pending_model_id
                )

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
