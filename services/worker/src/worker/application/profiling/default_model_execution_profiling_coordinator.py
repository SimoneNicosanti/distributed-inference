import asyncio
from contextlib import aclosing
from typing import override

from artifacts.contracts.artifact_ref import ArtifactRef
from artifacts.storage.artifact_store import ArtifactStore
from lifecycle.async_lifecycle import AsyncLifecycle
from shared.identifiers.identifiers import WorkerId
from shared.model.model_version import ModelVersionId
from worker.application.ports.outbound.profiling.model_execution.model_execution_profile_store import (
    ModelExecutionProfileStore,
)
from worker.application.ports.outbound.profiling.model_execution.model_execution_profiler import (
    ModelExecutionProfiler,
)
from worker.application.ports.outbound.profiling.model_execution.pending_models_poller import (
    PendingModelsPoller,
)
from worker.application.profiling.contracts.model_execution_profiling_coordinator import (
    ModelExecutionProfilingCoordinator,
)
from worker.domain.profiling.model_execution_profile import ModelExecutionProfile


class DefaultModelExecutionProfileCoordinator(
    ModelExecutionProfilingCoordinator, AsyncLifecycle
):
    def __init__(
        self,
        worker_id: WorkerId,
        pending_models_poller: PendingModelsPoller,
        model_execution_profiler: ModelExecutionProfiler,
        artifact_store: ArtifactStore,
        model_exec_profile_storage: ModelExecutionProfileStore,
    ):
        self._worker_id = worker_id
        self._pending_models_poller = pending_models_poller
        self._model_execution_profiler = model_execution_profiler
        self._artifact_store = artifact_store
        self._model_exec_profile_storage = model_exec_profile_storage

        self._profile_loop_task: asyncio.Task | None = None

        self._profiling_lock = asyncio.Lock()

    @override
    async def profile_model_execution(
        self, model_version_id: ModelVersionId
    ) -> ModelExecutionProfile:
        async with self._profiling_lock:
            artifact_ref = ArtifactRef(value=model_version_id.model_dump_json())
            async with self._artifact_store.download_artifact(
                artifact_ref
            ) as artifact_bundle:
                model_profile = (
                    await self._model_execution_profiler.profile_model_execution(
                        artifact_bundle
                    )
                )

                return model_profile

    async def _coordinate_model_execution_profiling(
        self, model_version_id: ModelVersionId
    ) -> None:
        model_profile = await self.profile_model_execution(
            model_version_id=model_version_id
        )
        await self._model_exec_profile_storage.put_model_execution_profile(
            worker_id=self._worker_id, model_execution_profile=model_profile
        )

    async def _profile_loop(self) -> None:
        async with aclosing(
            self._pending_models_poller.pending_models()
        ) as pending_models:
            async for pending_model in pending_models:
                await self._coordinate_model_execution_profiling(
                    model_version_id=pending_model
                )

    @override
    async def start(self) -> None:
        ## We start the profiling with polling for pending models
        async with asyncio.TaskGroup() as task_group:
            self._profile_loop_task = task_group.create_task(self._profile_loop())

    @override
    async def stop(self) -> None:
        if self._profile_loop_task is None:
            return
        self._profile_loop_task.cancel()
