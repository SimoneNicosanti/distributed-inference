import asyncio
from typing import override

from lifecycle.async_lifecycle import AsyncLifecycle
from shared.identifiers.identifiers import WorkerId
from worker.application.ports.outbound.activity_manager import ActivityManager
from worker.application.ports.outbound.profiling.resource.cpu_profiler import (
    CpuProfiler,
)
from worker.application.ports.outbound.profiling.resource.gpu_profiler import (
    GpuProfiler,
)
from worker.application.ports.outbound.profiling.resource.ram_profiler import (
    RamProfiler,
)
from worker.application.ports.outbound.profiling.resource.resource_profile_publisher import (
    ResourceProfilePublisher,
)
from worker.application.profiling.contracts.resource_profiling_coordinator import (
    ResourceProfilingCoordinator,
)
from worker.domain.profiling.resource_profile import ResourceProfile

RESOURCE_PROFILING_INTERVAL_S = 60 * 5  # 5 minutes


class DefaultResourceProfilingCoordinator(ResourceProfilingCoordinator, AsyncLifecycle):
    def __init__(
        self,
        worker_id: WorkerId,
        activity_manager: ActivityManager,
        gpu_profiler: GpuProfiler,
        cpu_profiler: CpuProfiler,
        memory_profiler: RamProfiler,
        res_profile_publisher: ResourceProfilePublisher,
    ):
        self._worker_id = worker_id
        self._activity_manager = activity_manager

        self._gpu_profiler = gpu_profiler
        self._cpu_profiler = cpu_profiler
        self._memory_profiler = memory_profiler

        self._res_profile_publisher = res_profile_publisher

        self._loop_task: asyncio.Task[None] | None = None
        self._running_lock = asyncio.Lock()
        pass

    @override
    async def profile_resources(self) -> ResourceProfile:
        async with self._running_lock:
            gpu_profiles = await self._gpu_profiler.profile_gpus()
            cpu_profile = await self._cpu_profiler.profile_cpu()
            memory_profile = await self._memory_profiler.profile_ram()
            resource_profile = ResourceProfile(
                gpu_profiles=gpu_profiles,
                cpu_profile=cpu_profile,
                ram_profile=memory_profile,
            )
            return resource_profile

    async def _coordinate_resource_profiling(self) -> None:
        resource_profile = await self.profile_resources()
        await self._res_profile_publisher.publish_resource_profile(
            self._worker_id, resource_profile
        )

    async def _profile_loop(self) -> None:
        while True:
            await self._coordinate_resource_profiling()
            await asyncio.sleep(RESOURCE_PROFILING_INTERVAL_S)

    @override
    async def start(self) -> None:
        ## We do periodical profiling here.
        async with asyncio.TaskGroup() as task_group:
            self._loop_task = task_group.create_task(self._profile_loop())

    @override
    async def stop(self) -> None:
        if self._loop_task is None:
            return
        self._loop_task.cancel()
