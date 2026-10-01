from abc import ABC, abstractmethod

from worker.domain.profiling.resource_profile import GpuProfile


class GpuProfiler(ABC):
    @abstractmethod
    async def profile_gpus(self) -> tuple[GpuProfile, ...]: ...
