from abc import ABC, abstractmethod

from worker.domain.profiling.resource_profile import CpuProfile


class CpuProfiler(ABC):
    @abstractmethod
    async def profile_cpu(self) -> CpuProfile: ...
