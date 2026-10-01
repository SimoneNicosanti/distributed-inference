from abc import ABC, abstractmethod

from worker.domain.profiling.resource_profile import RamProfile


class RamProfiler(ABC):
    @abstractmethod
    async def profile_ram(self) -> RamProfile: ...
