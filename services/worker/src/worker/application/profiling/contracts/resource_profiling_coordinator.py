from abc import ABC, abstractmethod

from worker.domain.profiling.resource_profile import ResourceProfile


class ResourceProfilingCoordinator(ABC):
    @abstractmethod
    async def profile_resources(self) -> ResourceProfile: ...
