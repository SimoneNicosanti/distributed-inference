from abc import ABC, abstractmethod

from worker.domain.profiling.network_profile import NetworkProfile


class NetworkProfilingCoordinator(ABC):
    @abstractmethod
    async def profile_network(self) -> NetworkProfile: ...
