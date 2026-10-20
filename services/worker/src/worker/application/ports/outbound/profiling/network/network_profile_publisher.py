from abc import ABC, abstractmethod

from worker.domain.profiling.network_profile import NetworkProfile


class NetworkProfilePublisher(ABC):
    @abstractmethod
    async def publish_network_profile(
        self, network_profile: NetworkProfile
    ) -> None: ...
