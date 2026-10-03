from abc import ABC, abstractmethod

from shared.service.service import WorkerId
from worker.domain.profiling.network_profile import NetworkProfile


class NetworkProfilePublisher(ABC):
    @abstractmethod
    async def publish_network_profile(
        self, worker_id: WorkerId, network_profile: NetworkProfile
    ) -> None: ...
