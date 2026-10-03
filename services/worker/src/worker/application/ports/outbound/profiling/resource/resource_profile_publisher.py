from abc import ABC, abstractmethod

from shared.service.service import WorkerId
from worker.domain.profiling.resource_profile import ResourceProfile


class ResourceProfilePublisher(ABC):
    @abstractmethod
    async def publish_resource_profile(
        self, worker_id: WorkerId, resource_profile: ResourceProfile
    ) -> None: ...
