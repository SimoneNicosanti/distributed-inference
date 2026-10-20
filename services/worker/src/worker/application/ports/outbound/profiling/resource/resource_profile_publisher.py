from abc import ABC, abstractmethod

from worker.domain.profiling.resource_profile import ResourceProfile


class ResourceProfilePublisher(ABC):
    @abstractmethod
    async def publish_resource_profile(
        self, resource_profile: ResourceProfile
    ) -> None: ...
