from abc import ABC, abstractmethod

from distributed_inference.application.worker.contracts.resource.resource_type import (
    ResourceType,
)


class ResourceManager(ABC):
    @abstractmethod
    def try_allocate_resources(self, resources: dict[ResourceType, int]) -> None: ...

    @abstractmethod
    def release_resources(self, resources: dict[ResourceType, int]) -> None: ...
