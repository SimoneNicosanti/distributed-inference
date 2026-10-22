from abc import ABC, abstractmethod

from integration.service_instance.service_instance import (
    ServiceInstance,
)
from shared.service.service import ServiceId


class Directory(ABC):
    @abstractmethod
    async def get_service_instance_by_service_id(
        self, service_id: ServiceId
    ) -> ServiceInstance: ...

    @abstractmethod
    async def get_all_service_instances(self) -> list[ServiceInstance]: ...
