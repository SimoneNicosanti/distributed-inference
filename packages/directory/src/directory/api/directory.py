from abc import ABC, abstractmethod

from directory.contracts.service_instance import (
    ServiceInstance,
)
from shared.identifiers.identifiers import ServiceId


class ServiceDirectory(ABC):
    @abstractmethod
    async def get_service_instance_by_service_id(
        self, service_id: ServiceId
    ) -> ServiceInstance: ...

    @abstractmethod
    async def get_all_service_instances(self) -> list[ServiceInstance]: ...

    @abstractmethod
    async def put_service_instance(self, service_instance: ServiceInstance) -> None: ...
