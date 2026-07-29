from abc import ABC, abstractmethod
from typing import List

from distributed_inference.directory.domain.service_instance import (
    ServiceInstance,
    ServiceType,
)
from distributed_inference.domain.identifiers import ServerId, ServiceId


class ServiceResolver(ABC):
    @abstractmethod
    async def resolve_service_by_id(self, service_id: ServiceId) -> ServiceInstance:
        pass

    @abstractmethod
    async def resolve_service_by_server(
        self, server_id: ServerId
    ) -> List[ServiceInstance]:
        pass

    @abstractmethod
    async def resolve_service_by_server_and_type(
        self, server_id: ServerId, service_type: ServiceType
    ) -> List[ServiceInstance]:
        pass
