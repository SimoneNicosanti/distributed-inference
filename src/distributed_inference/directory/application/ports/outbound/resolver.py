from abc import ABC, abstractmethod

from distributed_inference.directory.domain.service_instance import (
    ServiceInstance,
    ServiceProtocol,
    ServiceType,
)
from distributed_inference.domain.identifiers import ServerId, ServiceId


class ServiceResolver(ABC):
    @abstractmethod
    async def resolve_service_by_id(self, service_id: ServiceId) -> ServiceInstance: ...

    @abstractmethod
    async def resolve_service_by_server(
        self, server_id: ServerId
    ) -> list[ServiceInstance]: ...

    @abstractmethod
    async def resolve_service_by_type_and_protocol(
        self, service_type: ServiceType, service_protocol: ServiceProtocol
    ) -> list[ServiceInstance]: ...

    @abstractmethod
    async def resolve_service_by_server_and_type(
        self, server_id: ServerId, service_type: ServiceType
    ) -> list[ServiceInstance]: ...
