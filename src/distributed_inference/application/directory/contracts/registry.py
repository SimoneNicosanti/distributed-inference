from abc import ABC, abstractmethod

from distributed_inference.application.directory.domain.server_registration import (
    ServerRegistration,
)
from distributed_inference.application.directory.domain.service_registration import (
    ServiceRegistration,
)


class ServerRegistry(ABC):
    @abstractmethod
    async def register_server(
        self, server_registration: ServerRegistration
    ) -> None: ...


class ServiceRegistry(ABC):
    @abstractmethod
    async def register_service(
        self, service_registration: ServiceRegistration
    ) -> None: ...
