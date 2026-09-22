from abc import ABC, abstractmethod

from integration.directory.registration import (
    ServerRegistration,
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
