from abc import ABC, abstractmethod


class ServerDirectory(ABC):
    @abstractmethod
    def resolve_server(self, server_id) -> None: ...


class ServiceDirectory(ABC):
    @abstractmethod
    def resolve_service(self, service_id) -> None: ...
