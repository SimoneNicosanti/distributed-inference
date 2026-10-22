from abc import ABC, abstractmethod

from integration.service_instance.service_instance import ServiceInstance


class Registrar(ABC):
    @abstractmethod
    async def register_or_renew(self, service_instance: ServiceInstance) -> None: ...
