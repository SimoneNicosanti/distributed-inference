from abc import ABC, abstractmethod

from integration.service_instance.service_instance import CapabilityCouple


class CapabilityLifeChecker(ABC):
    @abstractmethod
    async def get_if_alive(self) -> CapabilityCouple | None: ...
