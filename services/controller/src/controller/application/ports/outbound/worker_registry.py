from abc import ABC, abstractmethod

from shared.service.service import WorkerId


class WorkerRegistryGateway(ABC):
    @abstractmethod
    async def get_active_workers(
        self,
    ) -> set[WorkerId]: ...
