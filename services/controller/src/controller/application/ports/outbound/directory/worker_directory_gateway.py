from abc import ABC, abstractmethod

from integration.service_instance.worker_instance import WorkerInstance
from shared.service.service import WorkerId


class WorkerDirectoryGateway(ABC):
    @abstractmethod
    async def get_registered_worker_ids(
        self,
    ) -> set[WorkerId]: ...

    @abstractmethod
    async def get_worker_instance_by_id(
        self, worker_id: WorkerId
    ) -> WorkerInstance: ...
