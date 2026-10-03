from abc import ABC, abstractmethod

from directory.contracts.service_instance import ServiceInstance
from shared.service.service import WorkerId
from worker.domain.directory.worker_instance import WorkerInstance


class ServiceDirectory(ABC):
    @abstractmethod
    async def put_service_instance(self, service_instance: ServiceInstance) -> None: ...

    @abstractmethod
    async def get_all_worker_instances(self) -> list[WorkerInstance]: ...

    @abstractmethod
    async def get_worker_instance_by_worker_id(
        self, worker_id: WorkerId
    ) -> WorkerInstance: ...
