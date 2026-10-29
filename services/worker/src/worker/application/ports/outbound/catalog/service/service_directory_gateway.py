from abc import ABC, abstractmethod

from integration.service_instance.ingress_instance import ResultSinkInstance
from integration.service_instance.worker_instance import WorkerInstance
from shared.service.service import SinkId, WorkerId


class ServiceDirectoryGateway(ABC):
    @abstractmethod
    async def get_all_worker_instances(self) -> list[WorkerInstance]: ...

    @abstractmethod
    async def get_worker_instance_by_id(
        self, worker_id: WorkerId
    ) -> WorkerInstance: ...

    @abstractmethod
    async def get_result_sink_instance_by_id(
        self, result_sink_id: SinkId
    ) -> ResultSinkInstance: ...
