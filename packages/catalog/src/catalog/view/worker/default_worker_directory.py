from typing import override

from pydantic import ValidationError

from catalog.api.service.service_directory import (
    ServiceDirectory,
    ServiceInstanceReadError,
)
from catalog.api.worker.worker_directory import (
    WorkerDirectory,
)
from integration.service_instance.worker_instance import WorkerInstance
from shared.service.service import WorkerId


class DefaultWorkerDirectory(WorkerDirectory):
    def __init__(
        self,
        directory: ServiceDirectory,
    ) -> None:

        self._directory = directory

    @override
    async def get_all_worker_instances(self) -> list[WorkerInstance]:
        service_instances = await self._directory.get_all_service_instances()

        worker_instances = []
        for service_instance in service_instances:
            try:
                service_instance_dump = service_instance.model_dump()
                worker_instance = WorkerInstance.model_validate(service_instance_dump)
                worker_instances.append(worker_instance)
            except ValidationError:
                continue
        return worker_instances

    @override
    async def get_worker_instance_by_id(self, worker_id: WorkerId) -> WorkerInstance:
        service_instance = await self._directory.get_service_instance_by_service_id(
            worker_id
        )

        try:
            service_instance_dump = service_instance.model_dump()
            worker_instance = WorkerInstance.model_validate(service_instance_dump)

            if worker_instance.service_id != worker_id:
                raise ServiceInstanceReadError(
                    "Discovered worker identity does not match requested worker"
                )

            return worker_instance
        except ValidationError as e:
            raise ServiceInstanceReadError(
                f"Could not find worker instance for worker id {worker_id}"
            ) from e
