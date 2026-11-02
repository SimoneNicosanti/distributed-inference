from typing import override

from catalog.api.worker.worker_directory import WorkerDirectory
from controller.application.ports.outbound.directory.worker_directory_gateway import (
    WorkerDirectoryGateway,
)
from integration.service_instance.worker_instance import WorkerInstance
from shared.service.service import WorkerId


class DefaultWorkerDirectoryGateway(WorkerDirectoryGateway):
    def __init__(self, directory: WorkerDirectory) -> None:
        self._directory = directory

    @override
    async def get_registered_worker_ids(self) -> set[WorkerId]:
        all_worker_instances = await self._directory.get_all_worker_instances()

        return {worker_instance.service_id for worker_instance in all_worker_instances}

    @override
    async def get_worker_instance_by_id(self, worker_id: WorkerId) -> WorkerInstance:
        return await self._directory.get_worker_instance_by_id(worker_id)
