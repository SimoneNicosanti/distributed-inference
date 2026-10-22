from typing import override

from pydantic import ValidationError

from catalog.api.directory import Directory
from controller.application.ports.outbound.directory.worker_directory_gateway import (
    WorkerDirectoryGateway,
)
from integration.service_instance.worker_instance import WorkerInstance
from shared.service.service import WorkerId


class DefaultWorkerDirectoryGateway(WorkerDirectoryGateway):
    def __init__(self, directory: Directory) -> None:
        self._directory = directory

    @override
    async def get_registered_worker_ids(self) -> set[WorkerId]:
        all_services = await self._directory.get_all_service_instances()

        worker_instances = []
        for service_instance in all_services:
            try:
                WorkerInstance.model_validate(service_instance.model_dump())
                worker_instances.append(service_instance)
            except ValidationError:
                continue

        return {instance.service_id for instance in worker_instances}

    @override
    async def get_worker_instance_by_id(self, worker_id: WorkerId) -> WorkerInstance:
        service_instance = await self._directory.get_service_instance_by_service_id(
            worker_id
        )

        try:
            worker_instance = WorkerInstance.model_validate(
                service_instance.model_dump()
            )
            return worker_instance
        except ValidationError as e:
            raise ValueError(
                f"Could not find worker instance for worker id {worker_id}"
            ) from e
