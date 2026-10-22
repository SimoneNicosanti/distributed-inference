from typing import override

from pydantic import ValidationError

from catalog.api.directory import Directory
from shared.service.service import ResultSinkId, WorkerId
from worker.application.ports.outbound.catalog.service.service_directory_gateway import (
    ServiceDirectoryGateway,
)
from integration.service_instance.ingress_instance import ResultSinkInstance
from integration.service_instance.worker_instance import WorkerInstance


class DefaultServiceDirectoryGateway(ServiceDirectoryGateway):
    def __init__(
        self,
        directory: Directory,
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
            return worker_instance
        except ValidationError as e:
            raise ValueError(
                f"Could not find worker instance for worker id {worker_id}"
            ) from e

    @override
    async def get_result_sink_instance_by_id(
        self, result_sink_id: ResultSinkId
    ) -> ResultSinkInstance:
        service_instance = await self._directory.get_service_instance_by_service_id(
            result_sink_id
        )

        try:
            service_instance_dump = service_instance.model_dump()
            result_sink_instance = ResultSinkInstance.model_validate(
                service_instance_dump
            )
            return result_sink_instance
        except ValidationError as e:
            raise ValueError(
                f"Could not find worker instance for worker id {result_sink_id}"
            ) from e
