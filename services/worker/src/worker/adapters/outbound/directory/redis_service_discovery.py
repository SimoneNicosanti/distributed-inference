from typing import override

from pydantic import ValidationError

from directory.contracts.service_instance import ServiceInstance
from directory.redis.redis_directory import RedisDirectory
from shared.identifiers.identifiers import WorkerId
from worker.application.ports.outbound.directory.service_directory import (
    ServiceDirectory,
)
from worker.domain.directory.worker_instance import WorkerInstance


class RedisServiceDiscovery(ServiceDirectory):
    def __init__(
        self,
        redis_directory: RedisDirectory,
    ) -> None:

        self._redis_directory = redis_directory

    @override
    async def put_service_instance(self, service_instance: ServiceInstance) -> None:
        await self._redis_directory.put_service_instance(service_instance)

    @override
    async def get_all_worker_instances(self) -> list[WorkerInstance]:
        service_instances = await self._redis_directory.get_all_service_instances()

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
    async def get_worker_instance_by_worker_id(
        self, worker_id: WorkerId
    ) -> WorkerInstance:
        service_instance = (
            await self._redis_directory.get_service_instance_by_service_id(worker_id)
        )

        try:
            service_instance_dump = service_instance.model_dump()
            worker_instance = WorkerInstance.model_validate(service_instance_dump)
            return worker_instance
        except ValidationError:
            raise ValueError(
                f"Could not find worker instance for worker id {worker_id}"
            )
