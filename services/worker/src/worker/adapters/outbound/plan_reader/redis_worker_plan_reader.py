from typing import override

from redis import asyncio as asyncio_redis

from integration.dto.worker_plan.worker_plan_dto import WorkerPlanDto
from redis_access.address_factory.worker_plan.address_factory import (
    WorkerPlanAddressFactory,
)
from redis_access.contract.redis_address import HashMapRedisAddress
from redis_access.redis_reader.abc.redis_reader import RedisReader, RedisReadError
from shared.plan.plan_version import PlanVersion
from shared.service.service import WorkerId
from worker.adapters.outbound.plan_reader.worker_plan_mapper import worker_plan_from_dto
from worker.application.ports.outbound.plan_reader.worker_plan_reader import (
    WorkerPlanReader,
    WorkerPlanReadError,
)
from worker.domain.plan.worker_plan import WorkerPlan


class RedisWorkerPlanReader(WorkerPlanReader):
    def __init__(
        self,
        worker_id: WorkerId,
        redis: asyncio_redis.Redis,
        redis_reader: RedisReader[HashMapRedisAddress],
        address_factory: WorkerPlanAddressFactory[HashMapRedisAddress],
    ) -> None:
        super().__init__()
        self._worker_id = worker_id
        self._redis = redis
        self._redis_reader = redis_reader
        self._address_factory = address_factory

    @override
    async def get_plan_by_version(self, version: PlanVersion) -> WorkerPlan:

        try:
            addr = self._address_factory.build_versioned_worker_plan_address(
                version, self._worker_id
            )
            value = await self._redis_reader.read(self._redis, addr)
            dto = WorkerPlanDto.model_validate_json(value)

            if dto.plan_version != version:
                raise ValueError("Stored plan version mismatch")
            worker_plan = worker_plan_from_dto(dto)

            return worker_plan
        except (RedisReadError, ValueError) as e:
            raise WorkerPlanReadError("Error while reading Worker Plan") from e
