import asyncio
from typing import override

from pydantic import ValidationError
from redis import RedisError
from redis import asyncio as asyncio_redis

from integration.dto.worker_plan.worker_plan_dto import WorkerPlanDto
from integration.redis.worker_plan.keys_factory import WorkerPlanKeyFactory
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
        key_factory: WorkerPlanKeyFactory,
    ) -> None:
        super().__init__()
        self._worker_id = worker_id
        self._redis = redis
        self._key_factory = key_factory
        self._cache: dict[PlanVersion, WorkerPlan] = {}
        self._cache_lock = asyncio.Lock()

    @override
    async def get_plan_by_version(self, version: PlanVersion) -> WorkerPlan:
        async with self._cache_lock:
            if version in self._cache:
                return self._cache[version]

            hash_key = self._key_factory.build_hash_key(version)
            field_key = self._key_factory.build_field_key(self._worker_id)

            try:
                value = await self._redis.hget(hash_key, field_key)
            except RedisError as e:
                raise WorkerPlanReadError(
                    "Could not reach redis to read the plan"
                ) from e

            if value is None:
                raise WorkerPlanReadError(
                    f"Plan with version {version.version_number} does not exist for {self._worker_id.id}"
                )

            try:
                dto = WorkerPlanDto.model_validate_json(value)
            except ValidationError as e:
                raise WorkerPlanReadError("Wrong format for found worker plan") from e

            if dto.plan_version != version:
                raise WorkerPlanReadError("Stored plan version mismatch")

            worker_plan = worker_plan_from_dto(dto)
            self._cache[version] = worker_plan

            return worker_plan
