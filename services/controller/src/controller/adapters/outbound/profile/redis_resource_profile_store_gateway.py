from collections.abc import Iterable
from typing import override

from redis import asyncio as redis_asyncio

from controller.adapters.outbound.profile.mappers.worker_resource_profile_mapper import (
    WorkerResourceProfileMapper,
)
from controller.application.ports.outbound.profile.resource_profile_store_gateway import (
    ResourceProfileStoreGateway,
)
from controller.domain.profile.resource_profile import (
    WorkerResourceProfile,
)
from integration.dto.worker_profiles.resource_profile_dto import ResourceProfileDto
from integration.redis.worker_profiles.resource_profile_keys import (
    build_resource_profile_key_per_worker,
)
from shared.service.service import WorkerId


class RedisResourceProfileStoreGateway(ResourceProfileStoreGateway):
    def __init__(self, redis: redis_asyncio.Redis) -> None:
        self._redis = redis

    @override
    async def get_resource_profiles_for_workers(
        self,
        worker_ids: Iterable[WorkerId],
    ) -> list[WorkerResourceProfile]:
        ordered_worker_ids = sorted(
            set(worker_ids),
            key=lambda worker_id: str(worker_id.service_id),
        )
        if not ordered_worker_ids:
            return []

        redis_keys = [
            build_resource_profile_key_per_worker(worker_id)
            for worker_id in ordered_worker_ids
        ]
        values = await self._redis.mget(redis_keys)

        profiles: list[WorkerResourceProfile] = []
        for worker_id, value in zip(ordered_worker_ids, values, strict=True):
            if value is None:
                continue

            dto = ResourceProfileDto.model_validate_json(value)
            if dto.worker_id != worker_id:
                raise ValueError(
                    "Stored resource profile worker does not match the requested worker"
                )

            profiles.append(
                WorkerResourceProfileMapper.worker_resource_profile_from_dto(dto)
            )

        return profiles
