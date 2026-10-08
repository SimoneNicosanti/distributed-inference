import asyncio
from hashlib import md5
from typing import override

from redis import asyncio as redis_asyncio

from integration.worker.profiling.redis.execution_profile_keys import (
    build_execution_profile_ids_key_per_worker,
    build_execution_profile_key_per_worker,
)
from shared.model.model_variant import ModelVariantId
from shared.service.service import WorkerId
from worker.application.ports.outbound.profiling.model_execution.model_execution_profile_publisher import (
    ModelExecutionProfilePublisher,
)
from worker.application.ports.outbound.profiling.model_execution.model_execution_profile_reader import (
    ModelExecutionProfileReader,
)
from worker.domain.profiling.model_execution.model_execution_profile import (
    ModelExecutionProfile,
)


class RedisModelExecutionProfileStore(
    ModelExecutionProfilePublisher, ModelExecutionProfileReader
):
    def __init__(self, redis: redis_asyncio.Redis) -> None:
        self._redis = redis

    @override
    async def puplish_model_execution_profile(
        self, worker_id: WorkerId, model_execution_profile: ModelExecutionProfile
    ) -> None:
        worker_key = build_execution_profile_key_per_worker(worker_id)
        worker_ids_key = build_execution_profile_ids_key_per_worker(worker_id)
        profile_id = await self.__compute_model_version_id_hash(
            model_execution_profile.model_version_id
        )
        profile_json = model_execution_profile.model_dump_json()
        model_variant_id_json = (
            model_execution_profile.model_version_id.model_dump_json()
        )

        async with self._redis.pipeline(transaction=True) as pipeline:
            pipeline.hset(worker_key, profile_id, profile_json)
            pipeline.sadd(worker_ids_key, model_variant_id_json)
            await pipeline.execute()

    @override
    async def get_model_execution_profile_by_worker_id(
        self, worker_id: WorkerId, model_version_id: ModelVariantId
    ) -> ModelExecutionProfile:
        worker_key = build_execution_profile_key_per_worker(worker_id)
        profile_id = await self.__compute_model_version_id_hash(model_version_id)

        value = await self._redis.hget(worker_key, profile_id)

        if value is None:
            raise ValueError("Model execution profile not found")

        return ModelExecutionProfile.model_validate_json(value)

    @override
    async def get_all_profiled_model_ids_by_worker_id(
        self, worker_id: WorkerId
    ) -> list[ModelVariantId]:
        return [
            ModelVariantId.model_validate_json(value)
            async for value in self._redis.sscan_iter(
                build_execution_profile_ids_key_per_worker(worker_id)
            )
        ]

    @override
    async def get_all_profiles_by_worker_id(
        self, worker_id: WorkerId
    ) -> list[ModelExecutionProfile]:
        worker_key = build_execution_profile_key_per_worker(worker_id)

        values = await self._redis.hvals(worker_key)

        return [ModelExecutionProfile.model_validate_json(value) for value in values]

    @override
    async def check_model_execution_profile_exists(
        self, worker_id: WorkerId, model_version_id: ModelVariantId
    ) -> bool:
        worker_key = build_execution_profile_key_per_worker(worker_id)
        profile_id = await self.__compute_model_version_id_hash(model_version_id)

        return await self._redis.hexists(worker_key, profile_id)

    @staticmethod
    async def __compute_model_version_id_hash(
        model_version_id: ModelVariantId,
    ) -> str:
        md5_value = await asyncio.to_thread(
            md5, model_version_id.model_dump_json().encode("utf-8")
        )
        return md5_value.hexdigest()
