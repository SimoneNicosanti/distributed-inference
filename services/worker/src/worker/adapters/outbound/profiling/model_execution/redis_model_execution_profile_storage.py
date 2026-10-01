import asyncio
from hashlib import md5
from typing import override

from redis import asyncio as redis_asyncio

from shared.identifiers.identifiers import WorkerId
from shared.model.model_version import ModelVersionId
from worker.application.ports.outbound.profiling.model_execution.model_execution_profile_store import (
    ModelExecutionProfileStore,
)
from worker.domain.profiling.model_execution_profile import ModelExecutionProfile


class RedisModelExecutionProfileStorage(ModelExecutionProfileStore):
    def __init__(self, redis: redis_asyncio.Redis) -> None:
        self._redis = redis

    @override
    async def put_model_execution_profile(
        self, worker_id: WorkerId, model_execution_profile: ModelExecutionProfile
    ) -> None:
        worker_key = await self.__build_worker_key(worker_id)
        profile_id = await self.__compute_model_version_id_hash(
            model_execution_profile.model_version_id
        )

        await self._redis.hset(
            worker_key, profile_id, model_execution_profile.model_dump_json()
        )

    @override
    async def get_model_execution_profile_by_worker_id(
        self, worker_id: WorkerId, model_version_id: ModelVersionId
    ) -> ModelExecutionProfile:
        worker_key = await self.__build_worker_key(worker_id)
        profile_id = await self.__compute_model_version_id_hash(model_version_id)

        value = await self._redis.hget(worker_key, profile_id)

        if value is None:
            raise ValueError("Model execution profile not found")

        return ModelExecutionProfile.model_validate_json(value)

    @override
    async def get_all_profiles_by_worker_id(
        self, worker_id: WorkerId
    ) -> list[ModelExecutionProfile]:
        worker_key = await self.__build_worker_key(worker_id)

        values = await self._redis.hvals(worker_key)

        return [ModelExecutionProfile.model_validate_json(value) for value in values]

    @override
    async def check_model_execution_profile_exists(
        self, worker_id: WorkerId, model_version_id: ModelVersionId
    ) -> bool:
        worker_key = await self.__build_worker_key(worker_id)
        profile_id = await self.__compute_model_version_id_hash(model_version_id)

        return await self._redis.hexists(worker_key, profile_id)

    @staticmethod
    async def __build_worker_key(
        worker_id: WorkerId,
    ) -> str:

        return f"worker:{worker_id.service_id}:model-execution-profiles"

    @staticmethod
    async def __compute_model_version_id_hash(
        model_version_id: ModelVersionId,
    ) -> str:
        md5_value = await asyncio.to_thread(
            md5, model_version_id.model_dump_json().encode("utf-8")
        )
        return md5_value.hexdigest()
