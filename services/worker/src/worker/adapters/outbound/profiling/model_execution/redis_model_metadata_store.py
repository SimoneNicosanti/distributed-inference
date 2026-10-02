from typing import override

from redis import asyncio as redis_asyncio

from shared.model.model_version import ModelVersionId
from worker.application.ports.outbound.profiling.model_execution.model_metadata_store import (
    ModelMetadataStore,
)


## TODO: Implement the redis model profile store
class RedisModelMetadataStore(ModelMetadataStore):
    def __init__(self, redis: redis_asyncio.Redis) -> None:
        self._redis = redis

    @override
    async def get_all_model_version_ids(self) -> list[ModelVersionId]:
        redis_key = await self.__build_redis_key()
        pass

    @staticmethod
    async def __build_redis_key() -> str:
        return "model-version-profiles:"
