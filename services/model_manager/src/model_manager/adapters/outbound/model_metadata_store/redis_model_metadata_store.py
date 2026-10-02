from redis import asyncio as redis_asyncio

from model_manager.application.ports.outbound.model_metadata_store import (
    ModelMetadataStore,
)


## TODO: Implement the redis model metadata store
class RedisModelMetadataStore(ModelMetadataStore):
    def __init__(self, redis: redis_asyncio.Redis) -> None:
        self._redis = redis
