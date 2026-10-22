from typing import override

from catalog.api.registrar import Registrar
from catalog.redis.keys_factory import (
    build_catalog_key,
    build_service_hset_key,
)
from integration.service_instance.service_instance import ServiceInstance
from redis import asyncio as redis_asyncio


class RedisRegistrar(Registrar):
    def __init__(self, redis: redis_asyncio.Redis, entry_ttl_s: int) -> None:
        self._redis = redis
        if entry_ttl_s <= 0:
            raise ValueError("entry_ttl must be positive")
        self._entry_ttl = entry_ttl_s

    @override
    async def register_or_renew(self, service_instance: ServiceInstance) -> None:
        catalog_key = build_catalog_key()
        service_hset_key = build_service_hset_key(service_instance.service_id)

        pipeline = self._redis.pipeline(transaction=True)
        pipeline.hset(
            catalog_key,
            service_hset_key,
            service_instance.model_dump_json(),
        )
        pipeline.hexpire(
            catalog_key,
            self._entry_ttl,
            service_hset_key,
        )
        await pipeline.execute()
