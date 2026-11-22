from typing import override

from catalog.api.service.service_registrar import ServiceRegistrar
from integration.service_instance.service_instance import ServiceInstance
from redis import asyncio as redis_asyncio
from redis_access.address_factory.service.service_instance_address_factory import (
    ServiceInstanceAddressFactory,
)
from redis_access.contract.redis_address import HashMapRedisAddress


class RedisServiceRegistrar(ServiceRegistrar):
    def __init__(
        self,
        redis: redis_asyncio.Redis,
        entry_ttl_s: int,
        address_factory: ServiceInstanceAddressFactory[HashMapRedisAddress],
    ) -> None:
        self._redis = redis
        if entry_ttl_s <= 0:
            raise ValueError("entry_ttl must be positive")
        self._entry_ttl = entry_ttl_s
        self._address_factory = address_factory

    @override
    async def register_or_renew(self, service_instance: ServiceInstance) -> None:
        addr = self._address_factory.build_service_instance_address(
            service_instance.service_id
        )

        pipeline = self._redis.pipeline(transaction=True)
        pipeline.hset(
            addr.hash_map_key,
            addr.field_key,
            service_instance.model_dump_json(),
        )
        pipeline.hexpire(addr.hash_map_key, self._entry_ttl, addr.field_key)
        await pipeline.execute()
