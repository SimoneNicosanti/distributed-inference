from typing import List, override

import pydantic

from distributed_inference.directory.adapters.outbound.redis import (
    redis_directory_utils,
)
from distributed_inference.directory.application.ports.outbound.resolver import (
    ServiceResolver,
)
from distributed_inference.directory.domain.service_instance import (
    ServiceInstance,
    ServiceType,
)
from distributed_inference.domain.identifiers import ServerId, ServiceId
from redis import asyncio as redis_asyncio


class RedisServiceResolver(ServiceResolver):
    def __init__(self, redis: redis_asyncio.Redis) -> None:
        self._redis = redis

    @override
    async def resolve_service_by_id(self, service_id: ServiceId) -> ServiceInstance:
        redis_key = redis_directory_utils.generate_redis_directory_service_key(
            service_id
        )

        result = await self._redis.get(redis_key)
        if result is None:
            raise KeyError(f"Service {service_id} not found")

        return ServiceInstance.model_validate_json(result)

    @override
    async def resolve_service_by_server(
        self, server_id: ServerId
    ) -> List[ServiceInstance]:
        redis_key_pattern = redis_directory_utils.generate_redis_directory_service_key_pattern_per_server(
            server_id
        )

        ## NOTE: This is not atomic. Deletion between the two calls might happen
        matched_keys: list[str] = [
            key async for key in self._redis.scan_iter(match=redis_key_pattern)
        ]
        if not matched_keys:
            return []
        unique_matched_keys = set(matched_keys)
        results = await self._redis.mget(unique_matched_keys)

        service_instances: List[ServiceInstance] = []
        for result in results:
            if result is None:
                continue
            try:
                service_instance = ServiceInstance.model_validate_json(result)
            except pydantic.ValidationError:
                continue
            service_instances.append(service_instance)

        return service_instances

    @override
    async def resolve_service_by_server_and_type(
        self, server_id: ServerId, service_type: ServiceType
    ) -> List[ServiceInstance]:

        service_instances = await self.resolve_service_by_server(server_id)
        filtered_service_instances = [
            service_instance
            for service_instance in service_instances
            if service_instance.service_type == service_type
        ]
        return filtered_service_instances
