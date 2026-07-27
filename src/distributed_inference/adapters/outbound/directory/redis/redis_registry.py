from typing import override

from redis import asyncio as redis_asyncio

from distributed_inference.adapters.outbound.directory.redis import (
    redis_directory_utils,
)
from distributed_inference.application.directory.contracts.registry import (
    ServerRegistry,
    ServiceRegistry,
)
from distributed_inference.application.directory.domain.server_registration import (
    ServerRegistration,
)
from distributed_inference.application.directory.domain.service_registration import (
    ServiceRegistration,
)

## TODO: We should add automatic deletion of old servers to implement TTL


## NOTE: This ServerRegistry might not be needed; we are interested about the services, not the servers
class RedisServerRegistry(ServerRegistry):
    def __init__(self, redis: redis_asyncio.Redis) -> None:
        self._redis = redis

    @override
    async def register_server(self, server_registration: ServerRegistration) -> None:
        server_id = server_registration.server_id
        redis_key: str = redis_directory_utils.generate_redis_directory_server_key(
            server_id
        )
        redis_value: str = server_registration.server_id.model_dump_json()
        success = await self._redis.set(redis_key, redis_value)

        if not success:
            raise Exception("Failed to register server")


class RedisServiceRegistry(ServiceRegistry):
    def __init__(self, redis: redis_asyncio.Redis) -> None:
        self._redis = redis

    @override
    async def register_service(self, service_registration: ServiceRegistration) -> None:

        service_instance = service_registration.service_instance
        service_id = service_registration.service_instance.service_id

        redis_key: str = redis_directory_utils.generate_redis_directory_service_key(
            service_id
        )
        redis_value: str = service_instance.model_dump_json()

        success = await self._redis.set(redis_key, redis_value)
        if not success:
            raise Exception("Failed to register service")
