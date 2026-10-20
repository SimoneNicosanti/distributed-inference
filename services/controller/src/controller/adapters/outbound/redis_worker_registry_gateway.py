from typing import override

from redis import asyncio as redis_asyncio

from controller.application.ports.outbound.worker_registry import WorkerRegistryGateway
from directory.contracts.service_instance import CapabilityType, ServiceInstance
from shared.service.service import WorkerId

_SERVICE_REGISTRY_KEY = "discovery:services"
_REQUIRED_WORKER_CAPABILITIES = frozenset(
    {
        CapabilityType.PARTITION_INFERENCE,
        CapabilityType.DEPLOYMENT_ACTUATION,
        CapabilityType.NETWORK_PROFILING,
        CapabilityType.EXECUTION_PROFILING,
        CapabilityType.RESOURCE_PROFILING,
    }
)


class RedisWorkerRegistryGateway(WorkerRegistryGateway):
    def __init__(self, redis: redis_asyncio.Redis) -> None:
        self._redis = redis

    @override
    async def get_active_workers(self) -> set[WorkerId]:
        values = await self._redis.hvals(_SERVICE_REGISTRY_KEY)
        service_instances = (
            ServiceInstance.model_validate_json(value) for value in values
        )

        return {
            service_instance.service_id
            for service_instance in service_instances
            if {capability.type for capability in service_instance.capabilities}
            >= _REQUIRED_WORKER_CAPABILITIES
        }
