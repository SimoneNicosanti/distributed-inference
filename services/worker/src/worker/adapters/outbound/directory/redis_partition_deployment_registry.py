from typing import override

from redis import asyncio as redis_asyncio

from shared.plan.partition_deployment_id import PartitionDeploymentId
from shared.plan.partition_replica_id import PartitionReplicaId
from worker.application.ports.outbound.directory.partition_deployment_registry import (
    PartitionDeploymentRegistry,
)


class RedisPartitionDeploymentRegistry(PartitionDeploymentRegistry):
    _DEPLOYMENTS_KEY = "partition-deployments"

    def __init__(self, redis: redis_asyncio.Redis) -> None:
        self._redis = redis

    @override
    async def register_partition_deployment(
        self, deployment_id: PartitionDeploymentId
    ) -> None:
        await self._redis.hset(
            self._DEPLOYMENTS_KEY,
            deployment_id.replica_id.model_dump_json(),
            deployment_id.model_dump_json(),
        )

    @override
    async def get_partition_deployment_id(
        self, replica_id: PartitionReplicaId
    ) -> PartitionDeploymentId:
        value = await self._redis.hget(
            self._DEPLOYMENTS_KEY,
            replica_id.model_dump_json(),
        )
        if value is None:
            raise KeyError(f"Partition replica {replica_id} is not deployed")

        deployment_id = PartitionDeploymentId.model_validate_json(value)
        if deployment_id.replica_id != replica_id:
            raise ValueError("Partition deployment registry ID mismatch")

        return deployment_id
