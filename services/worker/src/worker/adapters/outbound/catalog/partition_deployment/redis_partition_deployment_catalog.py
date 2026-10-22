from typing import override

from redis import asyncio as redis_asyncio

from integration.shared.id_json import (
    partition_deployment_id_from_json,
    partition_deployment_id_to_json,
    partition_replica_id_to_json,
)
from shared.plan.partition_deployment_id import PartitionDeploymentId
from shared.plan.partition_replica_id import PartitionReplicaId
from shared.plan.plan_version import PlanVersion
from worker.application.ports.outbound.catalog.partition_deployment.partition_deployment_directory import (
    PartitionDeploymentDirectory,
)
from worker.application.ports.outbound.catalog.partition_deployment.partition_deployment_registrar import (
    PartitionDeploymentRegistrar,
)


class RedisPartitionDeploymentCatalog(
    PartitionDeploymentRegistrar, PartitionDeploymentDirectory
):
    def __init__(self, redis: redis_asyncio.Redis) -> None:
        self._redis = redis

    @override
    async def register_partition_deployment(
        self,
        plan_version: PlanVersion,
        deployment_id: PartitionDeploymentId,
    ) -> None:
        await self._redis.hset(
            self._build_deployments_key(plan_version),
            partition_replica_id_to_json(deployment_id.replica_id),
            partition_deployment_id_to_json(deployment_id),
        )

    @override
    async def get_partition_deployment_id(
        self,
        plan_version: PlanVersion,
        replica_id: PartitionReplicaId,
    ) -> PartitionDeploymentId:
        value = await self._redis.hget(
            self._build_deployments_key(plan_version),
            partition_replica_id_to_json(replica_id),
        )
        if value is None:
            raise KeyError(
                f"Partition replica {replica_id} is not deployed "
                f"for plan version {plan_version}"
            )

        deployment_id = partition_deployment_id_from_json(value)
        if deployment_id.replica_id != replica_id:
            raise ValueError("Partition deployment registry ID mismatch")

        return deployment_id

    @classmethod
    def _build_deployments_key(cls, plan_version: PlanVersion) -> str:
        return f"partition-deployments:{plan_version.version_number}"
