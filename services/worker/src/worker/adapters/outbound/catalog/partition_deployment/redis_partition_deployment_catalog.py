from typing import override

from redis import asyncio as redis_asyncio

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
            deployment_id.replica_id.model_dump_json(),
            deployment_id.model_dump_json(),
        )

    @override
    async def get_partition_deployment_id(
        self,
        plan_version: PlanVersion,
        replica_id: PartitionReplicaId,
    ) -> PartitionDeploymentId:
        value = await self._redis.hget(
            self._build_deployments_key(plan_version),
            replica_id.model_dump_json(),
        )
        if value is None:
            raise KeyError(
                f"Partition replica {replica_id} is not deployed "
                f"for plan version {plan_version}"
            )

        deployment_id = PartitionDeploymentId.model_validate_json(value)
        if deployment_id.replica_id != replica_id:
            raise ValueError("Partition deployment registry ID mismatch")

        return deployment_id

    @classmethod
    def _build_deployments_key(cls, plan_version: PlanVersion) -> str:
        return f"partition-deployments:{plan_version.version_number}"
