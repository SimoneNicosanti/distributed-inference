from typing import cast, override

from redis import asyncio as redis_asyncio

from front_end.application.ports.outbound.hop_deployment_directory import (
    HopDeploymentDirectory,
)
from integration.redis.hop_deployment.keys_factory import HopDeploymentKeyFactory
from integration.shared.id_json import (
    hop_deployment_id_from_json,
    hop_replica_id_to_json,
)
from shared.plan.hop_deployment_id import HopDeploymentId
from shared.plan.hop_logical_id import IngressReplicaId, SinkReplicaId
from shared.plan.plan_version import PlanVersion


class RedisHopDeploymentDirectoryg(HopDeploymentDirectory):
    def __init__(
        self,
        redis: redis_asyncio.Redis,
        key_factory_cls: type[HopDeploymentKeyFactory],
    ) -> None:
        self._redis = redis
        self._key_factory_cls = key_factory_cls

    @override
    async def get_hop_deployment_id[ReplicaIdT: IngressReplicaId | SinkReplicaId](
        self,
        plan_version: PlanVersion,
        replica_id: ReplicaIdT,
    ) -> HopDeploymentId[ReplicaIdT]:
        deployments_key = await self._key_factory_cls.build_hop_deployments_key(
            plan_version
        )
        value = await self._redis.hget(
            deployments_key,
            hop_replica_id_to_json(replica_id),
        )
        if value is None:
            raise KeyError(
                f"Hop replica {replica_id} is not deployed "
                f"for plan version {plan_version}"
            )

        deployment_id = hop_deployment_id_from_json(value)
        if deployment_id.replica_id != replica_id:
            raise ValueError("Hop deployment registry ID mismatch")

        return cast(HopDeploymentId[ReplicaIdT], deployment_id)
