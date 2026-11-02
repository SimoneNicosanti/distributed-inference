from typing import override

from catalog.api.hop.hop_deployment_directory import (
    HopDeploymentDirectory,
)
from catalog.api.hop.hop_deployment_registrar import (
    HopDeploymentRegistrar,
)
from catalog.redis.hop.keys_factory import HopDeploymentKeyFactory
from integration.shared.id_json import (
    planned_invocable_hop_id_to_json,
    service_id_from_json,
    service_id_to_json,
)
from redis import asyncio as redis_asyncio
from redis.typing import EncodableT, FieldT
from shared.plan.hop import PlannedInvocableHopId
from shared.plan.plan_version import PlanVersion
from shared.service.service import WorkerId


class RedisHopDeploymentCatalog(HopDeploymentRegistrar, HopDeploymentDirectory):
    def __init__(
        self,
        redis: redis_asyncio.Redis,
        key_factory_cls: type[HopDeploymentKeyFactory],
    ) -> None:
        self._redis = redis
        self._key_factory_cls = key_factory_cls

    @override
    async def register_worker_for_planned_invocable_hop[
        PlannedHopIdT: PlannedInvocableHopId
    ](
        self,
        plan_version: PlanVersion,
        planned_hops: tuple[PlannedHopIdT, ...],
        worker_id: WorkerId,
    ) -> None:
        deployments_key = await self._key_factory_cls.build_hop_deployments_key(
            plan_version
        )
        mapping: dict[FieldT, EncodableT] = {
            planned_invocable_hop_id_to_json(planned_hop): service_id_to_json(worker_id)
            for planned_hop in planned_hops
        }
        if not mapping:
            return

        await self._redis.hset(
            deployments_key,
            mapping=mapping,
        )

    @override
    async def get_worker_for_planned_invocable_hop(
        self,
        plan_version: PlanVersion,
        planned_hop: PlannedInvocableHopId,
    ) -> WorkerId:
        deployments_key = await self._key_factory_cls.build_hop_deployments_key(
            plan_version
        )
        value = await self._redis.hget(
            deployments_key,
            planned_invocable_hop_id_to_json(planned_hop),
        )
        if value is None:
            raise KeyError(
                f"Hop replica {planned_hop} is not deployed "
                f"for plan version {plan_version}"
            )

        worker_id = service_id_from_json(value)

        return worker_id
