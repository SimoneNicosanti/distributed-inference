from abc import ABC, abstractmethod
from typing import override

from redis_access.contract.redis_address import (
    HashMapRedisAddress,
    RedisAddress,
)
from shared.plan.plan_version import PlanVersion
from shared.service.service import WorkerId


class WorkerPlanAddressFactory[AddrT: RedisAddress](ABC):
    @staticmethod
    @abstractmethod
    def build_versioned_worker_plan_address(
        version: PlanVersion, worker_id: WorkerId
    ) -> AddrT: ...


class HashMapWorkerPlanAddressFactory(WorkerPlanAddressFactory[HashMapRedisAddress]):
    @staticmethod
    @override
    def build_versioned_worker_plan_address(
        version: PlanVersion, worker_id: WorkerId
    ) -> HashMapRedisAddress:

        return HashMapRedisAddress(
            hash_map_key=f"worker-plans:version-{version.version_number}",
            field_key=f"{worker_id.id}",
        )
