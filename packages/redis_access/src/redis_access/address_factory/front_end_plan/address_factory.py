from abc import ABC, abstractmethod
from typing import override

from redis_access.contract.redis_address import (
    HashMapRedisAddress,
    RedisAddress,
)
from shared.flow.flow import FlowId
from shared.plan.plan_version import PlanVersion


class FlowPlanAddressFactory[AddrT: RedisAddress](ABC):
    @staticmethod
    @abstractmethod
    async def build_versioned_flow_plan_address(
        version: PlanVersion, flow_id: FlowId
    ) -> AddrT: ...

    @staticmethod
    @abstractmethod
    async def build_flow_active_plan_version(
        flow_id: FlowId,
    ) -> AddrT: ...


class HashMapFlowPlanAddressFactory(FlowPlanAddressFactory[HashMapRedisAddress]):
    @staticmethod
    @override
    async def build_versioned_flow_plan_address(
        version: PlanVersion, flow_id: FlowId
    ) -> HashMapRedisAddress:
        return HashMapRedisAddress(
            hash_map_key=f"flow-plans:{flow_id.flow_id}",
            field_key=f"version-{version.version_number}",
        )

    @staticmethod
    @override
    async def build_flow_active_plan_version(
        flow_id: FlowId,
    ) -> HashMapRedisAddress:
        return HashMapRedisAddress(
            hash_map_key=f"flow-plans:{flow_id.flow_id}", field_key="active-version"
        )
