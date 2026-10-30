from abc import ABC, abstractmethod

from shared.plan.plan_version import PlanVersion
from worker.domain.plan.hop_routing_plan import HopRoutingPlan
from worker.domain.plan.partition_execution_plan import PartitionExecutionPlan
from worker.domain.plan.scheduling_plan import SchedulingPlan


class SchedulingPlanReader(ABC):
    @abstractmethod
    async def get_scheduling_plan_by_version(
        self, version: PlanVersion
    ) -> SchedulingPlan: ...


class PartitionExecutionPlanReader(ABC):
    @abstractmethod
    async def get_partition_execution_plan_by_version(
        self, version: PlanVersion
    ) -> PartitionExecutionPlan: ...


class HopRoutingPlanReader(ABC):
    @abstractmethod
    async def get_hop_routing_plan_by_version(
        self, version: PlanVersion
    ) -> HopRoutingPlan: ...
