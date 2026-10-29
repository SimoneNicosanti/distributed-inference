from abc import ABC, abstractmethod

from shared.plan.plan_version import PlanVersion
from worker.domain.plan.worker.partition_execution_plan import PartitionExecutionPlan
from worker.domain.plan.worker.routing_plan import RoutingPlan
from worker.domain.plan.worker.scheduling_plan import SchedulingPlan


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


class RoutingPlanReader(ABC):
    @abstractmethod
    async def get_routing_plan_by_version(
        self, version: PlanVersion
    ) -> RoutingPlan: ...
