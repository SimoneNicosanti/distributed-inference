from abc import ABC, abstractmethod

from shared.plan.plan_version import PlanVersion
from worker.domain.plan.partition_execution_plan import PartitionExecutionPlan


class PartitionExecutionPlanReader(ABC):
    @abstractmethod
    async def get_partition_execution_plan_by_version(
        self, version: PlanVersion
    ) -> PartitionExecutionPlan | None: ...
