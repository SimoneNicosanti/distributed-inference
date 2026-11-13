from abc import ABC, abstractmethod
from typing import TypeAlias

from controller.domain.optimization.solution.enriched_optimization_solution import (
    EnrichedOptimizationSolution,
)
from controller.domain.plan.flow_plan import FlowPlan
from controller.domain.plan.worker_plan import WorkerPlan
from shared.flow.flow import FlowId
from shared.service.service import WorkerId

WorkerPlanMapping: TypeAlias = dict[WorkerId, WorkerPlan]
FlowPlanMapping: TypeAlias = dict[FlowId, FlowPlan]


class PlanMappingsBuilder(ABC):
    @abstractmethod
    async def build_plan_mappings(
        self, solution: EnrichedOptimizationSolution
    ) -> tuple[WorkerPlanMapping, FlowPlanMapping]: ...
