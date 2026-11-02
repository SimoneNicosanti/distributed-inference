from abc import ABC, abstractmethod

from controller.domain.optimization.solving.enriched_optimization_solution import (
    EnrichedOptimizationSolution,
)
from controller.domain.plan.front_end_plan import FlowPlan
from controller.domain.plan.worker_plan import WorkerPlan
from shared.flow.flow import FlowId
from shared.plan.plan_version import PlanVersion
from shared.service.service import WorkerId

type WorkerPlanMapping = dict[WorkerId, WorkerPlan]
type FlowPlanMapping = dict[FlowId, FlowPlan]


class SolutionToPlanConverter(ABC):
    @abstractmethod
    async def convert_solution_to_plans(
        self, plan_version: PlanVersion, solution: EnrichedOptimizationSolution
    ) -> tuple[WorkerPlanMapping, FlowPlanMapping]: ...
