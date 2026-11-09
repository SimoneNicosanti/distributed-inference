from abc import ABC, abstractmethod

from controller.application.deployment.abc.plan_mappings_builder import (
    FlowPlanMapping,
    WorkerPlanMapping,
)
from controller.domain.optimization.solving.enriched_optimization_solution import (
    EnrichedOptimizationSolution,
)
from shared.plan.plan_version import PlanVersion


class SolutionToPlanConverter(ABC):
    @abstractmethod
    async def convert_solution_to_plans(
        self, plan_version: PlanVersion, solution: EnrichedOptimizationSolution
    ) -> tuple[WorkerPlanMapping, FlowPlanMapping]: ...
