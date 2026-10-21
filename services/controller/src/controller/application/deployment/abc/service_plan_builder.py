from abc import ABC, abstractmethod

from controller.domain.optimization.optimization_solution import OptimizationSolution
from controller.domain.plan.service_plan import ServicePlan
from shared.plan.plan_version import PlanVersion
from shared.service.service import WorkerId


class ServicePlanBuilder(ABC):
    @abstractmethod
    async def build_all_plans_from_solution(
        self, plan_version: PlanVersion, solution: OptimizationSolution
    ) -> dict[WorkerId, ServicePlan]: ...
