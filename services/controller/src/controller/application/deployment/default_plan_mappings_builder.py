from typing import override

from controller.application.deployment.abc.plan_mappings_builder import (
    FlowPlanMapping,
    PlanMappingsBuilder,
    WorkerPlanMapping,
)
from controller.application.deployment.abc.solution_to_plan_converter import (
    SolutionToPlanConverter,
)
from controller.application.ports.outbound.plan_version.plan_version_factory import (
    PlanVersionFactory,
)
from controller.domain.optimization.solving.enriched_optimization_solution import (
    EnrichedOptimizationSolution,
)


class DefaultPlanMappingsBuilder(PlanMappingsBuilder):
    def __init__(
        self,
        plan_version_factory: PlanVersionFactory,
        solution_to_plan_converter: SolutionToPlanConverter,
    ) -> None:
        super().__init__()
        self._plan_version_factory: PlanVersionFactory = plan_version_factory
        self._solution_to_plan_converter: SolutionToPlanConverter = (
            solution_to_plan_converter
        )

    @override
    async def build_plan_mappings(
        self, solution: EnrichedOptimizationSolution
    ) -> tuple[WorkerPlanMapping, FlowPlanMapping]:

        plan_version = await self._plan_version_factory.build_next_plan_version()

        return await self._solution_to_plan_converter.convert_solution_to_plans(
            plan_version,
            solution,
        )
