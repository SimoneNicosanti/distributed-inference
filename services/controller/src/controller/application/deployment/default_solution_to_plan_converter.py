from typing import override

from controller.application.deployment.abc.plan_mappings_builder import (
    FlowPlanMapping,
    WorkerPlanMapping,
)
from controller.application.deployment.abc.solution_to_plan_converter import (
    SolutionToPlanConverter,
)
from controller.application.deployment.builder.endpoint_replica_id_builder import (
    EndpointReplicaIdBuilder,
)
from controller.application.deployment.builder.flow_plan_builder import (
    FlowPlanBuilder,
)
from controller.application.deployment.builder.partition_replica_id_builder import (
    PartitionReplicaIdBuilder,
)
from controller.application.deployment.builder.worker_plan_builder import (
    WorkerPlanBuilder,
)
from controller.domain.optimization.solving.enriched_optimization_solution import (
    EnrichedOptimizationSolution,
)
from shared.plan.plan_version import PlanVersion


class DefaultSolutionToPlanConverter(SolutionToPlanConverter):
    def __init__(self) -> None:
        self._replica_id_builder = PartitionReplicaIdBuilder()
        self._worker_plan_builder = WorkerPlanBuilder()
        self._flow_plan_builder = FlowPlanBuilder()
        self._endpoint_id_builder = EndpointReplicaIdBuilder()

    @override
    async def convert_solution_to_plans(
        self,
        plan_version: PlanVersion,
        solution: EnrichedOptimizationSolution,
    ) -> tuple[WorkerPlanMapping, FlowPlanMapping]:
        optimization_solution = solution.solution
        replica_ids_by_flow = self._replica_id_builder.build_all(optimization_solution)
        endpoint_ids_by_flow = self._endpoint_id_builder.build_all(
            optimization_solution
        )
        worker_plan_mapping = self._worker_plan_builder.build_all(
            plan_version,
            optimization_solution,
            replica_ids_by_flow,
            endpoint_ids_by_flow,
        )

        flow_plan_mapping = self._flow_plan_builder.build_all(
            plan_version,
            solution,
            endpoint_ids_by_flow,
        )

        return worker_plan_mapping, flow_plan_mapping
