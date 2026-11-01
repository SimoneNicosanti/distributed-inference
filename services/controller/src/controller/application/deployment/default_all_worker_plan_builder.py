from typing import override

from controller.application.deployment.abc.solution_to_plan_converter import (
    FlowPlanMapping,
    SolutionToPlanConverter,
    WorkerPlanMapping,
)
from controller.application.deployment.builder.default_flow_plan_builder import (
    DefaultFlowPlanBuilder,
)
from controller.application.deployment.builder.default_worker_plan_builder import (
    DefaultWorkerPlanBuilder,
)
from controller.application.deployment.builder.endpoint_replica_id_builder import (
    EndpointReplicaIdBuilder,
)
from controller.application.deployment.builder.partition_replica_id_builder import (
    PartitionReplicaIdBuilder,
)
from controller.domain.optimization.solving.optimization_solution import (
    OptimizationSolution,
)
from shared.plan.plan_version import PlanVersion


class DefaultSolutionToPlanConverter(SolutionToPlanConverter):
    def __init__(self) -> None:
        self._replica_id_builder = PartitionReplicaIdBuilder()
        self._worker_plan_builder = DefaultWorkerPlanBuilder()
        self._flow_plan_builder = DefaultFlowPlanBuilder()
        self._endpoint_id_builder = EndpointReplicaIdBuilder()

    @override
    async def convert_solution_to_plans(
        self,
        plan_version: PlanVersion,
        solution: OptimizationSolution,
    ) -> tuple[WorkerPlanMapping, FlowPlanMapping]:
        replica_ids_by_flow = (
            self._replica_id_builder.build_all_partition_replica_ids_from_solution(
                solution
            )
        )
        endpoint_ids_by_flow = self._endpoint_id_builder.build_all(solution)
        worker_plan_mapping = self._worker_plan_builder.build_all(
            plan_version,
            solution,
            replica_ids_by_flow,
            endpoint_ids_by_flow,
        )

        flow_plan_mapping = self._flow_plan_builder.build_all(
            plan_version,
            solution,
            replica_ids_by_flow,
            endpoint_ids_by_flow,
        )

        return worker_plan_mapping, flow_plan_mapping
