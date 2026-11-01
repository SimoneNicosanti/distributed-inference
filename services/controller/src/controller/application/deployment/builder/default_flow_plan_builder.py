from controller.application.deployment.builder.endpoint_replica_id_builder import (
    EndpointIdsByFlow,
)
from controller.application.deployment.builder.partition_replica_id_builder import (
    ReplicaIdsByFlow,
)
from controller.domain.optimization.solving.optimization_solution import (
    OptimizationSolution,
)
from controller.domain.plan.front_end_plan import FlowPlan
from shared.flow.flow import FlowId
from shared.plan.plan_version import PlanVersion


class DefaultFlowPlanBuilder:
    def build_all(
        self,
        plan_version: PlanVersion,
        solution: OptimizationSolution,
        replica_ids_by_flow: ReplicaIdsByFlow,
        endpoint_ids_by_flow: EndpointIdsByFlow,
    ) -> dict[FlowId, FlowPlan]: ...
