from controller.application.deployment.builder.endpoint_replica_id_builder import (
    EndpointIdsByFlow,
)
from controller.domain.optimization.solution.enriched_optimization_solution import (
    EnrichedOptimizationSolution,
)
from controller.domain.plan.flow_plan import FlowPlan
from shared.flow.flow import FlowId
from shared.plan.plan_version import PlanVersion


class FlowPlanBuilder:
    @staticmethod
    def build_all(
        plan_version: PlanVersion,
        solution: EnrichedOptimizationSolution,
        endpoint_ids_by_flow: EndpointIdsByFlow,
    ) -> dict[FlowId, FlowPlan]:
        flow_ids = {flow.flow_id for flow in solution.solution.flow_solutions}
        if set(endpoint_ids_by_flow) != flow_ids:
            raise ValueError("Endpoint ids must match the flows in the solution")

        return {
            flow.flow_id: FlowPlan(
                plan_version=plan_version,
                model_variant_id=flow.model_variant_id,
                input_tensor_properties=solution.input_tensor_properties_by_variant[
                    flow.model_variant_id
                ],
                ingress_replica_id=endpoint_ids_by_flow[flow.flow_id].ingress,
                sink_replica_id=endpoint_ids_by_flow[flow.flow_id].sink,
            )
            for flow in solution.solution.flow_solutions
        }
