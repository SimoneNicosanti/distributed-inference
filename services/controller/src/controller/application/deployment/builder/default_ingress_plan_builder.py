from collections import defaultdict

from controller.application.deployment.builder.flow_solution_routes import (
    executable_components,
    input_tensors,
    result_tensors,
)
from controller.application.deployment.builder.partition_replica_id_builder import (
    ReplicaIdsByFlow,
)
from controller.domain.optimization.optimization_solution import OptimizationSolution
from controller.domain.plan.flow_plan import FlowPlan, IngressInputRoute
from controller.domain.plan.ingress_plan import IngressPlan
from shared.flow.flow import FlowId
from shared.plan.plan_version import PlanVersion
from shared.service.service import WorkerId


class DefaultIngressPlanBuilder:
    def build_all(
        self,
        plan_version: PlanVersion,
        solution: OptimizationSolution,
        replica_ids_by_flow: ReplicaIdsByFlow,
    ) -> dict[WorkerId, IngressPlan]:
        flow_plans_by_ingress: dict[WorkerId, list[tuple[FlowId, FlowPlan]]] = (
            defaultdict(list)
        )

        for flow_solution in solution.flow_solutions:
            components = executable_components(flow_solution)
            first_component = components[0]
            first_replica = replica_ids_by_flow[flow_solution.flow_id][
                first_component.component_index
            ]
            flow_plans_by_ingress[flow_solution.ingress_id].append(
                (
                    flow_solution.flow_id,
                    FlowPlan(
                        model_variant_id=flow_solution.model_variant_id,
                        input_routes=(
                            IngressInputRoute(
                                target_replica_id=first_replica,
                                tensors=input_tensors(
                                    flow_solution,
                                    first_component,
                                ),
                            ),
                        ),
                        result_tensors=result_tensors(
                            flow_solution,
                            components[-1],
                        ),
                    ),
                )
            )

        return {
            worker_id: IngressPlan(
                plan_version=plan_version,
                flow_ids=tuple(flow_id for flow_id, _ in flow_plans),
                flow_plans=tuple(flow_plan for _, flow_plan in flow_plans),
            )
            for worker_id, flow_plans in flow_plans_by_ingress.items()
        }
