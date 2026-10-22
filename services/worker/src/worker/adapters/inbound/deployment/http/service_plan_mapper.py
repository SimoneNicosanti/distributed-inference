from integration.dto.service_plan.service_plan_dto import ServicePlanDto
from worker.domain.plan.ingress.flow_plan import FlowPlan, IngressInputRoute
from worker.domain.plan.ingress.ingress_plan import IngressPlan
from worker.domain.plan.service_plan import ServicePlan
from worker.domain.plan.worker.deployment_plan import DeploymentPlan, ResourceAllocation
from worker.domain.plan.worker.partition_execution_plan import (
    PartitionExecutionPlan,
    PartitionIOScheme,
)
from worker.domain.plan.worker.routing_plan import (
    ReplicaInputBinding,
    ReplicaOutputRoute,
    ReplicaRouting,
    RoutingPlan,
)
from worker.domain.plan.worker.scheduling_plan import (
    PriorityKey,
    SchedulingPlan,
)
from worker.domain.plan.worker.worker_plan import WorkerPlan


def service_plan_from_dto(dto: ServicePlanDto) -> ServicePlan:
    worker = dto.worker_plan
    ingress = dto.ingress_plan

    return ServicePlan(
        worker_plan=WorkerPlan(
            plan_version=worker.plan_version,
            deployment_plan=DeploymentPlan(
                replica_ids=worker.deployment_plan.replica_ids,
                allocations=tuple(
                    ResourceAllocation(use_gpu=allocation.use_gpu)
                    for allocation in worker.deployment_plan.allocations
                ),
            ),
            scheduling_plan=SchedulingPlan(
                priority_keys=tuple(
                    PriorityKey(
                        flow_id=key.flow_id,
                        partition_replica_id=key.partition_replica_id,
                    )
                    for key in worker.scheduling_plan.priority_keys
                ),
                priorities=tuple(worker.scheduling_plan.priorities),
            ),
            partition_execution_plan=PartitionExecutionPlan(
                partition_ids=worker.partition_execution_plan.partition_ids,
                schemes=tuple(
                    PartitionIOScheme(inputs=scheme.inputs, outputs=scheme.outputs)
                    for scheme in worker.partition_execution_plan.schemes
                ),
            ),
            routing_plan=RoutingPlan(
                replica_ids=worker.routing_plan.replica_ids,
                routings=tuple(
                    ReplicaRouting(
                        input_bindings=tuple(
                            ReplicaInputBinding(
                                prev_replica_id=binding.prev_replica_id,
                                tensors=binding.tensors,
                            )
                            for binding in routing.input_bindings
                        ),
                        output_routes=tuple(
                            ReplicaOutputRoute(
                                next_replica_id=route.next_replica_id,
                                tensors=route.tensors,
                            )
                            for route in routing.output_routes
                        ),
                    )
                    for routing in worker.routing_plan.routings
                ),
            ),
        ),
        ingress_plan=(
            None
            if ingress is None
            else IngressPlan(
                plan_version=ingress.plan_version,
                flow_ids=ingress.flow_ids,
                flow_plans=tuple(
                    FlowPlan(
                        model_variant_id=flow_plan.model_variant_id,
                        input_routes=tuple(
                            IngressInputRoute(
                                target_replica_id=route.target_replica_id,
                                tensors=route.tensors,
                            )
                            for route in flow_plan.input_routes
                        ),
                        result_tensors=flow_plan.result_tensors,
                    )
                    for flow_plan in ingress.flow_plans
                ),
            )
        ),
    )
