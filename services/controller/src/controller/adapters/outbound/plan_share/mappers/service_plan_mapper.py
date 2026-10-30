from controller.domain.plan.service_plan import ServicePlan
from integration.dto.worker_plan.service_plan_dto import (
    DeploymentPlanDto,
    FlowPlanDto,
    IngressInputRouteDto,
    IngressPlanDto,
    PartitionExecutionPlanDto,
    PartitionIOSchemeDto,
    PriorityKeyDto,
    ReplicaInputBindingDto,
    ReplicaOutputRouteDto,
    ReplicaRoutingDto,
    ResourceAllocationDto,
    RoutingPlanDto,
    SchedulingPlanDto,
    ServicePlanDto,
    WorkerPlanDto,
)


class ServicePlanMapper:
    @staticmethod
    def to_service_plan_dto(plan: ServicePlan) -> ServicePlanDto:
        worker = plan.worker_plan
        ingress = plan.ingress_plan

        return ServicePlanDto(
            worker_plan=WorkerPlanDto(
                plan_version=worker.plan_version,
                deployment_plan=DeploymentPlanDto(
                    replica_ids=worker.deployment_plan.replica_ids,
                    allocations=tuple(
                        ResourceAllocationDto(use_gpu=allocation.use_gpu)
                        for allocation in worker.deployment_plan.allocations
                    ),
                ),
                scheduling_plan=SchedulingPlanDto(
                    priority_keys=tuple(
                        PriorityKeyDto(
                            flow_id=key.flow_id,
                            partition_replica_id=key.partition_replica_id,
                        )
                        for key in worker.scheduling_plan.priority_keys
                    ),
                    priorities=worker.scheduling_plan.priorities,
                ),
                partition_execution_plan=PartitionExecutionPlanDto(
                    partition_ids=worker.partition_execution_plan.partition_ids,
                    schemes=tuple(
                        PartitionIOSchemeDto(
                            inputs=scheme.inputs, outputs=scheme.outputs
                        )
                        for scheme in worker.partition_execution_plan.schemes
                    ),
                ),
                routing_plan=RoutingPlanDto(
                    replica_ids=worker.routing_plan.replica_ids,
                    routings=tuple(
                        ReplicaRoutingDto(
                            input_bindings=tuple(
                                ReplicaInputBindingDto(
                                    prev_replica_id=binding.prev_replica_id,
                                    tensors=binding.tensors,
                                )
                                for binding in routing.input_bindings
                            ),
                            output_routes=tuple(
                                ReplicaOutputRouteDto(
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
                else IngressPlanDto(
                    plan_version=ingress.plan_version,
                    flow_ids=ingress.flow_ids,
                    flow_plans=tuple(
                        FlowPlanDto(
                            model_variant_id=flow_plan.model_variant_id,
                            input_routes=tuple(
                                IngressInputRouteDto(
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
