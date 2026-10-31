from integration.dto.worker_plan.worker_plan_dto import WorkerPlanDto
from worker.adapters.context_mapper import (
    destination_hop_from_dto,
    invocable_hop_from_dto,
    source_hop_from_dto,
)
from worker.domain.plan.deployment_plan import DeploymentPlan, ResourceAllocation
from worker.domain.plan.hop_routing_plan import (
    HopAddress,
    HopInputRouting,
    HopOutputRouting,
    HopRoutingPlan,
    HopRoutings,
)
from worker.domain.plan.partition_execution_plan import (
    PartitionExecutionPlan,
    PartitionIOScheme,
)
from worker.domain.plan.scheduling_plan import (
    PriorityKey,
    SchedulingPlan,
)
from worker.domain.plan.worker_plan import WorkerPlan


def worker_plan_from_dto(dto: WorkerPlanDto) -> WorkerPlan:
    return WorkerPlan(
        plan_version=dto.plan_version,
        deployment_plan=DeploymentPlan(
            executable_hops=dto.deployment_plan.executable_hops,
            allocations=tuple(
                ResourceAllocation(use_gpu=allocation.use_gpu)
                for allocation in dto.deployment_plan.allocations
            ),
            ingress_hops=dto.deployment_plan.ingress_hops,
            sink_hops=dto.deployment_plan.sink_hops,
        ),
        scheduling_plan=SchedulingPlan(
            priority_keys=tuple(
                PriorityKey(
                    flow_id=key.flow_id,
                    executable_hop=key.executable_hop,
                )
                for key in dto.scheduling_plan.priority_keys
            ),
            priorities=tuple(dto.scheduling_plan.priorities),
        ),
        partition_execution_plan=PartitionExecutionPlan(
            partition_ids=dto.partition_execution_plan.partition_ids,
            schemes=tuple(
                PartitionIOScheme(inputs=scheme.inputs, outputs=scheme.outputs)
                for scheme in dto.partition_execution_plan.schemes
            ),
        ),
        hop_routing_plan=HopRoutingPlan(
            hop_addresses=tuple(
                HopAddress(
                    flow_id=address.flow_id,
                    hop=invocable_hop_from_dto(address.hop),
                )
                for address in dto.hop_routing_plan.hop_addresses
            ),
            routings=tuple(
                HopRoutings(
                    input_routings=tuple(
                        HopInputRouting(
                            source=HopAddress(
                                flow_id=binding.source.flow_id,
                                hop=source_hop_from_dto(binding.source.hop),
                            ),
                            tensors=binding.tensors,
                        )
                        for binding in routing.input_routings
                    ),
                    output_routings=tuple(
                        HopOutputRouting(
                            destination=HopAddress(
                                flow_id=route.destination.flow_id,
                                hop=destination_hop_from_dto(route.destination.hop),
                            ),
                            tensors=route.tensors,
                        )
                        for route in routing.output_routings
                    ),
                )
                for routing in dto.hop_routing_plan.routings
            ),
        ),
    )
