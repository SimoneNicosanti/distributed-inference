from controller.domain.hop.hop import DestinationHop, ExternalHop, SourceHop
from controller.domain.plan.worker_plan import WorkerPlan
from integration.dto.context.context_dto import (
    DestinationHopDto,
    ExternalHopDto,
    IngressHopDto,
    PartitionReplicaHopDto,
    SinkHopDto,
    SourceHopDto,
)
from integration.dto.worker_plan.worker_plan_dto import (
    DeploymentPlanDto,
    DestinationHopAddressDto,
    HopInputRoutingDto,
    HopOutputRoutingDto,
    HopRoutingPlanDto,
    HopRoutingsDto,
    InvocableHopAddressDto,
    PartitionExecutionPlanDto,
    PartitionIOSchemeDto,
    PriorityKeyDto,
    ResourceAllocationDto,
    SchedulingPlanDto,
    SourceHopAddressDto,
    WorkerPlanDto,
)
from shared.plan.hop_logical_id import (
    PlannedIngressReplicaId,
    PlannedPartitionReplicaId,
    PlannedSinkReplicaId,
)


class WorkerPlanMapper:
    @staticmethod
    def to_worker_plan_dto(plan: WorkerPlan) -> WorkerPlanDto:
        return WorkerPlanDto(
            plan_version=plan.plan_version,
            deployment_plan=DeploymentPlanDto(
                executable_hops=plan.deployment_plan.executable_hops,
                allocations=tuple(
                    ResourceAllocationDto(use_gpu=allocation.use_gpu)
                    for allocation in plan.deployment_plan.allocations
                ),
                ingress_hops=plan.deployment_plan.ingress_hops,
                sink_hops=plan.deployment_plan.sink_hops,
            ),
            scheduling_plan=SchedulingPlanDto(
                priority_keys=tuple(
                    PriorityKeyDto(
                        flow_id=key.flow_id,
                        executable_hop=key.executable_hop,
                    )
                    for key in plan.scheduling_plan.priority_keys
                ),
                priorities=plan.scheduling_plan.priorities,
            ),
            partition_execution_plan=PartitionExecutionPlanDto(
                partition_ids=plan.partition_execution_plan.partition_ids,
                schemes=tuple(
                    PartitionIOSchemeDto(inputs=scheme.inputs, outputs=scheme.outputs)
                    for scheme in plan.partition_execution_plan.schemes
                ),
            ),
            hop_routing_plan=HopRoutingPlanDto(
                hop_addresses=tuple(
                    InvocableHopAddressDto(
                        flow_id=address.flow_id,
                        hop=WorkerPlanMapper._destination_hop_to_dto(address.hop),
                    )
                    for address in plan.hop_routing_plan.hop_addresses
                ),
                routings=tuple(
                    HopRoutingsDto(
                        input_routings=tuple(
                            HopInputRoutingDto(
                                source=SourceHopAddressDto(
                                    flow_id=binding.source.flow_id,
                                    hop=WorkerPlanMapper._source_hop_to_dto(
                                        binding.source.hop
                                    ),
                                ),
                                tensors=binding.tensors,
                            )
                            for binding in routing.input_routings
                        ),
                        output_routings=tuple(
                            HopOutputRoutingDto(
                                destination=DestinationHopAddressDto(
                                    flow_id=route.destination.flow_id,
                                    hop=WorkerPlanMapper._destination_hop_to_dto(
                                        route.destination.hop
                                    ),
                                ),
                                tensors=route.tensors,
                            )
                            for route in routing.output_routings
                        ),
                    )
                    for routing in plan.hop_routing_plan.routings
                ),
            ),
        )

    @staticmethod
    def _source_hop_to_dto(hop: SourceHop) -> SourceHopDto:
        match hop:
            case ExternalHop():
                return ExternalHopDto()
            case PlannedIngressReplicaId():
                return IngressHopDto(ingress_replica_id=hop)
            case PlannedPartitionReplicaId():
                return PartitionReplicaHopDto(partition_replica_id=hop)

    @staticmethod
    def _destination_hop_to_dto(hop: DestinationHop) -> DestinationHopDto:
        match hop:
            case PlannedIngressReplicaId():
                return IngressHopDto(ingress_replica_id=hop)
            case PlannedPartitionReplicaId():
                return PartitionReplicaHopDto(partition_replica_id=hop)
            case PlannedSinkReplicaId():
                return SinkHopDto(sink_replica_id=hop)
