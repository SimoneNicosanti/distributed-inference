from collections.abc import Iterator, Sequence

from controller.application.deployment.builder.endpoint_replica_id_builder import (
    EndpointIdsByFlow,
    EndpointReplicaIds,
)
from controller.application.deployment.builder.partition_replica_id_builder import (
    ReplicaIdsByFlow,
)
from controller.domain.hop.hop import ExternalHop, InvocableHop, SourceHop
from controller.domain.optimization.solving.optimization_solution import (
    FlowOptimizationSolution,
    OptimizationSolution,
)
from controller.domain.plan.hop_routing_plan import (
    HopAddress,
    HopInputRouting,
    HopOutputRouting,
    HopRoutingPlan,
    HopRoutings,
)
from shared.plan.hop_logical_id import PlannedPartitionReplicaId
from shared.service.service import WorkerId


class HopRoutingPlanBuilder:
    @staticmethod
    def build_all(
        worker_ids: Sequence[WorkerId],
        solution: OptimizationSolution,
        replica_ids_by_flow: ReplicaIdsByFlow,
        endpoint_ids_by_flow: EndpointIdsByFlow,
    ) -> dict[WorkerId, HopRoutingPlan]:
        addresses_by_worker: dict[WorkerId, list[HopAddress[InvocableHop]]] = {
            worker_id: [] for worker_id in worker_ids
        }
        routings_by_worker: dict[WorkerId, list[HopRoutings]] = {
            worker_id: [] for worker_id in worker_ids
        }

        for flow in solution.flow_solutions:
            replicas = tuple(
                replica_ids_by_flow[flow.flow_id][component.component_index]
                for component in flow.executable_components
            )
            for (
                worker_id,
                address,
                routing,
            ) in HopRoutingPlanBuilder._build_flow_routings(
                flow, replicas, endpoint_ids_by_flow[flow.flow_id]
            ):
                addresses_by_worker[worker_id].append(address)
                routings_by_worker[worker_id].append(routing)

        return {
            worker_id: HopRoutingPlan(
                hop_addresses=tuple(addresses_by_worker[worker_id]),
                routings=tuple(routings_by_worker[worker_id]),
            )
            for worker_id in worker_ids
        }

    @staticmethod
    def _build_flow_routings(
        flow: FlowOptimizationSolution,
        replicas: Sequence[PlannedPartitionReplicaId],
        endpoints: EndpointReplicaIds,
    ) -> Iterator[tuple[WorkerId, HopAddress[InvocableHop], HopRoutings]]:
        components = flow.executable_components
        flow_input_tensors = flow.input_tensors
        flow_result_tensors = flow.result_tensors

        yield (
            flow.ingress_worker_id,
            HopAddress(flow.flow_id, endpoints.ingress),
            HopRoutings(
                input_routings=(
                    HopInputRouting(
                        source=HopAddress(flow.flow_id, ExternalHop()),
                        tensors=flow_input_tensors,
                    ),
                ),
                output_routings=(
                    HopOutputRouting(
                        destination=HopAddress(flow.flow_id, replicas[0]),
                        tensors=flow_input_tensors,
                    ),
                ),
            ),
        )

        for index, component in enumerate(components):
            previous_hop: SourceHop = (
                replicas[index - 1] if index > 0 else endpoints.ingress
            )
            next_replica = (
                replicas[index + 1] if index + 1 < len(replicas) else endpoints.sink
            )
            incoming_tensors = (
                flow_input_tensors
                if index == 0
                else flow.tensors_between(components[index - 1], component)
            )
            outgoing_tensors = (
                flow_result_tensors
                if index + 1 == len(replicas)
                else flow.tensors_between(component, components[index + 1])
            )
            yield (
                component.worker_id,
                HopAddress(flow.flow_id, replicas[index]),
                HopRoutings(
                    input_routings=(
                        HopInputRouting(
                            source=HopAddress(flow.flow_id, previous_hop),
                            tensors=incoming_tensors,
                        ),
                    ),
                    output_routings=(
                        HopOutputRouting(
                            destination=HopAddress(flow.flow_id, next_replica),
                            tensors=outgoing_tensors,
                        ),
                    ),
                ),
            )

        yield (
            flow.sink_worker_id,
            HopAddress(flow.flow_id, endpoints.sink),
            HopRoutings(
                input_routings=(
                    HopInputRouting(
                        source=HopAddress(flow.flow_id, replicas[-1]),
                        tensors=flow_result_tensors,
                    ),
                ),
                output_routings=(),
            ),
        )
