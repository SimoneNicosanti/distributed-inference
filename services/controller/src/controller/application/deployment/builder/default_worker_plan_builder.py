from collections import defaultdict
from collections.abc import Sequence

from controller.application.deployment.builder.endpoint_replica_id_builder import (
    EndpointIdsByFlow,
)
from controller.application.deployment.builder.flow_solution_routes import (
    executable_components,
    input_tensors,
    result_tensors,
    tensors_between,
)
from controller.application.deployment.builder.partition_replica_id_builder import (
    ReplicaIdsByFlow,
)
from controller.domain.backend import Backend
from controller.domain.hop.hop import ExternalHop, InvocableHop, SourceHop
from controller.domain.optimization.solving.optimization_solution import (
    ComponentSolution,
    OptimizationSolution,
)
from controller.domain.plan.deployment_plan import DeploymentPlan, ResourceAllocation
from controller.domain.plan.failure_plan import FailurePlan
from controller.domain.plan.hop_routing_plan import (
    HopAddress,
    HopInputRouting,
    HopOutputRouting,
    HopRoutingPlan,
    HopRoutings,
)
from controller.domain.plan.partition_execution_plan import (
    PartitionExecutionPlan,
    PartitionIOScheme,
)
from controller.domain.plan.scheduling_plan import PriorityKey, SchedulingPlan
from controller.domain.plan.worker_plan import WorkerPlan
from shared.flow.flow import FlowId
from shared.model.model_partition import ModelPartitionId
from shared.plan.hop_logical_id import (
    PlannedIngressReplicaId,
    PlannedPartitionReplicaId,
    PlannedSinkReplicaId,
)
from shared.plan.plan_version import PlanVersion
from shared.service.service import WorkerId


class DefaultWorkerPlanBuilder:
    def build_all(
        self,
        plan_version: PlanVersion,
        solution: OptimizationSolution,
        replica_ids_by_flow: ReplicaIdsByFlow,
        endpoint_ids_by_flow: EndpointIdsByFlow,
    ) -> dict[WorkerId, WorkerPlan]:
        involved_workers: dict[WorkerId, None] = {}
        replicas_by_worker: dict[WorkerId, list[PlannedPartitionReplicaId]] = (
            defaultdict(list)
        )
        allocations_by_worker: dict[WorkerId, list[ResourceAllocation]] = defaultdict(
            list
        )
        ingress_by_worker: dict[WorkerId, list[PlannedIngressReplicaId]] = defaultdict(
            list
        )
        sink_by_worker: dict[WorkerId, list[PlannedSinkReplicaId]] = defaultdict(list)
        addresses_by_worker: dict[WorkerId, list[HopAddress[InvocableHop]]] = (
            defaultdict(list)
        )
        routings_by_worker: dict[WorkerId, list[HopRoutings]] = defaultdict(list)
        priority_keys_by_worker: dict[WorkerId, list[PriorityKey]] = defaultdict(list)
        priorities_by_worker: dict[WorkerId, list[int]] = defaultdict(list)
        schemes_by_worker: dict[WorkerId, dict[ModelPartitionId, PartitionIOScheme]] = (
            defaultdict(dict)
        )
        failure_sinks_by_worker: dict[WorkerId, dict[FlowId, PlannedSinkReplicaId]] = (
            defaultdict(dict)
        )

        for flow_solution in solution.flow_solutions:
            involved_workers.setdefault(flow_solution.ingress_worker_id, None)
            involved_workers.setdefault(flow_solution.sink_worker_id, None)
            endpoints = endpoint_ids_by_flow[flow_solution.flow_id]
            ingress_by_worker[flow_solution.ingress_worker_id].append(endpoints.ingress)
            sink_by_worker[flow_solution.sink_worker_id].append(endpoints.sink)
            components = executable_components(flow_solution)
            flow_workers = {
                flow_solution.ingress_worker_id,
                flow_solution.sink_worker_id,
                *(component.worker_id for component in components),
            }
            for worker_id in flow_workers:
                failure_sinks_by_worker[worker_id][flow_solution.flow_id] = (
                    endpoints.sink
                )
            replicas = tuple(
                replica_ids_by_flow[flow_solution.flow_id][component.component_index]
                for component in components
            )

            for component, replica_id in zip(components, replicas, strict=True):
                worker_id = component.worker_id
                involved_workers.setdefault(worker_id, None)
                replicas_by_worker[worker_id].append(replica_id)
                allocations_by_worker[worker_id].append(
                    ResourceAllocation(use_gpu=component.backend is Backend.GPU)
                )
                priority_keys_by_worker[worker_id].append(
                    PriorityKey(
                        flow_id=flow_solution.flow_id,
                        executable_hop=replica_id,
                    )
                )
                priorities_by_worker[worker_id].append(component.priority)
                self._add_partition_scheme(
                    schemes_by_worker[worker_id],
                    replica_id.partition_id,
                    component,
                )

            flow_input_tensors = input_tensors(flow_solution, components[0])
            flow_result_tensors = result_tensors(flow_solution, components[-1])

            addresses_by_worker[flow_solution.ingress_worker_id].append(
                HopAddress(flow_solution.flow_id, endpoints.ingress)
            )
            routings_by_worker[flow_solution.ingress_worker_id].append(
                HopRoutings(
                    input_routings=(
                        HopInputRouting(
                            source=HopAddress(flow_solution.flow_id, ExternalHop()),
                            tensors=flow_input_tensors,
                        ),
                    ),
                    output_routings=(
                        HopOutputRouting(
                            destination=HopAddress(flow_solution.flow_id, replicas[0]),
                            tensors=flow_input_tensors,
                        ),
                    ),
                )
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
                    else tensors_between(
                        flow_solution,
                        components[index - 1],
                        component,
                    )
                )
                outgoing_tensors = (
                    flow_result_tensors
                    if index + 1 == len(replicas)
                    else tensors_between(
                        flow_solution,
                        component,
                        components[index + 1],
                    )
                )
                addresses_by_worker[component.worker_id].append(
                    HopAddress(flow_solution.flow_id, replicas[index])
                )
                routings_by_worker[component.worker_id].append(
                    HopRoutings(
                        input_routings=(
                            HopInputRouting(
                                source=HopAddress(flow_solution.flow_id, previous_hop),
                                tensors=incoming_tensors,
                            ),
                        ),
                        output_routings=(
                            HopOutputRouting(
                                destination=HopAddress(
                                    flow_solution.flow_id, next_replica
                                ),
                                tensors=outgoing_tensors,
                            ),
                        ),
                    )
                )

            addresses_by_worker[flow_solution.sink_worker_id].append(
                HopAddress(flow_solution.flow_id, endpoints.sink)
            )
            routings_by_worker[flow_solution.sink_worker_id].append(
                HopRoutings(
                    input_routings=(
                        HopInputRouting(
                            source=HopAddress(flow_solution.flow_id, replicas[-1]),
                            tensors=flow_result_tensors,
                        ),
                    ),
                    output_routings=(),
                )
            )

        return {
            worker_id: self._build_worker_plan(
                plan_version=plan_version,
                replicas=replicas_by_worker[worker_id],
                allocations=allocations_by_worker[worker_id],
                ingress_hops=ingress_by_worker[worker_id],
                sink_hops=sink_by_worker[worker_id],
                addresses=addresses_by_worker[worker_id],
                routings=routings_by_worker[worker_id],
                priority_keys=priority_keys_by_worker[worker_id],
                priorities=priorities_by_worker[worker_id],
                schemes=schemes_by_worker[worker_id],
                failure_sinks=failure_sinks_by_worker[worker_id],
            )
            for worker_id in involved_workers
        }

    @staticmethod
    def _add_partition_scheme(
        schemes: dict[ModelPartitionId, PartitionIOScheme],
        partition_id: ModelPartitionId,
        component: ComponentSolution,
    ) -> None:
        scheme = PartitionIOScheme(
            inputs=component.tensor_contract.input_tensor_keys,
            outputs=component.tensor_contract.output_tensor_keys,
        )
        existing = schemes.get(partition_id)
        if existing is not None and existing != scheme:
            raise ValueError(
                f"Partition {partition_id} has inconsistent execution schemes"
            )
        schemes.setdefault(partition_id, scheme)

    @staticmethod
    def _build_worker_plan(
        *,
        plan_version: PlanVersion,
        replicas: Sequence[PlannedPartitionReplicaId],
        allocations: Sequence[ResourceAllocation],
        ingress_hops: Sequence[PlannedIngressReplicaId],
        sink_hops: Sequence[PlannedSinkReplicaId],
        addresses: Sequence[HopAddress[InvocableHop]],
        routings: Sequence[HopRoutings],
        priority_keys: Sequence[PriorityKey],
        priorities: Sequence[int],
        schemes: dict[ModelPartitionId, PartitionIOScheme],
        failure_sinks: dict[FlowId, PlannedSinkReplicaId],
    ) -> WorkerPlan:
        replica_ids = tuple(replicas)
        return WorkerPlan(
            plan_version=plan_version,
            failure_plan=FailurePlan(
                flow_ids=tuple(failure_sinks),
                sink_replica_ids=tuple(failure_sinks.values()),
            ),
            deployment_plan=DeploymentPlan(
                executable_hops=replica_ids,
                allocations=tuple(allocations),
                ingress_hops=tuple(ingress_hops),
                sink_hops=tuple(sink_hops),
            ),
            scheduling_plan=SchedulingPlan(
                priority_keys=tuple(priority_keys),
                priorities=tuple(priorities),
            ),
            partition_execution_plan=PartitionExecutionPlan(
                partition_ids=tuple(schemes),
                schemes=tuple(schemes.values()),
            ),
            hop_routing_plan=HopRoutingPlan(
                hop_addresses=tuple(addresses),
                routings=tuple(routings),
            ),
        )
