from collections import defaultdict
from collections.abc import Sequence

from controller.application.deployment.builder.flow_solution_routes import (
    executable_components,
    input_tensors,
    result_tensors,
    tensors_between,
)
from controller.application.deployment.builder.partition_replica_id_builder import (
    ReplicaIdsByFlow,
)
from controller.domain.optimization.optimization_solution import (
    ComponentSolution,
    OptimizationSolution,
)
from controller.domain.plan.deployment_plan import DeploymentPlan, ResourceAllocation
from controller.domain.plan.partition_execution_plan import (
    PartitionExecutionPlan,
    PartitionIOScheme,
)
from controller.domain.plan.routing_plan import (
    ReplicaInputBinding,
    ReplicaOutputRoute,
    ReplicaRouting,
    RoutingPlan,
)
from controller.domain.plan.scheduling_plan import PriorityKey, SchedulingPlan
from controller.domain.plan.worker_plan import WorkerPlan
from shared.model.model_partition import ModelPartitionId
from shared.plan.hop_logical_id import PartitionReplicaId
from shared.plan.plan_version import PlanVersion
from shared.service.service import WorkerId


class DefaultWorkerPlanBuilder:
    def build_all(
        self,
        plan_version: PlanVersion,
        solution: OptimizationSolution,
        replica_ids_by_flow: ReplicaIdsByFlow,
    ) -> dict[WorkerId, WorkerPlan]:
        involved_workers: dict[WorkerId, None] = {}
        replicas_by_worker: dict[WorkerId, list[PartitionReplicaId]] = defaultdict(list)
        routings_by_worker: dict[WorkerId, list[ReplicaRouting]] = defaultdict(list)
        priority_keys_by_worker: dict[WorkerId, list[PriorityKey]] = defaultdict(list)
        priorities_by_worker: dict[WorkerId, list[int]] = defaultdict(list)
        schemes_by_worker: dict[WorkerId, dict[ModelPartitionId, PartitionIOScheme]] = (
            defaultdict(dict)
        )

        for flow_solution in solution.flow_solutions:
            involved_workers.setdefault(flow_solution.ingress_id, None)
            components = executable_components(flow_solution)
            replicas = tuple(
                replica_ids_by_flow[flow_solution.flow_id][component.component_index]
                for component in components
            )

            for component, replica_id in zip(components, replicas, strict=True):
                worker_id = component.worker_id
                involved_workers.setdefault(worker_id, None)
                replicas_by_worker[worker_id].append(replica_id)
                priority_keys_by_worker[worker_id].append(
                    PriorityKey(
                        flow_id=flow_solution.flow_id,
                        partition_replica_id=replica_id,
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

            for index, component in enumerate(components):
                previous_replica = replicas[index - 1] if index > 0 else None
                next_replica = (
                    replicas[index + 1] if index + 1 < len(replicas) else None
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
                routings_by_worker[component.worker_id].append(
                    ReplicaRouting(
                        input_bindings=(
                            ReplicaInputBinding(
                                prev_replica_id=previous_replica,
                                tensors=incoming_tensors,
                            ),
                        ),
                        output_routes=(
                            ReplicaOutputRoute(
                                next_replica_id=next_replica,
                                tensors=outgoing_tensors,
                            ),
                        ),
                    )
                )

        return {
            worker_id: self._build_worker_plan(
                plan_version=plan_version,
                replicas=replicas_by_worker[worker_id],
                routings=routings_by_worker[worker_id],
                priority_keys=priority_keys_by_worker[worker_id],
                priorities=priorities_by_worker[worker_id],
                schemes=schemes_by_worker[worker_id],
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
        replicas: Sequence[PartitionReplicaId],
        routings: Sequence[ReplicaRouting],
        priority_keys: Sequence[PriorityKey],
        priorities: Sequence[int],
        schemes: dict[ModelPartitionId, PartitionIOScheme],
    ) -> WorkerPlan:
        replica_ids = tuple(replicas)
        return WorkerPlan(
            plan_version=plan_version,
            deployment_plan=DeploymentPlan(
                replica_ids=replica_ids,
                allocations=tuple(
                    ResourceAllocation(use_gpu=False) for _ in replica_ids
                ),
            ),
            scheduling_plan=SchedulingPlan(
                priority_keys=tuple(priority_keys),
                priorities=tuple(priorities),
            ),
            partition_execution_plan=PartitionExecutionPlan(
                partition_ids=tuple(schemes),
                schemes=tuple(schemes.values()),
            ),
            routing_plan=RoutingPlan(
                replica_ids=replica_ids,
                routings=tuple(routings),
            ),
        )
