from collections.abc import Sequence

from controller.application.deployment.builder.endpoint_replica_id_builder import (
    EndpointIdsByFlow,
)
from controller.application.deployment.builder.partition_replica_id_builder import (
    ReplicaIdsByFlow,
)
from controller.domain.backend import Backend
from controller.domain.optimization.solving.optimization_solution import (
    OptimizationSolution,
)
from controller.domain.plan.deployment_plan import DeploymentPlan, ResourceAllocation
from shared.plan.hop_logical_id import (
    PlannedIngressReplicaId,
    PlannedPartitionReplicaId,
    PlannedSinkReplicaId,
)
from shared.service.service import WorkerId


class DeploymentPlanBuilder:
    @staticmethod
    def build_all(
        worker_ids: Sequence[WorkerId],
        solution: OptimizationSolution,
        replica_ids_by_flow: ReplicaIdsByFlow,
        endpoint_ids_by_flow: EndpointIdsByFlow,
    ) -> dict[WorkerId, DeploymentPlan]:
        replicas_by_worker: dict[WorkerId, list[PlannedPartitionReplicaId]] = {
            worker_id: [] for worker_id in worker_ids
        }
        allocations_by_worker: dict[WorkerId, list[ResourceAllocation]] = {
            worker_id: [] for worker_id in worker_ids
        }
        ingress_by_worker: dict[WorkerId, list[PlannedIngressReplicaId]] = {
            worker_id: [] for worker_id in worker_ids
        }
        sink_by_worker: dict[WorkerId, list[PlannedSinkReplicaId]] = {
            worker_id: [] for worker_id in worker_ids
        }

        for flow in solution.flow_solutions:
            endpoints = endpoint_ids_by_flow[flow.flow_id]
            ingress_by_worker[flow.ingress_worker_id].append(endpoints.ingress)
            sink_by_worker[flow.sink_worker_id].append(endpoints.sink)
            for component in flow.executable_components:
                replica = replica_ids_by_flow[flow.flow_id][component.component_index]
                replicas_by_worker[component.worker_id].append(replica)
                allocations_by_worker[component.worker_id].append(
                    ResourceAllocation(use_gpu=component.backend is Backend.GPU)
                )

        return {
            worker_id: DeploymentPlan(
                executable_hops=tuple(replicas_by_worker[worker_id]),
                allocations=tuple(allocations_by_worker[worker_id]),
                ingress_hops=tuple(ingress_by_worker[worker_id]),
                sink_hops=tuple(sink_by_worker[worker_id]),
            )
            for worker_id in worker_ids
        }
