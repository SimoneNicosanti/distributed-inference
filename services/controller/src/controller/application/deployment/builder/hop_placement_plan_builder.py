from collections.abc import Sequence

from controller.application.deployment.builder.endpoint_replica_id_builder import (
    EndpointIdsByFlow,
)
from controller.application.deployment.builder.partition_replica_id_builder import (
    ReplicaIdsByFlow,
)
from controller.domain.hop.hop import InvocableHop
from controller.domain.optimization.solution.optimization_solution import (
    OptimizationSolution,
)
from controller.domain.plan.hop_placement_plan import HopPlacementPlan
from shared.service.service import WorkerId


class HopPlacementPlanBuilder:
    @staticmethod
    def build_all(
        worker_ids: Sequence[WorkerId],
        solution: OptimizationSolution,
        replica_ids_by_flow: ReplicaIdsByFlow,
        endpoint_ids_by_flow: EndpointIdsByFlow,
    ) -> dict[WorkerId, HopPlacementPlan]:
        placements_by_worker: dict[WorkerId, dict[InvocableHop, WorkerId]] = {
            worker_id: {} for worker_id in worker_ids
        }

        for flow in solution.flow_solutions:
            endpoints = endpoint_ids_by_flow[flow.flow_id]
            placements: dict[InvocableHop, WorkerId] = {
                endpoints.ingress: flow.ingress_worker_id,
            }
            for component in flow.executable_components:
                replica = replica_ids_by_flow[flow.flow_id][component.component_index]
                placements[replica] = component.worker_id
            placements[endpoints.sink] = flow.sink_worker_id

            for worker_id in dict.fromkeys(placements.values()):
                placements_by_worker[worker_id].update(placements)

        return {
            worker_id: HopPlacementPlan(mapping=placements)
            for worker_id, placements in placements_by_worker.items()
        }
