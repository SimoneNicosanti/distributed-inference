from collections.abc import Sequence

from controller.application.deployment.builder.endpoint_replica_id_builder import (
    EndpointIdsByFlow,
)
from controller.domain.optimization.solving.optimization_solution import (
    OptimizationSolution,
)
from controller.domain.plan.failure_plan import FailurePlan
from shared.flow.flow import FlowId
from shared.plan.hop_logical_id import PlannedSinkReplicaId
from shared.service.service import WorkerId


class FailurePlanBuilder:
    @staticmethod
    def build_all(
        worker_ids: Sequence[WorkerId],
        solution: OptimizationSolution,
        endpoint_ids_by_flow: EndpointIdsByFlow,
    ) -> dict[WorkerId, FailurePlan]:
        sinks_by_worker: dict[WorkerId, dict[FlowId, PlannedSinkReplicaId]] = {
            worker_id: {} for worker_id in worker_ids
        }
        for flow in solution.flow_solutions:
            flow_workers = {
                flow.ingress_worker_id,
                flow.sink_worker_id,
                *(component.worker_id for component in flow.executable_components),
            }
            for worker_id in flow_workers:
                sinks_by_worker[worker_id][flow.flow_id] = endpoint_ids_by_flow[
                    flow.flow_id
                ].sink

        return {
            worker_id: FailurePlan(
                flow_ids=tuple(sinks),
                sink_replica_ids=tuple(sinks.values()),
            )
            for worker_id, sinks in sinks_by_worker.items()
        }
