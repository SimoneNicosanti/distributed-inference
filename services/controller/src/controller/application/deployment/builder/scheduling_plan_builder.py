from collections.abc import Sequence

from controller.application.deployment.builder.partition_replica_id_builder import (
    ReplicaIdsByFlow,
)
from controller.domain.plan.scheduling_plan import PriorityKey, SchedulingPlan
from optimizer.domain.solution.optimization_solution import (
    OptimizationSolution,
)
from shared.service.service import WorkerId


class SchedulingPlanBuilder:
    @staticmethod
    def build_all(
        worker_ids: Sequence[WorkerId],
        solution: OptimizationSolution,
        replica_ids_by_flow: ReplicaIdsByFlow,
    ) -> dict[WorkerId, SchedulingPlan]:
        keys_by_worker: dict[WorkerId, list[PriorityKey]] = {
            worker_id: [] for worker_id in worker_ids
        }
        priorities_by_worker: dict[WorkerId, list[int]] = {
            worker_id: [] for worker_id in worker_ids
        }
        for flow in solution.flow_solutions:
            for component in flow.executable_components:
                replica = replica_ids_by_flow[flow.flow_id][component.component_index]
                keys_by_worker[component.worker_id].append(
                    PriorityKey(flow_id=flow.flow_id, executable_hop=replica)
                )
                priorities_by_worker[component.worker_id].append(component.priority)

        return {
            worker_id: SchedulingPlan(
                priority_keys=tuple(keys_by_worker[worker_id]),
                priorities=tuple(priorities_by_worker[worker_id]),
            )
            for worker_id in worker_ids
        }
