from collections.abc import Sequence

from controller.application.deployment.builder.partition_replica_id_builder import (
    ReplicaIdsByFlow,
)
from controller.domain.plan.partition_execution_plan import (
    PartitionExecutionPlan,
    PartitionIOScheme,
)
from optimizer.domain.solution.optimization_solution import (
    ComponentSolution,
    OptimizationSolution,
)
from shared.model.model_partition import ModelPartitionId
from shared.service.service import WorkerId


class PartitionExecutionPlanBuilder:
    @staticmethod
    def build_all(
        worker_ids: Sequence[WorkerId],
        solution: OptimizationSolution,
        replica_ids_by_flow: ReplicaIdsByFlow,
    ) -> dict[WorkerId, PartitionExecutionPlan]:
        schemes_by_worker: dict[WorkerId, dict[ModelPartitionId, PartitionIOScheme]] = {
            worker_id: {} for worker_id in worker_ids
        }
        for flow in solution.flow_solutions:
            for component in flow.executable_components:
                replica = replica_ids_by_flow[flow.flow_id][component.component_index]
                PartitionExecutionPlanBuilder._add_partition_scheme(
                    schemes_by_worker[component.worker_id],
                    replica.partition_id,
                    component,
                )

        return {
            worker_id: PartitionExecutionPlan(
                partition_ids=tuple(schemes),
                schemes=tuple(schemes.values()),
            )
            for worker_id, schemes in schemes_by_worker.items()
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
