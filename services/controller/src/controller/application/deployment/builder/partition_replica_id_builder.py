from collections import defaultdict
from collections.abc import Mapping

from optimizer.domain.solution.optimization_solution import (
    OptimizationSolution,
)
from shared.flow.flow import FlowId
from shared.model.model_partition import ModelPartitionId
from shared.plan.hop_logical_id import PlannedPartitionReplicaId

type ReplicaIdsByComponentIndex = Mapping[int, PlannedPartitionReplicaId]
type ReplicaIdsByFlow = Mapping[FlowId, ReplicaIdsByComponentIndex]


class PartitionReplicaIdBuilder:
    @staticmethod
    def build_all(
        solution: OptimizationSolution,
    ) -> ReplicaIdsByFlow:

        next_replica_index: dict[ModelPartitionId, int] = defaultdict(int)
        replica_ids_by_flow: dict[FlowId, dict[int, PlannedPartitionReplicaId]] = {}

        for flow_solution in solution.flow_solutions:
            replica_ids_by_component: dict[int, PlannedPartitionReplicaId] = {}
            for component in flow_solution.executable_components:
                partition_id = ModelPartitionId(
                    model_variant_id=flow_solution.model_variant_id,
                    layers=component.executable_layer_keys,
                )
                replica_ids_by_component[component.component_index] = (
                    PlannedPartitionReplicaId(
                        partition_id=partition_id,
                        replica_idx=next_replica_index[partition_id],
                    )
                )
                next_replica_index[partition_id] += 1

            replica_ids_by_flow[flow_solution.flow_id] = replica_ids_by_component

        return replica_ids_by_flow
