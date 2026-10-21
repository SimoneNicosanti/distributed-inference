from collections import defaultdict
from collections.abc import Mapping

from controller.application.deployment.builder.flow_solution_routes import (
    executable_components,
)
from controller.domain.optimization.optimization_solution import OptimizationSolution
from shared.flow.flow import FlowId
from shared.model.model_partition import ModelPartitionId
from shared.plan.partition_replica_id import PartitionReplicaId

type ReplicaIdsByComponentIndex = Mapping[int, PartitionReplicaId]
type ReplicaIdsByFlow = Mapping[FlowId, ReplicaIdsByComponentIndex]


class PartitionReplicaIdBuilder:
    @staticmethod
    def build_all_partition_replica_ids_from_solution(
        solution: OptimizationSolution,
    ) -> ReplicaIdsByFlow:
        flow_ids = [flow.flow_id for flow in solution.flow_solutions]
        if len(flow_ids) != len(set(flow_ids)):
            raise ValueError("Flow ids in an optimization solution must be unique")

        next_replica_index: dict[ModelPartitionId, int] = defaultdict(int)
        replica_ids_by_flow: dict[FlowId, dict[int, PartitionReplicaId]] = {}

        for flow_solution in solution.flow_solutions:
            replica_ids_by_component: dict[int, PartitionReplicaId] = {}
            for component in executable_components(flow_solution):
                partition_id = ModelPartitionId(
                    model_variant_id=flow_solution.model_variant_id,
                    layers=component.executable_layer_keys,
                )
                replica_ids_by_component[component.component_index] = (
                    PartitionReplicaId(
                        partition_id=partition_id,
                        replica_idx=next_replica_index[partition_id],
                    )
                )
                next_replica_index[partition_id] += 1

            replica_ids_by_flow[flow_solution.flow_id] = replica_ids_by_component

        return replica_ids_by_flow
