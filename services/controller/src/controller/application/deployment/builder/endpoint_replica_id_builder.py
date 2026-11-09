from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

from controller.domain.optimization.solving.optimization_solution import (
    OptimizationSolution,
)
from shared.flow.flow import FlowId
from shared.model.model_variant import ModelVariantId
from shared.plan.hop_logical_id import PlannedIngressReplicaId, PlannedSinkReplicaId


@dataclass(frozen=True, slots=True)
class EndpointReplicaIds:
    ingress: PlannedIngressReplicaId
    sink: PlannedSinkReplicaId


type EndpointIdsByFlow = Mapping[FlowId, EndpointReplicaIds]


class EndpointReplicaIdBuilder:
    @staticmethod
    def build_all(solution: OptimizationSolution) -> EndpointIdsByFlow:
        endpoints: dict[FlowId, EndpointReplicaIds] = {}
        indexes: dict[ModelVariantId, int] = defaultdict(int)
        for flow in solution.flow_solutions:
            if flow.flow_id in endpoints:
                raise ValueError("Flow ids must be unique")
            index = indexes[flow.model_variant_id]
            indexes[flow.model_variant_id] += 1
            endpoints[flow.flow_id] = EndpointReplicaIds(
                ingress=PlannedIngressReplicaId(flow.model_variant_id, index),
                sink=PlannedSinkReplicaId(flow.model_variant_id, index),
            )
        return endpoints
