from dataclasses import dataclass

from front_end.domain.model_specs import ModelSpecs
from front_end.domain.tensor_map import TensorMap
from shared.plan.hop_logical_id import (
    IngressReplicaId,
    SinkReplicaId,
)


@dataclass(frozen=True, slots=True)
class FlowPlan:
    model_specs: ModelSpecs

    ingress_replica_id: IngressReplicaId
    sink_replica_id: SinkReplicaId

    def __post_init__(self) -> None:
        if (
            self.ingress_replica_id.model_variant_id
            != self.model_specs.model_variant_id
            or self.sink_replica_id.model_variant_id
            != self.model_specs.model_variant_id
            or self.ingress_replica_id.model_variant_id
            != self.sink_replica_id.model_variant_id
        ):
            raise ValueError(
                "Model variant id must be consistent across ingress and sink"
            )

    def validate_tensor_map(self, tensor_map: TensorMap) -> None:
        self.model_specs.validate_tensor_map(tensor_map)
