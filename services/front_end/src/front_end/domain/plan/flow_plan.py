from dataclasses import dataclass

from shared.model.model_variant import ModelVariantId
from shared.plan.hop_logical_id import (
    IngressReplicaId,
    SinkReplicaId,
)


@dataclass(frozen=True, slots=True)
class FlowPlan:
    model_variant_id: ModelVariantId
    model_input_specs: dict[str, tuple[int, ...]]

    ingress_replica_id: IngressReplicaId
    sink_replica_id: SinkReplicaId

    def __post_init__(self) -> None:
        if (
            self.ingress_replica_id.model_variant_id != self.model_variant_id
            or self.sink_replica_id.model_variant_id != self.model_variant_id
            or self.ingress_replica_id.model_variant_id
            != self.sink_replica_id.model_variant_id
        ):
            raise ValueError(
                "Model variant id must be consistent across ingress and sink"
            )
