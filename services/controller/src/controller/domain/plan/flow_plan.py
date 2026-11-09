from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from controller.domain.profile.model_variant_static_profile import TensorProperty
from shared.model.keys import TensorKey
from shared.model.model_variant import ModelVariantId
from shared.plan.hop_logical_id import PlannedIngressReplicaId, PlannedSinkReplicaId
from shared.plan.plan_version import PlanVersion


@dataclass(frozen=True, slots=True)
class FlowPlan:
    plan_version: PlanVersion
    model_variant_id: ModelVariantId
    input_tensor_properties: Mapping[TensorKey, TensorProperty]
    ingress_replica_id: PlannedIngressReplicaId
    sink_replica_id: PlannedSinkReplicaId

    def __post_init__(self) -> None:
        if (
            self.ingress_replica_id.model_variant_id != self.model_variant_id
            or self.sink_replica_id.model_variant_id != self.model_variant_id
        ):
            raise ValueError(
                "Model variant id must be consistent across ingress and sink"
            )
        if not self.input_tensor_properties:
            raise ValueError("Flow plan must contain model input tensor properties")

        object.__setattr__(
            self,
            "input_tensor_properties",
            MappingProxyType(dict(self.input_tensor_properties)),
        )
