from dataclasses import dataclass

from shared.model.keys import TensorKey
from shared.model.model_variant import ModelVariantId
from shared.plan.partition_replica_id import PartitionReplicaId
from shared.service.service import SinkId


@dataclass(frozen=True, slots=True)
class IngressInputRoute:
    target_replica_id: PartitionReplicaId
    tensors: tuple[TensorKey, ...]

    def __post_init__(self) -> None:
        if not self.tensors:
            raise ValueError("Ingress input route must carry at least one tensor")

        if len(self.tensors) != len(set(self.tensors)):
            raise ValueError("Ingress input route tensors must be unique")


@dataclass(frozen=True, slots=True)
class FlowPlan:
    model_variant_id: ModelVariantId
    input_routes: tuple[IngressInputRoute, ...]
    sink_id: SinkId

    def __post_init__(self) -> None:
        if not self.input_routes:
            raise ValueError("Flow plan must contain at least one input route")

        target_replica_ids = [route.target_replica_id for route in self.input_routes]
        if len(target_replica_ids) != len(set(target_replica_ids)):
            raise ValueError("Flow plan input route targets must be unique")

        target_partition_ids = [
            target_replica_id.partition_id for target_replica_id in target_replica_ids
        ]
        if len(target_partition_ids) != len(set(target_partition_ids)):
            raise ValueError(
                "Flow plan must contain only one input route per partition"
            )

        for input_route in self.input_routes:
            target_replica_id = input_route.target_replica_id
            if target_replica_id.partition_id.model_variant_id != self.model_variant_id:
                raise ValueError(
                    "Flow plan input routes must target the configured model variant"
                )

    def route_input_tensors(
        self,
        available_tensors: set[TensorKey],
    ) -> dict[PartitionReplicaId, set[TensorKey]]:
        routed: dict[PartitionReplicaId, set[TensorKey]] = {}
        missing: dict[PartitionReplicaId, set[TensorKey]] = {}

        for input_route in self.input_routes:
            target_replica_id = input_route.target_replica_id
            route_tensors = set(input_route.tensors)
            routed[target_replica_id] = route_tensors

            missing_tensors = route_tensors - available_tensors
            if missing_tensors:
                missing[target_replica_id] = missing_tensors

        if missing:
            raise ValueError(f"Missing ingress input tensors: {missing}")

        return routed
