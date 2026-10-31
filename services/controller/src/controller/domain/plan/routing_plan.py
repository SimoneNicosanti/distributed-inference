from dataclasses import dataclass

from shared.model.keys import TensorKey
from shared.plan.hop_logical_id import PartitionReplicaId


@dataclass(frozen=True, slots=True)
class ReplicaInputBinding:
    prev_replica_id: PartitionReplicaId | None
    tensors: tuple[TensorKey, ...]

    def __post_init__(self) -> None:
        if not self.tensors:
            raise ValueError("Input binding must carry at least one tensor")
        if len(self.tensors) != len(set(self.tensors)):
            raise ValueError("Carried tensors must be unique")


@dataclass(frozen=True, slots=True)
class ReplicaOutputRoute:
    next_replica_id: PartitionReplicaId | None
    tensors: tuple[TensorKey, ...]

    def __post_init__(self) -> None:
        if not self.tensors:
            raise ValueError("Output route must carry at least one tensor")
        if len(self.tensors) != len(set(self.tensors)):
            raise ValueError("Carried tensors must be unique")


@dataclass(frozen=True, slots=True)
class ReplicaRouting:
    input_bindings: tuple[ReplicaInputBinding, ...]
    output_routes: tuple[ReplicaOutputRoute, ...]

    def __post_init__(self) -> None:
        previous = [binding.prev_replica_id for binding in self.input_bindings]
        if len(previous) != len(set(previous)):
            raise ValueError("Input binding sources must be unique")

        following = [route.next_replica_id for route in self.output_routes]
        if len(following) != len(set(following)):
            raise ValueError("Output route destinations must be unique")

        tensors = [
            tensor for binding in self.input_bindings for tensor in binding.tensors
        ]
        if len(tensors) != len(set(tensors)):
            raise ValueError("Input tensors must have a single source")


@dataclass(frozen=True, slots=True)
class RoutingPlan:
    replica_ids: tuple[PartitionReplicaId, ...]
    routings: tuple[ReplicaRouting, ...]

    def __post_init__(self) -> None:
        if len(self.replica_ids) != len(set(self.replica_ids)):
            raise ValueError("Replicas must be unique")
        if len(self.replica_ids) != len(self.routings):
            raise ValueError("Each replica must have a route")
