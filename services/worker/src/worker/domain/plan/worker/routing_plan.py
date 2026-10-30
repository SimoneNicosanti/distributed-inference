from dataclasses import dataclass

from shared.model.keys import TensorKey
from shared.plan.partition_replica_id import PartitionReplicaId
from worker.domain.plan.hop.hop import (
    DestinationHop,
    SourceHop,
)


@dataclass(frozen=True, slots=True)
class ReplicaInputBinding:
    source: SourceHop
    tensors: tuple[TensorKey, ...]

    def __post_init__(self) -> None:
        if len(self.tensors) == 0:
            raise ValueError("Input binding must carry at least one tensor")

        if len(self.tensors) != len(set(self.tensors)):
            raise ValueError("Carried tensors must be unique")


@dataclass(frozen=True, slots=True)
class ReplicaOutputRoute:
    destination: DestinationHop
    tensors: tuple[TensorKey, ...]

    def __post_init__(self) -> None:
        if len(self.tensors) == 0:
            raise ValueError("Output route must carry at least one tensor")

        if len(self.tensors) != len(set(self.tensors)):
            raise ValueError("Carried tensors must be unique")


@dataclass(frozen=True, slots=True)
class ReplicaRouting:
    input_bindings: tuple[ReplicaInputBinding, ...]
    output_routes: tuple[ReplicaOutputRoute, ...]

    def __post_init__(self) -> None:
        prev_replica_ids = [binding.source for binding in self.input_bindings]
        if len(prev_replica_ids) != len(set(prev_replica_ids)):
            raise ValueError("Input binding sources must be unique")

        next_replica_ids = [route.destination for route in self.output_routes]
        if len(next_replica_ids) != len(set(next_replica_ids)):
            raise ValueError("Output route destinations must be unique")

        input_tensors = [
            tensor for binding in self.input_bindings for tensor in binding.tensors
        ]
        if len(input_tensors) != len(set(input_tensors)):
            raise ValueError("Input tensors must have a single source")

    def check_inputs_ready(
        self,
        arrived: dict[
            SourceHop,
            set[TensorKey],
        ],
    ) -> bool:
        expected = {
            binding.source: set(binding.tensors) for binding in self.input_bindings
        }

        unexpected_sources = arrived.keys() - expected.keys()
        if unexpected_sources:
            raise ValueError(f"Unexpected input sources: {unexpected_sources}")

        for source, tensors in arrived.items():
            if not tensors.issubset(expected[source]):
                raise ValueError(f"Unexpected tensors from source {source}")

        return all(
            arrived.get(source, set()) == tensors
            for source, tensors in expected.items()
        )

    def route_output_tensors(
        self,
        available_tensors: set[TensorKey],
    ) -> dict[DestinationHop, set[TensorKey]]:
        routed = {route.destination: set(route.tensors) for route in self.output_routes}

        missing = {
            replica_id: tensors - available_tensors
            for replica_id, tensors in routed.items()
            if not tensors.issubset(available_tensors)
        }
        if missing:
            raise ValueError(f"Missing output tensors: {missing}")

        return routed


@dataclass(frozen=True, slots=True)
class RoutingPlan:
    replica_ids: tuple[PartitionReplicaId, ...]
    routings: tuple[ReplicaRouting, ...]

    def __post_init__(self) -> None:
        if len(self.replica_ids) != len(set(self.replica_ids)):
            raise ValueError("Replicas must be unique")

        if len(self.replica_ids) != len(self.routings):
            raise ValueError("Each replica must have a route")

    def get_routing_by_replica_id(
        self, replica_id: PartitionReplicaId
    ) -> ReplicaRouting:
        return self.routings[self.replica_ids.index(replica_id)]
