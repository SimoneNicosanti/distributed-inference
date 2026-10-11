from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from shared.model.keys import TensorKey
from shared.plan.partition_replica_id import PartitionReplicaId


class ReplicaInputBinding(BaseModel):
    model_config = ConfigDict(frozen=True)

    prev_replica_id: PartitionReplicaId | None
    tensors: tuple[TensorKey, ...]

    @model_validator(mode="after")
    def validate_input_binding(self) -> Self:
        if len(self.tensors) == 0:
            raise ValueError("Input binding must carry at least one tensor")

        if len(self.tensors) != len(set(self.tensors)):
            raise ValueError("Carried tensors must be unique")

        return self


class ReplicaOutputRoute(BaseModel):
    model_config = ConfigDict(frozen=True)

    next_replica_id: PartitionReplicaId | None
    tensors: tuple[TensorKey, ...]

    @model_validator(mode="after")
    def validate_output_route(self) -> Self:
        if len(self.tensors) == 0:
            raise ValueError("Output route must carry at least one tensor")

        if len(self.tensors) != len(set(self.tensors)):
            raise ValueError("Carried tensors must be unique")

        return self


class ReplicaRouting(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_bindings: tuple[ReplicaInputBinding, ...]
    output_routes: tuple[ReplicaOutputRoute, ...]

    @model_validator(mode="after")
    def validate_routing(self) -> Self:
        prev_replica_ids = [binding.prev_replica_id for binding in self.input_bindings]
        if len(prev_replica_ids) != len(set(prev_replica_ids)):
            raise ValueError("Input binding sources must be unique")

        next_replica_ids = [route.next_replica_id for route in self.output_routes]
        if len(next_replica_ids) != len(set(next_replica_ids)):
            raise ValueError("Output route destinations must be unique")

        input_tensors = [
            tensor for binding in self.input_bindings for tensor in binding.tensors
        ]
        if len(input_tensors) != len(set(input_tensors)):
            raise ValueError("Input tensors must have a single source")

        return self

    def check_inputs_ready(
        self,
        arrived: dict[
            PartitionReplicaId | None,
            set[TensorKey],
        ],
    ) -> bool:
        expected = {
            binding.prev_replica_id: set(binding.tensors)
            for binding in self.input_bindings
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
    ) -> dict[PartitionReplicaId | None, set[TensorKey]]:
        routed = {
            route.next_replica_id: set(route.tensors) for route in self.output_routes
        }

        missing = {
            replica_id: tensors - available_tensors
            for replica_id, tensors in routed.items()
            if not tensors.issubset(available_tensors)
        }
        if missing:
            raise ValueError(f"Missing output tensors: {missing}")

        return routed


class RoutingPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    replica_ids: tuple[PartitionReplicaId, ...]
    routings: tuple[ReplicaRouting, ...]

    @model_validator(mode="after")
    def validate_routing_plan(self) -> Self:

        if len(self.replica_ids) == 0:
            raise ValueError("Routing plan must not be empty")

        if len(self.replica_ids) != len(set(self.replica_ids)):
            raise ValueError("Replicas must be unique")

        if len(self.replica_ids) != len(self.routings):
            raise ValueError("Each replica must have a route")

        return self

    def get_routing_by_replica_id(
        self, replica_id: PartitionReplicaId
    ) -> ReplicaRouting:
        return self.routings[self.replica_ids.index(replica_id)]
