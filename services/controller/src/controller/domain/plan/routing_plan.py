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
        if not self.tensors:
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
        if not self.tensors:
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

        return self


class RoutingPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    replica_ids: tuple[PartitionReplicaId, ...]
    routings: tuple[ReplicaRouting, ...]

    @model_validator(mode="after")
    def validate_routing_plan(self) -> Self:
        if len(self.replica_ids) != len(set(self.replica_ids)):
            raise ValueError("Replicas must be unique")
        if len(self.replica_ids) != len(self.routings):
            raise ValueError("Each replica must have a route")
        return self
