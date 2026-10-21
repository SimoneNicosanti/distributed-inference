from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from shared.model.keys import TensorKey
from shared.model.model_variant import ModelVariantId
from shared.plan.partition_replica_id import PartitionReplicaId


class IngressInputRoute(BaseModel):
    model_config = ConfigDict(frozen=True)

    target_replica_id: PartitionReplicaId
    tensors: tuple[TensorKey, ...]

    @model_validator(mode="after")
    def validate_route(self) -> Self:
        if not self.tensors:
            raise ValueError("Ingress input route must carry at least one tensor")
        if len(self.tensors) != len(set(self.tensors)):
            raise ValueError("Ingress input route tensors must be unique")
        return self


class FlowPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_variant_id: ModelVariantId
    input_routes: tuple[IngressInputRoute, ...]
    result_tensors: tuple[TensorKey, ...]

    @model_validator(mode="after")
    def validate_flow_plan(self) -> Self:
        if not self.input_routes:
            raise ValueError("Flow plan must contain at least one input route")

        targets = [route.target_replica_id for route in self.input_routes]
        if len(targets) != len(set(targets)):
            raise ValueError("Flow plan input route targets must be unique")

        partitions = [target.partition_id for target in targets]
        if len(partitions) != len(set(partitions)):
            raise ValueError(
                "Flow plan must contain only one input route per partition"
            )

        if any(
            target.partition_id.model_variant_id != self.model_variant_id
            for target in targets
        ):
            raise ValueError(
                "Flow plan input routes must target the configured model variant"
            )

        if not self.result_tensors:
            raise ValueError("Flow plan must contain at least one result tensor")
        if len(self.result_tensors) != len(set(self.result_tensors)):
            raise ValueError("Flow plan result tensors must be unique")

        return self
