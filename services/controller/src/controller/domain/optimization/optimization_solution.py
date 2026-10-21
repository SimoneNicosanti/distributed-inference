from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

from controller.domain.profile.model_variant_topology import ComponentTensorContract
from shared.flow.flow import FlowId
from shared.model.keys import LayerKey, TensorKey
from shared.model.model_variant import ModelVariantId
from shared.service.service import WorkerId


class OptimizationSolutionId(BaseModel):
    model_config = ConfigDict(frozen=True)


class ComponentSolution(BaseModel):
    model_config = ConfigDict(frozen=True)

    component_index: PositiveInt
    worker_id: WorkerId

    layer_keys: tuple[LayerKey, ...]
    input_layer_keys: tuple[LayerKey, ...]
    output_layer_keys: tuple[LayerKey, ...]

    tensor_contract: ComponentTensorContract

    priority: PositiveInt

    mean_execution_time_s: NonNegativeFloat
    mean_execution_queue_time_s: NonNegativeFloat
    mean_transmission_time_s: NonNegativeFloat
    transmission_latency_s: NonNegativeFloat

    @model_validator(mode="after")
    def validate_boundary_layers(self) -> Self:
        layer_keys = set(self.layer_keys)
        input_layer_keys = set(self.input_layer_keys)
        output_layer_keys = set(self.output_layer_keys)

        if len(layer_keys) != len(self.layer_keys):
            raise ValueError("Component layer keys must be unique")
        if len(input_layer_keys) != len(self.input_layer_keys):
            raise ValueError("Component input layer keys must be unique")
        if len(output_layer_keys) != len(self.output_layer_keys):
            raise ValueError("Component output layer keys must be unique")

        boundary_layer_keys = input_layer_keys | output_layer_keys
        if not boundary_layer_keys.issubset(layer_keys):
            raise ValueError("Component boundary layers must belong to the component")
        if input_layer_keys & output_layer_keys:
            raise ValueError("Component input and output layers must be disjoint")

        return self

    @property
    def executable_layer_keys(self) -> tuple[LayerKey, ...]:
        boundary_layer_keys = set(self.input_layer_keys) | set(self.output_layer_keys)
        return tuple(
            layer for layer in self.layer_keys if layer not in boundary_layer_keys
        )


class CutSolution(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_component_index: PositiveInt
    destination_component_index: PositiveInt
    tensor_keys: tuple[TensorKey, ...]

    @field_validator("tensor_keys")
    @classmethod
    def canonicalize_tensor_keys(
        cls,
        tensor_keys: tuple[TensorKey, ...],
    ) -> tuple[TensorKey, ...]:
        if len(tensor_keys) != len(set(tensor_keys)):
            raise ValueError("Cut tensor keys must be unique")
        return tuple(sorted(tensor_keys))

    @model_validator(mode="after")
    def validate_adjacent_components(self) -> Self:
        if self.destination_component_index != self.source_component_index + 1:
            raise ValueError("A cut must connect adjacent components")
        return self


class FlowOptimizationSolution(BaseModel):
    model_config = ConfigDict(frozen=True)

    flow_id: FlowId
    ingress_id: WorkerId
    model_variant_id: ModelVariantId
    components: tuple[ComponentSolution, ...]
    cuts: tuple[CutSolution, ...]

    mean_batch_fill_time_s: NonNegativeFloat
    mean_response_time_s: NonNegativeFloat

    @model_validator(mode="after")
    def validate_ingress_placement(self) -> Self:
        if not self.components:
            raise ValueError("A flow solution must contain at least one component")
        return self


class OptimizationSolution(BaseModel):
    model_config = ConfigDict(frozen=True)

    flow_solutions: tuple[FlowOptimizationSolution, ...]
    objective_value: float
    is_optimal: bool
