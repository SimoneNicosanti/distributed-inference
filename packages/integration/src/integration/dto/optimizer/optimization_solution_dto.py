from typing import Annotated, Self

from pydantic import Field, model_validator

from integration.dto.model.profiled_model_variant_dto import ModelInputConcreteShapesDto
from integration.dto.optimizer.common import (
    FiniteFloat,
    FlowReference,
    NonNegativeFloat,
    OptimizerDto,
    PositiveInt,
    WorkerReference,
)
from integration.dto.worker_profiles.execution_profile_dto import BackendDto
from shared.model.keys import LayerKey, TensorKey
from shared.model.model_variant import ModelVariantId


class ComponentTensorContractDto(OptimizerDto):
    input_tensor_keys: tuple[TensorKey, ...]
    output_tensor_keys: tuple[TensorKey, ...]


class ComponentSolutionDto(OptimizerDto):
    component_index: PositiveInt
    worker_id: WorkerReference
    backend: BackendDto
    layer_keys: tuple[LayerKey, ...]
    input_layer_keys: tuple[LayerKey, ...]
    output_layer_keys: tuple[LayerKey, ...]
    tensor_contract: ComponentTensorContractDto
    priority: PositiveInt
    mean_execution_time_s: NonNegativeFloat
    mean_execution_queue_time_s: NonNegativeFloat
    mean_transmission_time_s: NonNegativeFloat
    transmission_latency_s: NonNegativeFloat


class CutSolutionDto(OptimizerDto):
    source_component_index: PositiveInt
    destination_component_index: PositiveInt
    tensor_keys: tuple[TensorKey, ...]


class FlowOptimizationSolutionDto(OptimizerDto):
    flow_id: FlowReference
    ingress_worker_id: WorkerReference
    sink_worker_id: WorkerReference
    model_variant_id: ModelVariantId
    components: Annotated[tuple[ComponentSolutionDto, ...], Field(min_length=1)]
    cuts: tuple[CutSolutionDto, ...]
    mean_batch_fill_time_s: NonNegativeFloat
    mean_response_time_s: NonNegativeFloat
    input_shapes: ModelInputConcreteShapesDto | None = None


class OptimizationSolutionDto(OptimizerDto):
    flow_solutions: tuple[FlowOptimizationSolutionDto, ...]
    objective_value: FiniteFloat
    is_optimal: Annotated[bool, Field(strict=True)]

    @model_validator(mode="after")
    def validate_unique_flows(self) -> Self:
        ids = [flow.flow_id for flow in self.flow_solutions]
        if len(ids) != len(set(ids)):
            raise ValueError("Solution contains duplicate flows")
        return self
