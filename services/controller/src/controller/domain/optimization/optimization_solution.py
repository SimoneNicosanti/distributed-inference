import math
from dataclasses import dataclass, field
from uuid import UUID, uuid4

from controller.domain.profile.model_variant_topology import ComponentTensorContract
from shared.flow.flow import FlowId
from shared.model.keys import LayerKey, TensorKey
from shared.model.model_variant import ModelVariantId
from shared.service.service import WorkerId


@dataclass(frozen=True, slots=True)
class OptimizationSolutionId:
    value: UUID = field(default_factory=uuid4)


@dataclass(frozen=True, slots=True)
class ComponentSolution:
    component_index: int
    worker_id: WorkerId

    layer_keys: tuple[LayerKey, ...]
    input_layer_keys: tuple[LayerKey, ...]
    output_layer_keys: tuple[LayerKey, ...]

    tensor_contract: ComponentTensorContract

    priority: int

    mean_execution_time_s: float
    mean_execution_queue_time_s: float
    mean_transmission_time_s: float
    transmission_latency_s: float

    def __post_init__(self) -> None:
        if self.component_index <= 0:
            raise ValueError("Component index must be positive")
        if self.priority <= 0:
            raise ValueError("Component priority must be positive")
        times = (
            self.mean_execution_time_s,
            self.mean_execution_queue_time_s,
            self.mean_transmission_time_s,
            self.transmission_latency_s,
        )
        if any(not math.isfinite(value) or value < 0 for value in times):
            raise ValueError("Component times must be finite and non-negative")

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

    @property
    def executable_layer_keys(self) -> tuple[LayerKey, ...]:
        boundary_layer_keys = set(self.input_layer_keys) | set(self.output_layer_keys)
        return tuple(
            layer for layer in self.layer_keys if layer not in boundary_layer_keys
        )


@dataclass(frozen=True, slots=True)
class CutSolution:
    source_component_index: int
    destination_component_index: int
    tensor_keys: tuple[TensorKey, ...]

    def __post_init__(self) -> None:
        if self.source_component_index <= 0 or self.destination_component_index <= 0:
            raise ValueError("Cut component indices must be positive")
        tensor_keys = tuple(self.tensor_keys)
        if len(tensor_keys) != len(set(tensor_keys)):
            raise ValueError("Cut tensor keys must be unique")
        object.__setattr__(self, "tensor_keys", tuple(sorted(tensor_keys)))
        if self.destination_component_index != self.source_component_index + 1:
            raise ValueError("A cut must connect adjacent components")


@dataclass(frozen=True, slots=True)
class FlowOptimizationSolution:
    flow_id: FlowId
    ingress_id: WorkerId
    model_variant_id: ModelVariantId
    components: tuple[ComponentSolution, ...]
    cuts: tuple[CutSolution, ...]

    mean_batch_fill_time_s: float
    mean_response_time_s: float

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("A flow solution must contain at least one component")
        if self.components[0].worker_id != self.ingress_id:
            raise ValueError("The first component must be placed on the ingress worker")
        times = self.mean_batch_fill_time_s, self.mean_response_time_s
        if any(not math.isfinite(value) or value < 0 for value in times):
            raise ValueError("Flow times must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class OptimizationSolution:
    flow_solutions: tuple[FlowOptimizationSolution, ...]
    objective_value: float
    is_optimal: bool
