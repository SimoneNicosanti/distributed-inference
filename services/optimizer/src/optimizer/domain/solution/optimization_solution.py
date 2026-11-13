import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from optimizer.domain.problem.other.backend import Backend
from shared.flow.flow import FlowId
from shared.model.keys import LayerKey, TensorKey
from shared.model.model_variant import ModelVariantId
from shared.service.service import WorkerId


@dataclass(frozen=True, slots=True)
class ComponentTensorContract:
    input_tensor_keys: tuple[TensorKey, ...]
    output_tensor_keys: tuple[TensorKey, ...]

    def __post_init__(self) -> None:
        for field_name in ("input_tensor_keys", "output_tensor_keys"):
            tensor_keys = tuple(getattr(self, field_name))
            if len(tensor_keys) != len(set(tensor_keys)):
                raise ValueError("Component tensor keys must be unique")
            object.__setattr__(self, field_name, tuple(sorted(tensor_keys)))


@dataclass(frozen=True, slots=True)
class ComponentSolution:
    component_index: int
    worker_id: WorkerId
    backend: Backend

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
        object.__setattr__(self, "backend", Backend(self.backend))
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
    ingress_worker_id: WorkerId
    sink_worker_id: WorkerId
    model_variant_id: ModelVariantId
    components: tuple[ComponentSolution, ...]
    cuts: tuple[CutSolution, ...]

    mean_batch_fill_time_s: float
    mean_response_time_s: float
    input_shapes: Mapping[TensorKey, tuple[int, ...]] | None = None

    def __post_init__(self) -> None:
        if self.input_shapes is not None:
            if not self.input_shapes or any(
                type(dimension) is not int or dimension <= 0
                for shape in self.input_shapes.values()
                for dimension in shape
            ):
                raise ValueError(
                    "Selected input shapes must have positive integer dimensions"
                )
            object.__setattr__(
                self, "input_shapes", MappingProxyType(dict(self.input_shapes))
            )
        # TODO: Validate exactly one cut for each consecutive component pair,
        # with no duplicate cuts or references to components outside this flow.
        if not self.components:
            raise ValueError("A flow solution must contain at least one component")
        if self.components[0].worker_id != self.ingress_worker_id:
            raise ValueError("The first component must be placed on the ingress worker")
        if self.components[-1].worker_id != self.sink_worker_id:
            raise ValueError("The last component must be placed on the sink worker")
        times = self.mean_batch_fill_time_s, self.mean_response_time_s
        if any(not math.isfinite(value) or value < 0 for value in times):
            raise ValueError("Flow times must be finite and non-negative")

        component_indexes = [component.component_index for component in self.components]
        if component_indexes != sorted(component_indexes):
            raise ValueError("Flow components must be ordered by component index")
        if len(component_indexes) != len(set(component_indexes)):
            raise ValueError("Flow component indexes must be unique")
        if not self.executable_components:
            raise ValueError(f"Flow {self.flow_id} has no executable components")

    @property
    def executable_components(self) -> tuple[ComponentSolution, ...]:
        return tuple(
            component
            for component in self.components
            if component.executable_layer_keys
        )

    @property
    def input_tensors(self) -> tuple[TensorKey, ...]:
        first_executable = self.executable_components[0]
        position = self.components.index(first_executable)

        if position == 0:
            return first_executable.tensor_contract.input_tensor_keys

        return self._cut_tensors(
            self.components[position - 1],
            first_executable,
        )

    @property
    def result_tensors(self) -> tuple[TensorKey, ...]:
        last_executable = self.executable_components[-1]
        position = self.components.index(last_executable)
        if position + 1 == len(self.components):
            return last_executable.tensor_contract.output_tensor_keys

        return self._cut_tensors(
            last_executable,
            self.components[position + 1],
        )

    def tensors_between(
        self,
        source: ComponentSolution,
        destination: ComponentSolution,
    ) -> tuple[TensorKey, ...]:
        # TODO: Reject executable components between source and destination
        # when collapsing the intermediate components into a direct route.
        source_position = self.components.index(source)
        destination_position = self.components.index(destination)
        if source_position >= destination_position:
            raise ValueError("Source must precede destination")
        cuts = tuple(
            self._cut_tensors(
                self.components[position],
                self.components[position + 1],
            )
            for position in range(source_position, destination_position)
        )
        if any(tensors != cuts[0] for tensors in cuts[1:]):
            raise ValueError(
                "Cannot collapse a non-executable component that changes routed tensors"
            )
        return cuts[0]

    def _cut_tensors(
        self,
        source: ComponentSolution,
        destination: ComponentSolution,
    ) -> tuple[TensorKey, ...]:
        matching = tuple(
            cut
            for cut in self.cuts
            if cut.source_component_index == source.component_index
            and cut.destination_component_index == destination.component_index
        )
        if len(matching) != 1:
            raise ValueError("Each adjacent component pair must have exactly one cut")
        return matching[0].tensor_keys


@dataclass(frozen=True, slots=True)
class OptimizationSolution:
    flow_solutions: tuple[FlowOptimizationSolution, ...]
    objective_value: float
    is_optimal: bool

    def __post_init__(self) -> None:
        flow_ids = [flow_solution.flow_id for flow_solution in self.flow_solutions]
        if len(flow_ids) != len(set(flow_ids)):
            raise ValueError("Flow ids in an optimization solution must be unique")
