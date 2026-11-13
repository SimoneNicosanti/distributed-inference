from enum import StrEnum, auto
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from integration.dto.model.profiled_model_variant_dto import ModelInputConcreteShapesDto
from integration.dto.optimization.optimization_dto import (
    FixedPlacementDto,
    FlowEndpointPlacementDto,
    LocalWorkerConfigDto,
    ValidLocalWorkerConfigDto,
)
from integration.dto.optimizer.common import (
    FlowReference,
    NonNegativeFloat,
    NonNegativeInt,
    OptimizerDto,
    PerformanceMetricDto,
    PositiveFloat,
    PositiveInt,
    WorkerReference,
)
from integration.dto.worker_profiles.execution_profile_dto import BackendDto
from shared.model.keys import LayerKey, TensorKey
from shared.model.model_variant import ModelVariantId


class ObjectiveKindDto(StrEnum):
    ARRIVAL_WEIGHTED_RESPONSE_TIME = auto()
    ARRIVAL_WEIGHTED_ACCURACY = auto()
    DEADLINE_VIOLATIONS = auto()
    ARRIVAL_WEIGHTED_NORMALIZED_DELAY = auto()
    CONSTANT = auto()


class OptimizationPolicyDto(OptimizerDto):
    max_components: PositiveInt
    priority_count: PositiveInt
    compute_utilization_limit: Annotated[float, Field(strict=True, gt=0, lt=1)]
    network_utilization_limit: Annotated[float, Field(strict=True, gt=0, lt=1)]
    objective: ObjectiveKindDto


class ModelConfigurationKeyDto(OptimizerDto):
    model_variant_id: ModelVariantId
    input_shapes: ModelInputConcreteShapesDto


def configuration_identity(
    key: ModelConfigurationKeyDto,
) -> tuple[ModelVariantId, tuple[tuple[TensorKey, tuple[int, ...]], ...]]:
    return key.model_variant_id, tuple(sorted(key.input_shapes.input_shapes.items()))


def _unique[T](values: tuple[T, ...], description: str) -> set[T]:
    unique = set(values)
    if len(unique) != len(values):
        raise ValueError(f"Duplicate {description}")
    return unique


class EdgeDataDto(OptimizerDto):
    source: LayerKey
    target: LayerKey
    tensors: tuple[TensorKey, ...]


class VariantDataDto(OptimizerDto):
    model_variant_id: ModelVariantId
    accuracies: tuple[PerformanceMetricDto, ...]
    layers: Annotated[tuple[LayerKey, ...], Field(min_length=1)]
    input_layers: Annotated[tuple[LayerKey, ...], Field(min_length=1)]
    output_layers: Annotated[tuple[LayerKey, ...], Field(min_length=1)]
    edges: tuple[EdgeDataDto, ...]
    weight_bytes_by_layer: dict[LayerKey, NonNegativeInt]

    @model_validator(mode="after")
    def validate_graph(self) -> Self:
        layers = _unique(self.layers, "layers")
        inputs = _unique(self.input_layers, "input layers")
        outputs = _unique(self.output_layers, "output layers")
        if not (inputs | outputs) <= layers or inputs & outputs:
            raise ValueError("Invalid boundary layers")
        if set(self.weight_bytes_by_layer) != layers:
            raise ValueError("Weight costs must cover exactly the declared layers")
        _unique(tuple((edge.source, edge.target) for edge in self.edges), "edges")
        positions = {layer: index for index, layer in enumerate(self.layers)}
        producers: dict[TensorKey, LayerKey] = {}
        for edge in self.edges:
            if edge.source not in layers or edge.target not in layers:
                raise ValueError("Edge references an unknown layer")
            if positions[edge.source] >= positions[edge.target]:
                raise ValueError("Layers must be in topological order")
            _unique(edge.tensors, "edge tensors")
            for tensor in edge.tensors:
                if tensor in producers and producers[tensor] != edge.source:
                    raise ValueError("Tensor has multiple producers")
                producers[tensor] = edge.source
        return self


class ConfigurationDataDto(OptimizerDto):
    key: ModelConfigurationKeyDto
    batch_size: PositiveInt
    tensor_bytes: dict[TensorKey, PositiveInt]
    max_cut_size_bytes: PositiveInt


class ExecutionTimesDto(OptimizerDto):
    configuration: ModelConfigurationKeyDto
    worker_id: WorkerReference
    layer_times_s: dict[LayerKey, NonNegativeFloat]


class WorkerDataDto(OptimizerDto):
    worker_id: WorkerReference
    backend: BackendDto
    available_inference_memory_bytes: NonNegativeInt


class ConnectionDataDto(OptimizerDto):
    source: WorkerReference
    target: WorkerReference
    throughput_bytes_per_sec: PositiveFloat
    round_trip_time_s: NonNegativeFloat


class FlowDataDto(OptimizerDto):
    flow_id: FlowReference
    arrival_rate_per_s: PositiveFloat
    max_response_time_s: PositiveFloat
    performance_requirement: PerformanceMetricDto
    max_energy_consumption: PositiveFloat
    configurations: Annotated[tuple[ModelConfigurationKeyDto, ...], Field(min_length=1)]
    workers: Annotated[tuple[WorkerReference, ...], Field(min_length=1)]
    endpoint_placement: FlowEndpointPlacementDto
    local_worker_config: LocalWorkerConfigDto


class CompactOptimizationProblemDto(OptimizerDto):
    """Transport contract: exactly one configuration for every declared variant."""

    kind: Literal["compact"] = "compact"
    variants: Annotated[tuple[VariantDataDto, ...], Field(min_length=1)]
    configurations: Annotated[tuple[ConfigurationDataDto, ...], Field(min_length=1)]
    execution_times_s: tuple[ExecutionTimesDto, ...]
    flows: Annotated[tuple[FlowDataDto, ...], Field(min_length=1)]
    workers: Annotated[tuple[WorkerDataDto, ...], Field(min_length=1)]
    connections: tuple[ConnectionDataDto, ...]
    policy: OptimizationPolicyDto

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        _unique(
            tuple(variant.model_variant_id for variant in self.variants), "variants"
        )
        workers = _unique(tuple(worker.worker_id for worker in self.workers), "workers")
        _unique(tuple(flow.flow_id for flow in self.flows), "flows")
        configurations = _unique(
            tuple(configuration_identity(item.key) for item in self.configurations),
            "configurations",
        )
        variants = {variant.model_variant_id: variant for variant in self.variants}
        configured_variants = tuple(
            item.key.model_variant_id for item in self.configurations
        )
        if len(configured_variants) != len(set(configured_variants)) or set(
            configured_variants
        ) != set(variants):
            raise ValueError(
                "Compact optimization requires exactly one configuration per variant"
            )
        for configuration in self.configurations:
            variant = variants.get(configuration.key.model_variant_id)
            if variant is None:
                raise ValueError("Configuration references an unknown variant")
            tensors = {tensor for edge in variant.edges for tensor in edge.tensors}
            inputs = {
                tensor
                for edge in variant.edges
                if edge.source in variant.input_layers
                for tensor in edge.tensors
            }
            shapes = configuration.key.input_shapes.input_shapes
            if set(configuration.tensor_bytes) != tensors or set(shapes) != inputs:
                raise ValueError("Configuration tensor keys do not match the graph")
            if not shapes or any(
                not shape or shape[0] != configuration.batch_size
                for shape in shapes.values()
            ):
                raise ValueError("Input shapes disagree with the batch size")
        _unique(
            tuple(
                (configuration_identity(item.configuration), item.worker_id)
                for item in self.execution_times_s
            ),
            "configuration/worker execution costs",
        )
        for cost in self.execution_times_s:
            if (
                configuration_identity(cost.configuration) not in configurations
                or cost.worker_id not in workers
            ):
                raise ValueError(
                    "Execution costs reference an unknown configuration or worker"
                )
            variant = variants[cost.configuration.model_variant_id]
            if not set(cost.layer_times_s) <= set(variant.layers):
                raise ValueError("Execution costs reference an unknown layer")
        _unique(
            tuple((link.source, link.target) for link in self.connections),
            "connections",
        )
        for link in self.connections:
            if link.source == link.target or not {link.source, link.target} <= workers:
                raise ValueError("Invalid directed network connection")
        worker_ids = {str(worker.id) for worker in workers}
        for flow in self.flows:
            flow_workers = _unique(flow.workers, "flow workers")
            keys = _unique(
                tuple(configuration_identity(key) for key in flow.configurations),
                "flow configurations",
            )
            if not flow_workers <= workers or not keys <= configurations:
                raise ValueError("Flow references an unknown worker or configuration")
            allowed = {str(worker.id) for worker in flow_workers}
            for placement in (
                flow.endpoint_placement.ingress_placement,
                flow.endpoint_placement.sink_placement,
            ):
                if (
                    isinstance(placement, FixedPlacementDto)
                    and str(placement.worker_id.id) not in allowed
                ):
                    raise ValueError("Fixed endpoint is outside the flow's workers")
            local = flow.local_worker_config
            if (
                isinstance(local, ValidLocalWorkerConfigDto)
                and str(local.worker_id.id) not in worker_ids
            ):
                raise ValueError("Local worker is not declared in the problem")
        return self


type OptimizationProblemDto = CompactOptimizationProblemDto
