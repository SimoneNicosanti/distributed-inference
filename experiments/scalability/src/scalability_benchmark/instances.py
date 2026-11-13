"""One full scenario per seed; optimizer inputs are immutable subset views."""

import math
from dataclasses import dataclass
from pathlib import Path

from optimizer.domain.problem.compact.compact_optimization_problem import (
    CompactOptimizationProblem,
    ConfigurationData,
    ConfigurationWorker,
    FlowData,
    ModelConfigurationKey,
    OptimizationPolicy,
    VariantData,
    WorkerData,
)
from optimizer.domain.problem.other.backend import Backend
from optimizer.domain.problem.other.edge_description import EdgeDescription
from optimizer.domain.problem.other.model_input_concrete_shapes import (
    ModelInputConcreteShapes,
)
from optimizer.domain.problem.other.network_connection import NetworkConnection
from scalability_benchmark.campaigns import RunSpec
from scalability_benchmark.catalog import discover_catalog
from scalability_benchmark.configuration import ExperimentConfig
from scalability_benchmark.flows import SyntheticFlow, flow_pool
from scalability_benchmark.hardware import SyntheticWorker, select_workers, worker_pool
from scalability_benchmark.network import network_pool, select_network
from scalability_benchmark.profiling.execution import (
    ExecutionEstimate,
    execution_profile,
)
from scalability_benchmark.profiling.memory import estimate_memory
from scalability_benchmark.profiling.static import (
    StaticModel,
    dto_record,
    profile_entry,
)
from scalability_benchmark.profiling.static_costs import (
    layer_weight_size_bytes,
    tensor_size_bytes,
)
from shared.model.keys import LayerKey


def request_record(request: SyntheticFlow) -> dict[str, object]:
    return {
        "flow_id": str(request.flow_id.flow_id),
        "user_id": str(request.flow_id.user_id.id),
        "arrival_rate_per_s": request.arrival_rate_per_s,
        "deadline_s": request.max_response_time_s,
        "minimum_accuracy": request.performance_requirement.value,
        "dataset": request.performance_requirement.dataset,
        "max_energy_consumption": request.max_energy_consumption,
    }


@dataclass(frozen=True)
class Scenario:
    config: ExperimentConfig
    seed: int
    models: tuple[StaticModel, ...]
    workers: tuple[SyntheticWorker, ...]
    links: tuple[NetworkConnection, ...]
    execution: dict[tuple[str, str], ExecutionEstimate]
    flows: dict[float, tuple[SyntheticFlow, ...]]
    reference_time_s: float

    def as_record(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "assumptions": self.config.assumptions(),
            "reference_time_s": self.reference_time_s,
            "models": [
                {
                    "name": model.entry.settings.name,
                    "artifacts": model.entry.artifacts,
                    "static_profile": dto_record(model.dto),
                    "memory_diagnostic": estimate_memory(model.profile).as_record(),
                }
                for model in self.models
            ],
            "workers": [worker.as_record() for worker in self.workers],
            "network": [
                {
                    "source": str(link.source.id),
                    "target": str(link.target.id),
                    "throughput_bytes_per_sec": link.throughput_bytes_per_sec,
                    "round_trip_time_s": link.round_trip_time_s,
                }
                for link in self.links
            ],
            "execution": [
                {
                    "model": name,
                    "tier": tier,
                    "backend": estimate.profile.shape_execution_profiles[0]
                    .backend_profiles[0]
                    .backend.value,
                    "fallback_operators": estimate.fallback_operators,
                    "layers": {
                        layer: {
                            "execution_time_s": profile.execution_time_s,
                            "service_second_moment_s2": profile.execution_time_s**2,
                            "estimated_traffic_bytes": profile.memory_bytes,
                        }
                        for layer, profile in estimate.profile.shape_execution_profiles[
                            0
                        ]
                        .backend_profiles[0]
                        .layer_profiles.items()
                    },
                }
                for (name, tier), estimate in self.execution.items()
            ],
            "flows": {
                str(load): [request_record(request) for request in flows]
                for load, flows in self.flows.items()
            },
            "device_worker_id": str(self.workers[0].id.id),
            "endpoint_policy": "fixed_device_ingress_and_sink; is_inference_runner=true",
            "projection": "CPU/GPU profiles match simulated hardware; inference memory uses CPU RAM or GPU VRAM",
            "memory_constraint": "production weights plus profiled maximum-cut activations; peak liveness is diagnostic only",
        }


async def build_scenario(
    config: ExperimentConfig, seed: int, cache: Path, software_hash: str
) -> Scenario:
    entries = discover_catalog(config, seed)
    models = tuple(
        [await profile_entry(entry, cache, software_hash) for entry in entries]
    )
    workers = worker_pool(config, seed)
    execution = {
        (model.entry.settings.name, tier): execution_profile(
            model.profile, hardware, config.execution, seed, tier
        )
        for model in models
        for tier, hardware in config.hardware.items()
    }
    reference = execution[config.flows.reference_model, config.flows.reference_tier]
    reference_time = sum(
        profile.execution_time_s
        for profile in reference.profile.shape_execution_profiles[0]
        .backend_profiles[0]
        .layer_profiles.values()
    )
    loads = sorted(
        set(config.main.loads) | set(config.sensitivity.loads) | {config.pilot.load}
    )
    return Scenario(
        config,
        seed,
        models,
        workers,
        network_pool(config, seed, workers),
        execution,
        {
            load: flow_pool(config, seed, load, reference_time, workers[0].id)
            for load in loads
        },
        reference_time,
    )


def make_instance(
    scenario: Scenario,
    spec: RunSpec,
    *,
    config: ExperimentConfig | None = None,
) -> CompactOptimizationProblem:
    """Project seeded profile subsets directly into the optimizer's problem."""
    if (
        spec.seed != scenario.seed
        or not 1 <= spec.flow_count <= len(scenario.flows.get(spec.load, ()))
        or not 1 <= spec.model_count <= len(scenario.models)
    ):
        raise ValueError("Run specification does not belong to this scenario")
    if (spec.edge_count, spec.cloud_count) not in scenario.config.infrastructure:
        raise ValueError("Unknown infrastructure configuration")
    models = scenario.models[: spec.model_count]
    workers = select_workers(scenario.workers, spec.edge_count, spec.cloud_count)
    variants = {}
    configurations = {}
    execution: dict[ConfigurationWorker, dict[LayerKey, float]] = {}
    for model in models:
        profile = model.profile
        if len(profile.shape_profiles) != 1:
            raise ValueError("Compact requires exactly one configuration per variant")
        shape = profile.shape_profiles[0]
        key = ModelConfigurationKey(
            model.entry.variant_id,
            ModelInputConcreteShapes(shape.input_shapes.input_shapes),
        )
        dimensions = tuple(key.input_shapes.input_shapes.values())
        if (
            any(not value for value in dimensions)
            or len({value[0] for value in dimensions}) != 1
        ):
            raise ValueError("Compact inputs must share a non-scalar batch dimension")
        maximum = shape.max_cut_size_mib
        if maximum is None or not math.isfinite(maximum) or maximum <= 0:
            raise ValueError("Compact requires a positive finite max_cut_size_mib")
        topology = profile.topology
        layers = topology.layers_in_topological_order()
        descriptions = topology.layers()
        inputs = tuple(layer for layer in layers if descriptions[layer].is_input)
        outputs = tuple(layer for layer in layers if descriptions[layer].is_output)
        variants[key.model_variant_id] = VariantData(
            accuracies=profile.info.accuracies,
            layers=layers,
            input_layers=inputs,
            output_layers=outputs,
            edges=tuple(
                EdgeDescription(edge.source, edge.target, edge.tensors)
                for _, edge in sorted(topology.edges().items())
            ),
            weight_bytes_by_layer={
                layer: layer_weight_size_bytes(profile, layer) for layer in layers
            },
        )
        configurations[key] = ConfigurationData(
            batch_size=dimensions[0][0],
            tensor_bytes={
                tensor: tensor_size_bytes(profile, tensor)
                for tensor in topology.tensor_routes()
            },
            max_cut_size_bytes=math.ceil(maximum * 1024**2),
        )
        for worker in workers:
            measured = scenario.execution[
                model.entry.settings.name, worker.tier
            ].profile
            shapes = tuple(
                item
                for item in measured.shape_execution_profiles
                if item.input_shapes == shape.input_shapes
            )
            if len(shapes) > 1:
                raise ValueError("Duplicate execution shape profiles")
            backends = (
                tuple(
                    item
                    for item in shapes[0].backend_profiles
                    if item.backend.value == worker.resource.execution_backend.value
                )
                if shapes
                else ()
            )
            if len(backends) > 1:
                raise ValueError("Duplicate execution backend profiles")
            execution[key, worker.id] = {
                **(
                    {
                        layer: item.execution_time_s
                        for layer, item in backends[0].layer_profiles.items()
                    }
                    if backends
                    else {}
                ),
                **dict.fromkeys((*inputs, *outputs), 0.0),
            }
    flows = {}
    for request in scenario.flows[spec.load][: spec.flow_count]:
        requirement = request.performance_requirement
        eligible = tuple(
            key
            for key in configurations
            if any(
                metric.type == requirement.type
                and metric.dataset == requirement.dataset
                and metric.value >= requirement.value
                for metric in variants[key.model_variant_id].accuracies
            )
        )
        if not eligible:
            raise ValueError("No candidate model meets the flow's accuracy requirement")
        flows[request.flow_id] = FlowData(
            arrival_rate_per_s=request.arrival_rate_per_s,
            max_response_time_s=request.max_response_time_s,
            performance_requirement=requirement,
            max_energy_consumption=request.max_energy_consumption,
            configurations=eligible,
            workers=tuple(worker.id for worker in workers),
            endpoint_placement=request.endpoint_placement,
            local_worker_config=request.local_worker_config,
        )
    settings = (config or scenario.config).optimization
    return CompactOptimizationProblem(
        variants=variants,
        configurations=configurations,
        execution_times_s=execution,
        flows=flows,
        workers={
            worker.id: WorkerData(
                Backend(worker.resource.execution_backend.value),
                worker.resource.available_inference_memory_bytes,
            )
            for worker in workers
        },
        connections=select_network(workers, scenario.links),
        policy=OptimizationPolicy(
            max_components=settings.max_components,
            priority_count=settings.priority_count,
            compute_utilization_limit=settings.compute_utilization_limit,
            network_utilization_limit=settings.network_utilization_limit,
            objective=spec.objective,
        ),
    )
