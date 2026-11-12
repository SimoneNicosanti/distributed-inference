"""One full scenario per seed; optimizer inputs are immutable subset views."""

from dataclasses import dataclass
from pathlib import Path

from controller.domain.optimization.request.optimization_request import (
    OptimizationRequest,
    OptimizationRequestsByFlow,
)
from controller.domain.optimization.solving.optimization_context import (
    OptimizationContext,
)
from controller.domain.profile.model_variant_execution_profile import (
    WorkerExecutionProfiles,
)
from controller.domain.profile.network_topology import NetworkConnection
from scalability_benchmark.campaigns import RunSpec
from scalability_benchmark.catalog import discover_catalog
from scalability_benchmark.configuration import ExperimentConfig
from scalability_benchmark.flows import flow_pool
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


def request_record(request: OptimizationRequest) -> dict[str, object]:
    return {
        "flow_id": str(request.flow_id.flow_id),
        "user_id": str(request.flow_id.user_id.id),
        "arrival_rate_per_s": request.stream_info.arrival_rate,
        "deadline_s": request.qos_requirements.max_mean_response_time,
        "minimum_accuracy": request.qos_requirements.min_model_performance.value,
        "dataset": request.qos_requirements.min_model_performance.dataset,
        "max_energy_consumption": request.qos_requirements.max_energy_consumption,
    }


@dataclass(frozen=True)
class Scenario:
    config: ExperimentConfig
    seed: int
    models: tuple[StaticModel, ...]
    workers: tuple[SyntheticWorker, ...]
    links: tuple[NetworkConnection, ...]
    execution: dict[tuple[str, str], ExecutionEstimate]
    flows: dict[float, tuple[OptimizationRequest, ...]]
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
    scenario: Scenario, spec: RunSpec
) -> tuple[OptimizationRequestsByFlow, OptimizationContext]:
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
    requests = {
        request.flow_id: request
        for request in scenario.flows[spec.load][: spec.flow_count]
    }
    context = OptimizationContext(
        static_profiles_by_variant={
            model.entry.variant_id: model.profile for model in models
        },
        network_topology=select_network(workers, scenario.links),
        resource_profiles_by_worker={worker.id: worker.resource for worker in workers},
        execution_profiles_by_worker={
            worker.id: WorkerExecutionProfiles(
                worker.id,
                {
                    model.entry.variant_id: scenario.execution[
                        model.entry.settings.name, worker.tier
                    ].profile
                    for model in models
                },
            )
            for worker in workers
        },
    )
    return requests, context
