"""Generate once, pin model identities, calibrate on the shared GPU, then pair."""

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from scalability_benchmark.campaigns import RunSpec
from scalability_benchmark.catalog import discover_catalog
from scalability_benchmark.flows import SyntheticFlow
from scalability_benchmark.hardware import select_workers, worker_pool
from scalability_benchmark.instances import Scenario, make_instance
from scalability_benchmark.network import network_pool
from scalability_benchmark.profiling.execution import execution_profile
from scalability_benchmark.profiling.static import profile_entry
from scalability_benchmark.randomness import fingerprint, stable_uuid, stream

from optimizer.domain.problem.compact.compact_optimization_problem import (
    CompactOptimizationProblem,
)
from optimizer.domain.problem.other.flow_endpoint_placement import (
    FixedPlacement,
    FlowEndpointPlacement,
)
from optimizer.domain.problem.other.local_worker_config import ValidLocalWorkerConfig
from priority_effect.configuration import Config, Workload
from shared.flow.flow import FlowId


def jsonable(value: Any) -> Any:
    """Canonical, lossless JSON representation, including structured mapping keys."""
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: jsonable(getattr(value, field.name)) for field in fields(value)
        }
    if isinstance(value, Mapping):
        if all(isinstance(key, str) for key in value):
            return {str(key): jsonable(item) for key, item in sorted(value.items())}
        pairs = [[jsonable(key), jsonable(item)] for key, item in value.items()]
        return sorted(pairs, key=lambda pair: fingerprint(pair[0]))
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((jsonable(item) for item in value), key=fingerprint)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass(frozen=True)
class Spec:
    seed: int
    flow_count: int
    load: float
    workload: str


def planned(config: Config, *, pilot: bool = False) -> tuple[Spec, ...]:
    return tuple(
        Spec(seed, count, load, workload.name)
        for seed in (config.seeds[:1] if pilot else config.seeds)
        for count in (config.flows[:1] if pilot else config.flows)
        for load in (config.pilot_loads if pilot else config.target_loads)
        for workload in config.workloads
    )


def calibrate(
    load: float, services: tuple[float, ...], weights: tuple[float, ...]
) -> tuple[float, ...]:
    if not 0 < load < 1 or not services or len(services) != len(weights):
        raise ValueError(
            "Calibration needs load in (0,1) and matching nonempty vectors"
        )
    if any(not math.isfinite(x) or x <= 0 for x in (*services, *weights)):
        raise ValueError(
            "Reference demands and arrival weights must be finite and positive"
        )
    denominator = sum(w * s for w, s in zip(weights, services, strict=True))
    return tuple(load * w / denominator for w in weights)


def assign_models(
    workload: Workload, count: int, demands: dict[str, float], seed: int
) -> tuple[str, ...]:
    if count in workload.assignments:
        return workload.assignments[count]
    if workload.category == "homogeneous":
        return (workload.models[0],) * count
    ranked = sorted(workload.models, key=lambda name: (demands[name], name))
    # Maximize the profiled demand contrast across the two architecture families.
    pairs = [
        (a, b)
        for a in ranked
        for b in ranked
        if a.startswith("resnet") and b.startswith("efficientnet")
    ]
    if not pairs:
        raise ValueError("Heterogeneous pool requires ResNet and EfficientNet variants")
    left, right = max(
        pairs, key=lambda pair: (abs(demands[pair[0]] - demands[pair[1]]), pair)
    )
    selected = [left, right]
    while len(selected) < count:
        remaining = [name for name in ranked if name not in selected]
        selected.append(
            max(
                remaining,
                key=lambda name: (
                    min(abs(demands[name] - demands[other]) for other in selected),
                    name,
                ),
            )
        )
    if len({demands[name] for name in selected}) < 2:
        raise ValueError(
            "Heterogeneous models have identical generated service demands"
        )
    stream(seed, "priority-model-assignment", workload.name, str(count)).shuffle(
        selected
    )
    return tuple(selected)


async def prepare(
    config: Config, seed: int, cache: Path, software_hash: str
) -> Scenario:
    base = config.benchmark_config()
    entries = sorted(
        discover_catalog(base, seed), key=lambda entry: entry.settings.name
    )
    models = tuple(
        [await profile_entry(entry, cache, software_hash) for entry in entries]
    )
    workers = select_workers(worker_pool(base, seed), 1, 0)
    execution = {
        (model.entry.settings.name, tier): execution_profile(
            model.profile, base.hardware[tier], base.execution, seed, tier
        )
        for model in models
        for tier in ("device", "edge")
    }
    return Scenario(
        base,
        seed,
        models,
        workers,
        network_pool(base, seed, workers),
        execution,
        {},
        0.0,
    )


def paired_problem(
    problem: CompactOptimizationProblem, priority_count: int
) -> CompactOptimizationProblem:
    if priority_count not in (1, 2):
        raise ValueError("Only P=1 and P=2 are supported")
    return replace(
        problem, policy=replace(problem.policy, priority_count=priority_count)
    )


def generate(
    config: Config, prepared: Scenario, spec: Spec
) -> tuple[CompactOptimizationProblem, dict[str, Any]]:
    from priority_effect.reference import reference_estimate

    workload = next(w for w in config.workloads if w.name == spec.workload)
    demands = {
        name: sum(
            layer.execution_time_s
            for layer in estimate.profile.shape_execution_profiles[0]
            .backend_profiles[0]
            .layer_profiles.values()
        )
        for (name, tier), estimate in prepared.execution.items()
        if tier == "edge"
    }
    assignments = assign_models(workload, spec.flow_count, demands, spec.seed)
    services = tuple(demands[name] for name in assignments)
    weights = config.arrival_weights.get(spec.flow_count) or tuple(
        stream(
            spec.seed, "priority-arrival", workload.name, str(spec.flow_count), str(i)
        ).uniform(*config.arrival_weight_range)
        for i in range(spec.flow_count)
    )
    rates = calibrate(spec.load, services, weights)
    by_name = {model.entry.settings.name: model for model in prepared.models}
    device = prepared.workers[0].id
    flows = tuple(
        SyntheticFlow(
            FlowId(stable_uuid(spec.seed, "priority-flow", str(i))),
            rate,
            1.0,
            None,
            prepared.config.flows.max_energy_consumption,
            FlowEndpointPlacement(FixedPlacement(device), FixedPlacement(device)),
            ValidLocalWorkerConfig(device, True),
            by_name[name].entry.variant_id,
        )
        for i, (name, rate) in enumerate(zip(assignments, rates, strict=True))
    )
    selected = tuple(by_name[name] for name in sorted(set(assignments)))
    scenario = replace(
        prepared,
        models=selected,
        flows={spec.load: flows},
        reference_time_s=mean(services),
    )
    run = RunSpec(
        "main",
        spec.seed,
        spec.load,
        spec.flow_count,
        len(selected),
        1,
        0,
        config.solver.timeout_s,
        objective=config.optimization.objective,
    )
    problem = make_instance(scenario, run)
    reference = reference_estimate(problem, config.optimization.relative_error)
    deadlines = reference["response_times_s"]
    problem = replace(
        problem,
        flows={
            flow: replace(
                data,
                max_response_time_s=deadlines[str(flow.flow_id)]
                * config.deadline.multiplier,
            )
            for flow, data in problem.flows.items()
        },
    )
    # Recheck the final deadlines without changing any rate or deadline after solving.
    reference = reference_estimate(problem, config.optimization.relative_error)
    record = scenario.as_record()
    record.update(
        {
            "spec": jsonable(spec),
            "workload": workload.name,
            "category": workload.category,
            "flow_count": spec.flow_count,
            "load": spec.load,
            "seed": spec.seed,
            "assumptions": config.model_dump(mode="json", exclude={"output"}),
            "profiling_assumptions": {
                "execution": prepared.config.execution.model_dump(mode="json"),
                "synthetic_accuracy": prepared.config.catalog.model_dump(mode="json"),
            },
            "arrival_process": "independent_poisson",
            "reference": reference,
            "reference_services_s": dict(
                zip(
                    (str(flow.flow_id.flow_id) for flow in flows), services, strict=True
                )
            ),
            "arrival_weights": weights,
            "service_demand_cv": pstdev(services) / mean(services),
            "profiled_model_demands_s": demands,
            "model_assignments": {
                str(flow.flow_id.flow_id): name
                for flow, name in zip(flows, assignments, strict=True)
            },
            "problem": jsonable(problem),
            "flows": [
                {
                    "flow_id": str(flow.flow_id),
                    "model": assignments[i],
                    "arrival_rate_per_s": data.arrival_rate_per_s,
                    "deadline_s": data.max_response_time_s,
                    "accuracy_requirement": None,
                }
                for i, (flow, data) in enumerate(problem.flows.items())
            ],
        }
    )
    record["problem_fingerprint"] = fingerprint(record["problem"])
    return problem, record
