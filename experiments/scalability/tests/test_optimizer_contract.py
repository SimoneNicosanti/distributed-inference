"""Current optimizer contracts; legacy experiment tests remain untouched."""

import math
from dataclasses import replace

import pytest
import pytest_asyncio
from scalability_benchmark import instances, provenance
from scalability_benchmark.campaigns import RunSpec
from scalability_benchmark.catalog import (
    CatalogEntry,
    selection_order,
    synthetic_accuracies,
)
from scalability_benchmark.configuration import BENCHMARK_OBJECTIVES
from scalability_benchmark.instances import build_scenario, make_instance
from scalability_benchmark.profiling.static_costs import (
    layer_weight_size_bytes,
    tensor_size_bytes,
)
from scalability_benchmark.randomness import fingerprint
from scalability_benchmark.solver import execute_solver

from controller.domain.backend import Backend as ProfileBackend
from optimizer.application.formulation.compact.misocp.misocp_compact_formulation_builder import (
    MisocpCompactFormulationBuilder,
)
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation_config import (
    MisocpCompactFormulationConfig,
)
from optimizer.domain.problem.compact.compact_optimization_problem import (
    CompactOptimizationProblem,
)
from optimizer.domain.problem.other.backend import Backend
from optimizer.domain.problem.other.flow_endpoint_placement import FixedPlacement
from optimizer.domain.problem.other.local_worker_config import ValidLocalWorkerConfig
from shared.model.model import ModelId
from shared.model.model_variant import ModelVariantId


@pytest_asyncio.fixture
async def scenario(config, make_model, monkeypatch, tmp_path):
    settings = {item.name: item for item in config.catalog.models}
    metrics = synthetic_accuracies(config, 42)
    entries = tuple(
        CatalogEntry(
            settings[name],
            tmp_path / settings[name].filename,
            ModelVariantId(ModelId(name), "fp32-batch1"),
            metrics[name],
            {settings[name].filename: "mock-hash"},
        )
        for name in selection_order(config, 42)
    )

    async def profile(entry, cache, software_hash):
        return make_model(entry)

    monkeypatch.setattr(instances, "discover_catalog", lambda config, seed: entries)
    monkeypatch.setattr(instances, "profile_entry", profile)
    first = await build_scenario(config, 42, tmp_path, "test-software")
    second = await build_scenario(config, 42, tmp_path, "test-software")
    assert fingerprint(first.as_record()) == fingerprint(second.as_record())
    return first


def _spec(scenario, **changes):
    return replace(
        RunSpec("main", 42, scenario.config.main.loads[0], 1, 1, 0, 0, 10), **changes
    )


def test_direct_problem_projection_and_nested_entities(scenario):
    small = make_instance(scenario, _spec(scenario))
    large = make_instance(
        scenario, _spec(scenario, flow_count=7, model_count=10, edge_count=10)
    )
    assert isinstance(small, CompactOptimizationProblem)
    assert (
        len(large.flows) == 7 and len(large.configurations) == len(large.variants) == 10
    )
    assert len(small.connections) == 0 and len(large.connections) == 110
    assert small.variants.items() <= large.variants.items()
    assert small.configurations.items() <= large.configurations.items()
    for key, times in small.execution_times_s.items():
        assert large.execution_times_s[key] == times
    for worker, data in small.workers.items():
        assert large.workers[worker] == data
    device = scenario.workers[0].id
    for data in large.flows.values():
        assert data.endpoint_placement.ingress_placement == FixedPlacement(device)
        assert data.endpoint_placement.sink_placement == FixedPlacement(device)
        assert data.local_worker_config == ValidLocalWorkerConfig(device, True)
        assert len(data.configurations) == 10
    for model in scenario.models:
        key = next(
            key
            for key in large.configurations
            if key.model_variant_id == model.entry.variant_id
        )
        configuration = large.configurations[key]
        assert dict(key.input_shapes.input_shapes) == dict(
            model.profile.shape_profiles[0].input_shapes.input_shapes
        )
        assert configuration.max_cut_size_bytes == math.ceil(
            model.profile.shape_profiles[0].max_cut_size_mib * 1024**2
        )
        assert configuration.tensor_bytes == {
            tensor: tensor_size_bytes(model.profile, tensor)
            for tensor in model.profile.topology.tensor_routes()
        }
        assert large.variants[key.model_variant_id].weight_bytes_by_layer == {
            layer: layer_weight_size_bytes(model.profile, layer)
            for layer in model.profile.topology.layers()
        }
        for worker in scenario.workers:
            expected = (
                scenario.execution[model.entry.settings.name, worker.tier]
                .profile.shape_execution_profiles[0]
                .backend_profiles[0]
            )
            assert large.workers[worker.id].backend == Backend(expected.backend.value)
            assert (
                large.workers[worker.id].available_inference_memory_bytes
                == worker.resource.available_inference_memory_bytes
            )
            assert large.execution_times_s[key, worker.id] == {
                layer: value.execution_time_s
                for layer, value in expected.layer_profiles.items()
            }
    formulation = MisocpCompactFormulationBuilder().build_formulation(
        large, MisocpCompactFormulationConfig(0.1)
    )
    assert formulation.problem is large
    with pytest.raises(TypeError):
        large.workers[device] = small.workers[device]


def test_calibration_and_objectives_do_not_regenerate_entities(scenario):
    for load, flows in scenario.flows.items():
        assert sum(
            flow.arrival_rate_per_s for flow in flows
        ) * scenario.reference_time_s == pytest.approx(load)
    baseline = make_instance(scenario, _spec(scenario))
    for objective in BENCHMARK_OBJECTIVES:
        problem = make_instance(scenario, _spec(scenario, objective=objective))
        assert replace(problem, policy=baseline.policy) == baseline
        assert problem.policy.objective is objective


def test_explicit_configuration_controls_problem_policy(scenario):
    settings = scenario.config.optimization.model_copy(update={"max_components": 2})
    config = scenario.config.model_copy(update={"optimization": settings})
    problem = make_instance(scenario, _spec(scenario), config=config)
    assert problem.policy.max_components == 2
    assert scenario.config.optimization.max_components == 7


@pytest.mark.parametrize(
    "changes",
    [
        {"seed": 43},
        {"load": 0.9},
        {"flow_count": 8},
        {"model_count": 11},
        {"edge_count": 11},
    ],
)
def test_instance_rejects_invalid_subsets(scenario, changes):
    with pytest.raises(ValueError):
        make_instance(scenario, _spec(scenario, **changes))


def test_missing_gpu_backend_preserves_only_boundary_costs(scenario):
    model = scenario.models[0]
    estimate = scenario.execution[model.entry.settings.name, "edge"]
    shape = estimate.profile.shape_execution_profiles[0]
    wrong_backend = replace(shape.backend_profiles[0], backend=ProfileBackend.CPU)
    estimate = replace(
        estimate,
        profile=replace(
            estimate.profile,
            shape_execution_profiles=(
                replace(shape, backend_profiles=(wrong_backend,)),
            ),
        ),
    )
    scenario = replace(
        scenario,
        execution={**scenario.execution, (model.entry.settings.name, "edge"): estimate},
    )
    problem = make_instance(scenario, _spec(scenario, edge_count=1))
    key = next(iter(problem.configurations))
    edge = scenario.workers[1].id
    assert problem.execution_times_s[key, edge] == {"input": 0.0, "output": 0.0}


@pytest.mark.parametrize("count", [0, 2])
def test_variant_requires_exactly_one_shape(scenario, count):
    model = scenario.models[0]
    model = replace(
        model,
        profile=replace(
            model.profile, shape_profiles=model.profile.shape_profiles * count
        ),
    )
    scenario = replace(scenario, models=(model, *scenario.models[1:]))
    with pytest.raises(ValueError, match="exactly one configuration"):
        make_instance(scenario, _spec(scenario))


def test_accuracy_requirement_filters_candidates(scenario):
    load = scenario.config.main.loads[0]
    request = scenario.flows[load][0]
    maximum = max(model.entry.accuracy.value for model in scenario.models)
    request = replace(
        request,
        performance_requirement=replace(request.performance_requirement, value=maximum),
    )
    scenario = replace(
        scenario, flows={**scenario.flows, load: (request, *scenario.flows[load][1:])}
    )
    problem = make_instance(scenario, _spec(scenario, model_count=10))
    assert len(problem.flows[request.flow_id].configurations) == 1
    request = replace(
        request,
        performance_requirement=replace(request.performance_requirement, value=1.1),
    )
    scenario = replace(
        scenario, flows={**scenario.flows, load: (request, *scenario.flows[load][1:])}
    )
    with pytest.raises(ValueError, match="No candidate"):
        make_instance(scenario, _spec(scenario, model_count=10))


@pytest.mark.asyncio
@pytest.mark.parametrize("objective", BENCHMARK_OBJECTIVES)
async def test_benchmark_executes_native_optimizer_full_strategy(scenario, objective):
    settings = scenario.config.optimization.model_copy(
        update={"max_components": 1, "priority_count": 1}
    )
    config = scenario.config.model_copy(update={"optimization": settings})
    scenario = replace(scenario, config=config)
    result = await execute_solver(
        scenario, _spec(scenario, objective=objective), config
    )
    assert result["solution_extracted"], result
    assert result["feasible_incumbent"] and result["solver_status_name"] == "OPTIMAL"
    assert result["error_type"] is None
    assert result["raw_solver_statistics"] is not None
    assert result["effective_solver_configuration"]["objective"] == objective.value


@pytest.mark.asyncio
async def test_native_infeasibility_keeps_recorded_evidence(scenario):
    settings = scenario.config.optimization.model_copy(
        update={"max_components": 1, "priority_count": 1}
    )
    config = scenario.config.model_copy(update={"optimization": settings})
    device = replace(
        scenario.workers[0],
        resource=replace(
            scenario.workers[0].resource,
            ram_profile=replace(
                scenario.workers[0].resource.ram_profile, available_bytes=0
            ),
        ),
    )
    scenario = replace(scenario, config=config, workers=(device, *scenario.workers[1:]))
    result = await execute_solver(scenario, _spec(scenario), config)
    assert result["termination"] == "proven_infeasible", result
    assert result["infeasibility_proven"] and not result["solution_extracted"]
    assert result["raw_solver_statistics"] is not None


def test_optimizer_source_participates_in_provenance(monkeypatch):
    before = provenance.software_provenance()
    original = provenance.file_digest

    def changed(path):
        value = original(path)
        return (
            value + "changed" if "services/optimizer/src/" in path.as_posix() else value
        )

    monkeypatch.setattr(provenance, "file_digest", changed)
    after = provenance.software_provenance()
    assert any(
        name.startswith("services/optimizer/src/") for name in before["source_files"]
    )
    assert before["source_hash"] != after["source_hash"]
