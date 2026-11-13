import math
from dataclasses import replace
from uuid import UUID

import pytest

from optimizer.application.formulation.compact.misocp.misocp_compact_formulation_builder import (
    MisocpCompactFormulationBuilder,
    MisocpCompactFormulationBuildError,
)
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation_config import (
    MisocpCompactFormulationConfig,
)
from optimizer.domain.problem.compact.compact_optimization_problem import (
    CompactOptimizationProblem,
    ConfigurationData,
    FlowData,
    ModelConfigurationKey,
    ObjectiveKind,
    OptimizationPolicy,
    VariantData,
    WorkerData,
)
from optimizer.domain.problem.other.backend import Backend
from optimizer.domain.problem.other.edge_description import EdgeDescription
from optimizer.domain.problem.other.flow_endpoint_placement import (
    AnyPlacement,
    FlowEndpointPlacement,
)
from optimizer.domain.problem.other.local_worker_config import NoLocalWorkerConfig
from optimizer.domain.problem.other.model_input_concrete_shapes import (
    ModelInputConcreteShapes,
)
from shared.flow.flow import FlowId
from shared.model.model import ModelId
from shared.model.model_variant import (
    ModelVariantId,
    PerformanceMetric,
    PerformanceMetricType,
)
from shared.service.service import ServiceId
from shared.user.user import UserId


def _problem() -> CompactOptimizationProblem:
    flow = FlowId(UserId(UUID(int=1)), UUID(int=2))
    worker = ServiceId(UUID(int=3))
    variant = ModelVariantId(ModelId("model"), "fp32")
    configurations = tuple(
        ModelConfigurationKey(variant, ModelInputConcreteShapes({"x": (batch, 3)}))
        for batch in (1, 4)
    )
    metric = PerformanceMetric(PerformanceMetricType.ACCURACY, "dataset", 0.9)
    return CompactOptimizationProblem(
        variants={
            variant: VariantData(
                accuracies=(metric,),
                layers=("input", "compute", "output"),
                input_layers=("input",),
                output_layers=("output",),
                edges=(
                    EdgeDescription("input", "compute", frozenset({"x"})),
                    EdgeDescription("compute", "output", frozenset({"y"})),
                ),
                weight_bytes_by_layer={"input": 0, "compute": 10, "output": 0},
            )
        },
        configurations={
            key: ConfigurationData(batch, {"x": batch * 12, "y": batch * 4}, batch * 12)
            for key, batch in zip(configurations, (1, 4), strict=True)
        },
        execution_times_s={
            (key, worker): {"input": 0.0, "compute": cost, "output": 0.0}
            for key, cost in zip(configurations, (0.1, 0.2), strict=True)
        },
        flows={
            flow: FlowData(
                1.0,
                10.0,
                metric,
                1000.0,
                configurations,
                (worker,),
                FlowEndpointPlacement(AnyPlacement(), AnyPlacement()),
                NoLocalWorkerConfig(),
            )
        },
        workers={worker: WorkerData(Backend.CPU, 100_000_000)},
        connections=(),
        policy=OptimizationPolicy(
            1, 1, 0.8, 0.7, ObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_TIME
        ),
    )


def _build(problem: CompactOptimizationProblem, error: float = 0.1):
    return MisocpCompactFormulationBuilder().build_formulation(
        problem, MisocpCompactFormulationConfig(error)
    )


def test_compiles_every_shape_with_distinct_coefficients_and_shared_topology():
    problem = _problem()
    compiled = _build(problem)
    assert compiled.problem is problem
    flow = compiled.flows[0]
    one, four = compiled.configurations_by_flow[flow]
    assert compiled.service_arrival_rate_per_s[flow, one] == 1
    assert compiled.service_arrival_rate_per_s[flow, four] == 0.25
    assert compiled.batch_fill_time_s[flow, one] == 0
    assert compiled.batch_fill_time_s[flow, four] == 1.5
    assert (
        compiled.configuration_index[flow, one]
        != compiled.configuration_index[flow, four]
    )
    assert len(compiled.layer_index) == 3
    assert len(compiled.tensor_index) == 2
    assert compiled.compute_bin_limits[0] == 0
    assert compiled.compute_bin_limits[-1] == 0.8
    assert compiled.network_bin_limits[-1] == 0.7
    assert compiled.successors_by_worker[compiled.workers[0]] == ()
    # max(lambda_batch * total_time**2) = 0.01, then divide by 2*(1-.8)**2.
    assert compiled.max_queue_wait_s_by_worker[compiled.workers[0]] == pytest.approx(
        0.125
    )


def test_discretization_changes_without_changing_problem_or_coefficients():
    problem = _problem()
    coarse, fine = _build(problem, 0.2), _build(problem, 0.05)
    assert len(fine.compute_bin_limits) > len(coarse.compute_bin_limits)
    assert fine.service_arrival_rate_per_s == coarse.service_arrival_rate_per_s
    assert fine.configuration_index == coarse.configuration_index
    assert fine.problem is coarse.problem


def test_compilation_is_deterministic_and_does_not_fill_missing_execution_costs():
    problem = _problem()
    flow = next(iter(problem.flows))
    key = next(iter(problem.configurations))
    worker = next(iter(problem.workers))
    problem = replace(
        problem, execution_times_s={(key, worker): {"input": 0.0, "output": 0.0}}
    )
    reversed_problem = replace(
        problem,
        configurations=dict(reversed(tuple(problem.configurations.items()))),
        flows={
            flow: replace(
                problem.flows[flow],
                configurations=tuple(reversed(problem.flows[flow].configurations)),
            )
        },
    )
    a, b = _build(problem), _build(reversed_problem)
    assert a.configuration_index == b.configuration_index
    assert a.configurations_by_flow == b.configurations_by_flow
    assert "compute" not in a.problem.execution_times_s[key, worker]
    assert a.max_queue_wait_s_by_worker[worker] == 0


def test_nested_maps_are_copied_and_immutable():
    compiled = _build(_problem())
    with pytest.raises(TypeError):
        compiled.flow_index[compiled.flows[0]] = 100
    variant = next(iter(compiled.tensor_routes_by_variant))
    with pytest.raises(TypeError):
        compiled.tensor_routes_by_variant[variant]["x"] = (
            compiled.tensor_routes_by_variant[variant]["y"]
        )


def test_routes_remove_reachable_consumers_but_keep_incomparable_branches():
    problem = _problem()
    variant = next(iter(problem.variants))
    graph = replace(
        problem.variants[variant],
        layers=("input", "a", "b", "c", "output"),
        edges=(
            EdgeDescription("input", "a", frozenset({"x"})),
            EdgeDescription("input", "b", frozenset({"x"})),
            EdgeDescription("input", "c", frozenset({"x"})),
            EdgeDescription("a", "b"),
            EdgeDescription("b", "output", frozenset({"y"})),
            EdgeDescription("c", "output"),
        ),
    )
    compiled = _build(replace(problem, variants={variant: graph}, execution_times_s={}))
    route = compiled.tensor_routes_by_variant[variant]["x"]
    assert route.destinations == ("a", "b", "c")
    assert route.maximal_destinations == ("b", "c")


@pytest.mark.parametrize("rate", [0.0, -1.0, math.inf, math.nan])
def test_rejects_invalid_arrival_rates(rate):
    problem = _problem()
    flow = next(iter(problem.flows))
    with pytest.raises(MisocpCompactFormulationBuildError, match="Arrival rate"):
        _build(
            replace(
                problem,
                flows={flow: replace(problem.flows[flow], arrival_rate_per_s=rate)},
            )
        )


@pytest.mark.parametrize("cost", [-1.0, math.inf, math.nan])
def test_rejects_invalid_execution_costs(cost):
    problem = _problem()
    pair = next(iter(problem.execution_times_s))
    with pytest.raises(MisocpCompactFormulationBuildError, match="Execution times"):
        _build(replace(problem, execution_times_s={pair: {"compute": cost}}))


def test_rejects_duplicate_configurations_and_invalid_graph():
    problem = _problem()
    flow = next(iter(problem.flows))
    with pytest.raises(
        MisocpCompactFormulationBuildError, match="duplicate configurations"
    ):
        _build(
            replace(
                problem,
                flows={
                    flow: replace(
                        problem.flows[flow],
                        configurations=problem.flows[flow].configurations * 2,
                    )
                },
            )
        )
    variant = next(iter(problem.variants))
    with pytest.raises(MisocpCompactFormulationBuildError, match="topological"):
        _build(
            replace(
                problem,
                variants={
                    variant: replace(
                        problem.variants[variant],
                        layers=tuple(reversed(problem.variants[variant].layers)),
                    )
                },
            )
        )
