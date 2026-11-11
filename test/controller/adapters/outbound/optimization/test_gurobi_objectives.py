"""Objective/feasibility regression cases; require a solver when explicitly run."""

from dataclasses import replace

import pytest
from controller.adapters.outbound.optimization.gurobi_solver import (
    GurobiSolver,
    OptimizationInfeasibleError,
)
from test.controller.adapters.outbound.optimization.test_gurobi_solver import (
    _context,
    _flow,
    _problem_config,
    _request,
    _worker,
)

from controller.adapters.outbound.optimization.gurobi.entities.gurobi_run_statistics import (
    GurobiRunStatistics,
)
from controller.adapters.outbound.optimization.gurobi.entities.gurobi_solver_config import (
    GurobiSolverConfig,
)
from controller.domain.optimization.request.local_worker_config import (
    ValidLocalWorkerConfig,
)
from controller.domain.optimization.request.optimization_request import (
    OptimizationRequestsByFlow,
)
from controller.domain.optimization.solving.formulation.compact.compact_objective_kind import (
    CompactObjectiveKind,
)
from controller.domain.optimization.solving.optimization_context import (
    OptimizationContext,
)
from shared.model.model_variant import PerformanceMetric, PerformanceMetricType


@pytest.mark.parametrize("objective", list(CompactObjectiveKind))
def test_response_time_and_accuracy_objectives_require_deadline(
    objective: CompactObjectiveKind,
) -> None:
    worker = _worker(1)
    context = _context({worker: 0.2})
    request = _request(
        worker, arrival_rate=0.1, max_response_time_s=0.001, flow_id=_flow(1)
    )
    observed: list[GurobiRunStatistics] = []
    solver = GurobiSolver(
        GurobiSolverConfig(
            objective, observer=observed.append, threads=1, seed=0, mip_gap=0
        )
    )
    if objective in {
        CompactObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_TIME,
        CompactObjectiveKind.ARRIVAL_WEIGHTED_ACCURACY,
    }:
        with pytest.raises(OptimizationInfeasibleError):
            solver._solve({request.flow_id: request}, context, _problem_config(1))
        return
    result = solver._solve({request.flow_id: request}, context, _problem_config(1))
    assert result.flow_solutions[0].mean_response_time_s > 0.001
    expected_response = 0.2 + 0.1 * 0.2**2 / (2 * (1 - 0.1 * 0.2))
    expected = {
        CompactObjectiveKind.DEADLINE_VIOLATIONS: 1,
        CompactObjectiveKind.ARRIVAL_WEIGHTED_NORMALIZED_DELAY: expected_response
        / 0.001
        - 1,
    }
    assert result.objective_value == pytest.approx(expected[objective], abs=1e-5)
    assert observed[-1].objective == objective.value
    assert observed[-1].general_constraints == (
        1 if objective == CompactObjectiveKind.DEADLINE_VIOLATIONS else 0
    )


@pytest.mark.parametrize(
    "objective",
    [
        CompactObjectiveKind.DEADLINE_VIOLATIONS,
        CompactObjectiveKind.ARRIVAL_WEIGHTED_NORMALIZED_DELAY,
    ],
)
def test_satisfied_deadline_has_zero_violation_or_delay_cost(
    objective: CompactObjectiveKind,
) -> None:
    worker = _worker(1)
    request = _request(
        worker, arrival_rate=0.1, max_response_time_s=10, flow_id=_flow(1)
    )
    result = GurobiSolver(GurobiSolverConfig(objective))._solve(
        {request.flow_id: request}, _context({worker: 0.1}), _problem_config(1)
    )
    assert result.objective_value == pytest.approx(0, abs=1e-7)


def test_accuracy_uses_request_dataset_and_flow_arrival_weights() -> None:
    worker = _worker(1)
    context = _context({worker: 0.1})
    context = OptimizationContext(
        static_profiles_by_variant={
            variant: replace(
                profile,
                info=replace(
                    profile.info,
                    accuracies=(
                        PerformanceMetric(PerformanceMetricType.MAP_AT_50, "test", 0.7),
                        PerformanceMetric(
                            PerformanceMetricType.MAP_AT_50, "other", 0.9
                        ),
                    ),
                ),
            )
            for variant, profile in context.static_profiles_by_variant.items()
        },
        network_topology=context.network_topology,
        resource_profiles_by_worker=context.resource_profiles_by_worker,
        execution_profiles_by_worker=context.execution_profiles_by_worker,
    )
    requests: OptimizationRequestsByFlow = {}
    for number, arrival, dataset in ((1, 0.1, "test"), (2, 0.3, "other")):
        request = _request(
            worker,
            arrival_rate=arrival,
            max_response_time_s=10,
            flow_id=_flow(number),
        )
        request = replace(
            request,
            qos_requirements=replace(
                request.qos_requirements,
                min_model_performance=PerformanceMetric(
                    PerformanceMetricType.MAP_AT_50, dataset, 0.6
                ),
            ),
        )
        requests[request.flow_id] = request
    result = GurobiSolver(
        GurobiSolverConfig(CompactObjectiveKind.ARRIVAL_WEIGHTED_ACCURACY)
    )._solve(requests, context, _problem_config(1))
    assert result.objective_value == pytest.approx(0.25 * 0.7 + 0.75 * 0.9)


@pytest.mark.parametrize(
    "objective",
    [
        kind
        for kind in CompactObjectiveKind
        if kind
        in {
            CompactObjectiveKind.DEADLINE_VIOLATIONS,
            CompactObjectiveKind.ARRIVAL_WEIGHTED_NORMALIZED_DELAY,
        }
    ],
)
def test_soft_objectives_allow_network_response_above_deadline(
    objective: CompactObjectiveKind,
) -> None:
    worker, edge = _worker(1), _worker(2)
    context = _context({worker: 2, edge: 0.1}, throughput_bytes_per_s=4)
    request = _request(
        worker, arrival_rate=0.01, max_response_time_s=0.01, flow_id=_flow(1)
    )
    request = replace(
        request,
        local_worker_config=ValidLocalWorkerConfig(
            worker_id=worker, is_inference_runner=False
        ),
    )
    result = GurobiSolver(GurobiSolverConfig(objective, threads=1, seed=0))._solve(
        {request.flow_id: request}, context, _problem_config(3)
    )
    flow = result.flow_solutions[0]
    assert (
        sum(component.mean_transmission_time_s for component in flow.components) > 0.01
    )
    assert flow.mean_response_time_s > 0.01
