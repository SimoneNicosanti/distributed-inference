from typing import Any
from unittest.mock import Mock

import gurobipy as gp
import pytest
from gurobipy import GRB
from scalability_benchmark import solver as benchmark_solver
from scalability_benchmark.campaigns import RunSpec
from scalability_benchmark.configuration import ExperimentConfig
from scalability_benchmark.instances import Scenario
from scalability_benchmark.solver import classify

from controller.adapters.outbound.optimization.gurobi_problem_builder import (
    _is_metric_satisfied,
)
from controller.adapters.outbound.optimization.gurobi_solver import (
    GurobiSolver,
    _observe_model,
)
from controller.domain.optimization.solving.solver_config import (
    ObjectiveKind,
    SolverConfig,
    SolverRunStatistics,
)
from shared.model.model_variant import PerformanceMetric, PerformanceMetricType


def statistics(
    status: int, solutions: int, gap: float | None = None
) -> SolverRunStatistics:
    return SolverRunStatistics(
        status=status,
        runtime_s=300 if status == GRB.TIME_LIMIT else 1,
        solution_count=solutions,
        objective_value=2.0 if solutions else None,
        objective_bound=2.0 if gap == 0 else 2.02,
        relative_gap=gap,
        construction_time_s=0.1,
        time_limit_s=300,
        mip_gap_target=0.01,
        threads=0,
        seed=0,
    )


@pytest.mark.parametrize(
    "status,solutions,gap,termination",
    [
        (GRB.OPTIMAL, 1, 0.0, "optimal_closed_gap"),
        (GRB.OPTIMAL, 1, 0.005, "gap_target_reached"),
        (GRB.TIME_LIMIT, 1, 0.1, "timeout_with_incumbent"),
        (GRB.TIME_LIMIT, 0, None, "timeout_without_incumbent"),
        (GRB.INFEASIBLE, 0, None, "proven_infeasible"),
        (GRB.INF_OR_UNBD, 0, None, "other_termination"),
        (GRB.NUMERIC, 0, None, "other_termination"),
        (GRB.INTERRUPTED, 1, 0.1, "interrupted"),
    ],
)
def test_status_mapping(
    status: int, solutions: int, gap: float | None, termination: str
) -> None:
    record = classify(statistics(status, solutions, gap), None)
    assert record["termination"] == termination
    assert record["infeasibility_proven"] == (status == GRB.INFEASIBLE)
    if not solutions:
        assert record["objective_value"] is None and record["final_gap"] is None


def test_errors_have_no_invented_metrics() -> None:
    record = classify(None, RuntimeError("license unavailable"))
    assert record["termination"] == "solver_error"
    assert record["objective_value"] is None
    assert record["solver_runtime_s"] is None


def test_accuracy_direction_and_dataset_contract() -> None:
    accuracy = PerformanceMetric(PerformanceMetricType.ACCURACY, "synthetic", 0.8)
    assert _is_metric_satisfied(
        accuracy, PerformanceMetric(PerformanceMetricType.ACCURACY, "synthetic", 0)
    )
    assert not _is_metric_satisfied(
        accuracy, PerformanceMetric(PerformanceMetricType.ACCURACY, "synthetic", 0.9)
    )
    assert not _is_metric_satisfied(
        accuracy, PerformanceMetric(PerformanceMetricType.ACCURACY, "different", 0)
    )


def test_observer_guards_unavailable_attributes_and_captures_effective_parameters() -> (
    None
):
    model = Mock(spec=gp.Model)
    values: dict[str, Any] = {
        "Status": GRB.TIME_LIMIT,
        "SolCount": 0,
        "Runtime": 300.1,
        "ObjBound": -GRB.INFINITY,
    }

    def attribute(name: str) -> Any:
        if name not in values:
            raise AssertionError(f"Unexpected attribute access: {name}")
        return values[name]

    model.getAttr.side_effect = attribute
    model.Params.TimeLimit = 300
    model.Params.MIPGap = 0.01
    model.Params.Threads = 0
    model.Params.Seed = 0
    observed: list[SolverRunStatistics] = []
    _observe_model(
        model,
        SolverConfig(
            ObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_SLACK, observer=observed.append
        ),
        0.2,
    )
    assert observed[0].objective_value is None
    assert observed[0].relative_gap is None
    assert observed[0].objective_bound is None
    assert observed[0].time_limit_s == 300 and observed[0].mip_gap_target == 0.01


@pytest.mark.parametrize(
    "settings",
    [
        {"time_limit_s": float("nan")},
        {"mip_gap": float("inf")},
        {"threads": -1},
        {"seed": -1},
    ],
)
def test_solver_rejects_invalid_settings(settings: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        SolverConfig(ObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_SLACK, **settings)


@pytest.mark.asyncio
@pytest.mark.parametrize("time_limit_s", [60, 300])
@pytest.mark.parametrize("formulation", ["compact"])
async def test_runner_invokes_existing_solver_with_explicit_limits(
    config: ExperimentConfig,
    monkeypatch: pytest.MonkeyPatch,
    time_limit_s: int,
    formulation: str,
) -> None:
    from unittest.mock import AsyncMock

    configured: list[SolverConfig] = []
    config = config.model_copy(
        update={"solver": config.solver.model_copy(update={"formulation": formulation})}
    )
    gateway = AsyncMock(spec=GurobiSolver)
    gateway.solve.return_value = Mock()

    def factory(settings: SolverConfig) -> Any:
        configured.append(settings)
        assert settings.observer is not None
        settings.observer(statistics(GRB.OPTIMAL, 1, 0))
        return gateway

    monkeypatch.setattr(benchmark_solver, "GurobiSolver", factory)
    monkeypatch.setattr(
        benchmark_solver, "make_instance", lambda scenario, spec: ({}, Mock())
    )
    result = await benchmark_solver.execute_solver(
        Mock(spec=Scenario), RunSpec("main", 42, 0.6, 1, 1, 0, 0, time_limit_s), config
    )
    gateway.solve.assert_awaited_once()
    assert configured[0].time_limit_s == time_limit_s
    assert configured[0].mip_gap == 0.01
    assert configured[0].seed == config.solver.seed
    assert configured[0].threads is None
    assert configured[0].formulation == formulation
    assert result["requested_solver_configuration"]["formulation"] == formulation
    assert result["solution_extracted"] is True
