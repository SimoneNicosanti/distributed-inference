from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import gurobipy as gp
import pytest
from gurobipy import GRB
from optimizer.application.formulation.compact.compact_formulation_data import (
    CompactFormulationData,
)
from optimizer.application.formulation.compact.compact_formulation_data_builder import (
    CompactFormulationDataBuilder,
    _is_metric_satisfied,
)
from scalability_benchmark import solver as benchmark_solver
from scalability_benchmark.campaigns import RunSpec
from scalability_benchmark.configuration import BENCHMARK_OBJECTIVES, ExperimentConfig
from scalability_benchmark.instances import Scenario
from scalability_benchmark.solver import (
    CompactGurobiModel,
    CompactSolverOutput,
    classify,
)

from optimizer.adapters.outbound.gurobi.entities.gurobi_compiled_formulation import (
    GurobiCompiledFormulation,
)
from optimizer.adapters.outbound.gurobi.entities.gurobi_run_statistics import (
    GurobiRunStatistics,
)
from optimizer.adapters.outbound.gurobi.entities.gurobi_solver_config import (
    GurobiSolverConfig,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.compact_variables import (
    CompactVariables,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.gurobi_formulation_compiler import (
    GurobiModelBuilder,
)
from optimizer.adapters.outbound.gurobi.result_translation.formulations.compact.gurobi_misocp_compact_solved_compiled_formulation_translator import (
    GurobiCompactSolvedFormulationModelTranslator,
)
from optimizer.adapters.outbound.gurobi.solver.gurobi_solver import (
    GurobiSolver,
)
from optimizer.application.abc.solver_strategy import SolveError
from optimizer.application.ports.outbound.solver_result import SolverOutput
from optimizer.domain.problem.compact.compact_optimization_problem import (
    ObjectiveKind,
)
from shared.model.model_variant import PerformanceMetric, PerformanceMetricType


def solver_output(
    status: int, solutions: int, gap: float | None = None
) -> CompactSolverOutput:
    return SolverOutput(
        termination=GurobiSolver._termination_kind(status),
        solved_model=cast(CompactGurobiModel, object()) if solutions else None,
        runtime_s=300 if status == GRB.TIME_LIMIT else 1,
        objective_value=2.0 if solutions else None,
        objective_bound=2.0 if gap == 0 else 2.02,
        statistics=GurobiRunStatistics(
            native_status=status,
            solution_count=solutions,
            relative_gap=gap,
            construction_time_s=0.1,
            time_limit_s=300,
            mip_gap_target=0.01,
            threads=0,
            seed=0,
        ),
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
        (18, 1, 0.0, "other_termination"),  # Gurobi local-only optimality.
        (19, 0, None, "other_termination"),  # Gurobi local-only infeasibility.
    ],
)
def test_status_mapping(
    status: int, solutions: int, gap: float | None, termination: str
) -> None:
    output = solver_output(status, solutions, gap)
    record = classify(
        output,
        SolveError() if not solutions else None,
        objective=ObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_TIME,
    )
    assert record["termination"] == termination
    assert record["infeasibility_proven"] == (status == GRB.INFEASIBLE)
    assert record["solver_status"] == status
    if not solutions:
        assert record["objective_value"] is None and record["final_gap"] is None


def test_errors_have_no_invented_metrics() -> None:
    record = classify(
        None,
        RuntimeError("license unavailable"),
        objective=ObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_TIME,
    )
    assert record["termination"] == "solver_error"
    assert record["objective_value"] is None
    assert record["solver_runtime_s"] is None


def test_translation_failure_retains_native_evidence() -> None:
    record = classify(
        solver_output(GRB.TIME_LIMIT, 1, 0.1),
        ValueError("invalid extracted solution"),
        objective=ObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_TIME,
    )
    assert record["termination"] == "solver_error"
    assert record["timeout_reached"] is True
    assert record["feasible_incumbent"] is True
    assert record["solver_runtime_s"] == 300
    assert record["objective_value"] == 2.0


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


def test_output_guards_unavailable_attributes_and_captures_effective_parameters() -> (
    None
):
    model = Mock(spec=gp.Model)
    values = {"Runtime": 300.1, "ObjBound": -GRB.INFINITY, "NodeCount": 0.0}
    model.getAttr.side_effect = values.__getitem__
    model.Status = GRB.TIME_LIMIT
    model.SolCount = 0
    model.Params = SimpleNamespace(TimeLimit=300, MIPGap=0.01, Threads=0, Seed=0)
    for attribute in (
        "NumVars",
        "NumIntVars",
        "NumBinVars",
        "NumConstrs",
        "NumQConstrs",
        "NumGenConstrs",
    ):
        setattr(model, attribute, 0)
    wrapped = GurobiCompiledFormulation(
        native_model=model,
        formulation=cast(CompactFormulationData, object()),
        variables=cast(CompactVariables, object()),
        construction_time_s=0.2,
    )
    output = GurobiSolver[CompactFormulationData, CompactVariables]()._collect_result(
        wrapped, None
    )
    assert output.objective_value is None
    assert output.objective_bound is None
    assert output.runtime_s == 300.1
    assert output.statistics.relative_gap is None
    assert output.statistics.time_limit_s == 300
    assert output.statistics.mip_gap_target == 0.01


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
        GurobiSolverConfig(**settings)


@pytest.mark.asyncio
@pytest.mark.parametrize("time_limit_s", [60, 300])
@pytest.mark.parametrize("objective", BENCHMARK_OBJECTIVES)
async def test_runner_invokes_full_strategy_with_explicit_limits(
    config: ExperimentConfig,
    monkeypatch: pytest.MonkeyPatch,
    time_limit_s: int,
    objective: ObjectiveKind,
) -> None:
    output = solver_output(GRB.OPTIMAL, 1, 0)
    model = output.solved_model
    assert model is not None
    native_solve = AsyncMock(return_value=output)
    data_builder = Mock(return_value=object())
    translator = AsyncMock(return_value=Mock())

    @asynccontextmanager
    async def build_model(
        self: object, data: CompactFormulationData
    ) -> AsyncIterator[CompactGurobiModel]:
        assert data is data_builder.return_value
        yield model

    monkeypatch.setattr(GurobiSolver, "solve_formulation_model", native_solve)
    monkeypatch.setattr(
        CompactFormulationDataBuilder,
        "build_formulation_data",
        data_builder,
    )
    monkeypatch.setattr(
        GurobiModelBuilder,
        "build_formulation_model",
        build_model,
    )
    monkeypatch.setattr(
        GurobiCompactSolvedFormulationModelTranslator,
        "translate_solved_formulation_model",
        translator,
    )
    monkeypatch.setattr(
        benchmark_solver, "make_instance", lambda scenario, spec: ({}, Mock())
    )
    result = await benchmark_solver.execute_solver(
        Mock(spec=Scenario),
        RunSpec("main", 42, 0.6, 1, 1, 0, 0, time_limit_s, objective),
        config,
    )
    native_solve.assert_awaited_once()
    assert native_solve.await_args is not None
    settings = native_solve.await_args.args[1]
    assert isinstance(settings, GurobiSolverConfig)
    assert settings.time_limit_s == time_limit_s
    assert settings.mip_gap == config.solver.relative_gap
    assert settings.seed == config.solver.seed
    assert settings.threads == config.solver.threads
    assert settings.output_flag == config.solver.output_flag
    assert settings.record_incumbent_history is True
    assert data_builder.call_args.args[2].objective is objective
    translator.assert_awaited_once_with(model)
    requested = result["requested_solver_configuration"]
    effective = result["effective_solver_configuration"]
    assert isinstance(requested, dict) and isinstance(effective, dict)
    assert requested["objective"] == effective["objective"] == objective.value
    assert requested["formulation"] == effective["formulation"] == "compact"
    assert result["solution_extracted"] is True
