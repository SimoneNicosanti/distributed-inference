"""Strategy selection, stage budgets, persisted diagnostics and separated analysis."""

import csv
from dataclasses import replace

import pytest
from scalability_benchmark import analysis, cli, solver
from scalability_benchmark.campaigns import RunSpec, campaign_specs
from scalability_benchmark.configuration import ExperimentConfig, SolverSettings
from scalability_benchmark.persistence import ResultStore
from scalability_benchmark.randomness import fingerprint
from scalability_benchmark.solver import build_solver_strategy, execute_solver
from test.optimizer.test_cumulative_formulation import dag_problem

from optimizer.adapters.outbound.gurobi.entities.gurobi_solver_config import (
    GurobiSolverConfig,
)
from optimizer.adapters.outbound.gurobi.entities.gurobi_strategy_config import (
    GurobiStrategyConfig,
)
from optimizer.application.strategies.incremental.incremental_flow_solver_strategy import (
    IncrementalFlowSolverStrategy,
)
from optimizer.application.strategies.incremental.persistent_incremental_flow_solver_strategy import (
    PersistentIncrementalFlowSolverStrategy,
)
from optimizer.domain.formulation.compact.misocp.compact_formulation_kind import (
    CompactFormulationKind as Kind,
)
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation_config import (
    MisocpCompactFormulationConfig,
)
from optimizer.domain.problem.compact.compact_optimization_problem import ObjectiveKind

from . import test_optimizer_contract as fixture_module

scenario = fixture_module.scenario


@pytest.mark.parametrize("strategy", ["incremental", "incremental-persistent"])
def test_selection_defaults_budgets_and_run_identity(config, strategy):
    assert config.solver.strategy == "full"
    raw = config.model_dump()
    raw["solver"].update(
        strategy=strategy,
        insertion_time_limit_s=2,
    )
    selected = ExperimentConfig.model_validate(raw)
    old = next(campaign_specs(config, "main"))
    spec = next(campaign_specs(selected, "main"))
    assert spec.strategy == strategy
    assert spec.insertion_time_limit_s == 2
    assert fingerprint(spec.as_record()) != fingerprint(old.as_record())
    assert spec.solver_budget_upper_bound_s == spec.flow_count * 2
    assert old.solver_budget_upper_bound_s == old.timeout_s
    assert fingerprint(spec.as_record()) != fingerprint(
        replace(spec, insertion_time_limit_s=4).as_record()
    )
    for field, value in (
        ("strategy", "unknown"),
        ("insertion_time_limit_s", 0),
        ("insertion_time_limit_s", float("inf")),
    ):
        with pytest.raises(ValueError):
            SolverSettings.model_validate({**config.solver.model_dump(), field: value})


@pytest.mark.parametrize("strategy", ["incremental", "incremental-persistent"])
@pytest.mark.parametrize("solution_limit", [None, 1])
def test_optimizer_factory_optimizes_each_unrestricted_prefix_to_configured_gap(
    solution_limit,
    strategy,
):
    runner, _ = build_solver_strategy(
        MisocpCompactFormulationConfig(0.5),
        GurobiSolverConfig(
            time_limit_s=10,
            threads=1,
            seed=42,
            mip_gap=0.01,
            solution_limit=solution_limit,
        ),
        strategy_config=GurobiStrategyConfig(strategy, 2),
    )
    engine = (
        runner._strategy
        if isinstance(runner, PersistentIncrementalFlowSolverStrategy)
        else runner
    )
    assert isinstance(engine, IncrementalFlowSolverStrategy)
    config = engine._config.insertion_solver_config
    assert config.time_limit_s == 2
    assert config.solution_limit is None
    assert config.seed == 42 and config.threads == 1 and config.mip_gap == 0.01


@pytest.mark.parametrize("strategy", ["incremental", "incremental-persistent"])
def test_cli_selection_and_plan_does_not_profile(monkeypatch, capsys, strategy):
    monkeypatch.setattr("sys.argv", ["scalability", "plan", "--strategy", strategy])
    cli.main()
    assert "solver budget upper bound=" in capsys.readouterr().out


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", list(Kind))
@pytest.mark.parametrize("strategy", ["incremental", "incremental-persistent"])
async def test_native_benchmark_retains_all_stage_metrics_and_csv(
    scenario, config, kind, tmp_path, strategy
):
    # Keep the actual complete MISOCP inside the native restricted-license size.
    selected = config.model_copy(
        update={
            "optimization": config.optimization.model_copy(
                update={"max_components": 1, "priority_count": 2}
            )
        }
    )
    spec = RunSpec(
        "main",
        42,
        config.main.loads[0],
        2,
        1,
        0,
        0,
        5,
        ObjectiveKind.ARRIVAL_WEIGHTED_NORMALIZED_DELAY,
        kind,
        strategy,
    )
    result = await execute_solver(scenario, spec, selected)
    assert result["solution_extracted"], result
    assert result["feasible_incumbent"]
    assert result["strategy"] == strategy
    assert result["native_solve_count"] == 2
    assert result["successful_insertions"] == 1
    assert result["raw_solver_statistics"] is None
    assert result["last_attempt_solver_statistics"] is not None
    assert result["returned_source"] == "last_prefix"
    assert result["termination"] in {"optimal_closed_gap", "gap_target_reached"}
    assert result["final_gap"] <= config.solver.relative_gap
    assert result["objective_bound"] is not None
    diagnostics = result["strategy_diagnostics"]
    assert [a["flow_count"] for a in diagnostics["attempts"]] == [1, 2]
    assert all(a["solver_runtime_s"] >= 0 for a in diagnostics["attempts"])
    assert result["solver_runtime_s"] == sum(
        a["solver_runtime_s"] for a in diagnostics["attempts"]
    )
    with ResultStore(tmp_path, resume=False, retry_errors=False) as store:
        store.append(
            {"run_id": "incremental", "state": "finished", **spec.as_record(), **result}
        )
    with (tmp_path / "summary.csv").open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["strategy"] == strategy
    assert row["native_solve_count"] == "2"
    assert float(row["prefix_wall_time_s"]) >= float(row["construction_time_s"])
    assert "incremental_construction_time_s" not in row


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", list(Kind))
@pytest.mark.parametrize("failed_prefix", [1, 2])
@pytest.mark.parametrize("strategy", ["incremental", "incremental-persistent"])
async def test_native_prefix_failure_reports_only_full_problem_proof(
    scenario, config, monkeypatch, kind, failed_prefix, strategy
):
    problem = dag_problem(variants=1, workers=1, components=1, flows=2)
    failed_flow = tuple(problem.flows)[failed_prefix - 1]
    problem = replace(
        problem,
        flows={
            flow: replace(request, max_response_time_s=1e-12)
            if flow == failed_flow
            else request
            for flow, request in problem.flows.items()
        },
    )
    monkeypatch.setattr(solver, "make_instance", lambda *args, **kwargs: problem)
    spec = RunSpec(
        "main",
        42,
        config.main.loads[0],
        2,
        1,
        0,
        0,
        5,
        ObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_TIME,
        kind,
        "incremental",
    )
    result = await execute_solver(scenario, spec, config)
    assert not result["solution_extracted"]
    assert result["completed_flow_count"] == failed_prefix - 1
    assert result["native_solve_count"] == failed_prefix
    assert result["infeasibility_proven"] is (failed_prefix == 2)
    assert result["termination"] == (
        "proven_infeasible" if failed_prefix == 2 else "incremental_failure"
    )
    assert result["final_gap"] is None


@pytest.mark.parametrize("field", ["repair_time_limit_s", "final_time_limit_s"])
def test_removed_phase_limits_are_rejected(config, field):
    with pytest.raises(ValueError, match="Extra inputs"):
        SolverSettings.model_validate({**config.solver.model_dump(), field: 1})


def test_analysis_never_pools_strategies_and_preserves_full_output_paths(
    tmp_path, monkeypatch
):
    base = {
        "campaign": "main",
        "objective": "arrival_weighted_normalized_delay",
        "formulation": "assignment",
        "load": 0.7,
        "worker_count": 1,
        "flow_count": 2,
        "model_count": 1,
        "termination": "optimal_closed_gap",
        "feasible_incumbent": True,
        "solver_runtime_s": 1,
        "final_gap": 0,
        "timeout_reached": False,
        "infeasibility_proven": False,
    }
    records = [
        {**base, "strategy": strategy}
        for strategy in ("full", "incremental", "incremental-persistent")
    ]
    rows = analysis.aggregate(records)
    assert len(rows) == 9
    assert {row["strategy"] for row in rows} == {
        "full",
        "incremental",
        "incremental-persistent",
    }
    assert all(row["completed_count"] == 1 for row in rows)
    with ResultStore(tmp_path, resume=False, retry_errors=False) as store:
        for record in records:
            store.append({**record, "run_id": record["strategy"], "state": "finished"})
    destinations = []
    monkeypatch.setattr(
        analysis,
        "_plot_records",
        lambda records, rows, path: destinations.append(
            path.relative_to(tmp_path).as_posix()
        ),
    )
    analysis.analyze(tmp_path)
    assert destinations == [
        "analysis/assignment",
        "analysis/incremental/assignment",
        "analysis/incremental-persistent/assignment",
    ]
