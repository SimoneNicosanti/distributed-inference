"""Strategy selection, stage budgets, persisted diagnostics and separated analysis."""

import csv
from dataclasses import replace

import pytest
from scalability_benchmark import analysis, cli
from scalability_benchmark.campaigns import RunSpec, campaign_specs
from scalability_benchmark.configuration import ExperimentConfig, SolverSettings
from scalability_benchmark.persistence import ResultStore
from scalability_benchmark.randomness import fingerprint
from scalability_benchmark.solver import build_solver_strategy, execute_solver

from optimizer.adapters.outbound.gurobi.entities.gurobi_solver_config import (
    GurobiSolverConfig,
)
from optimizer.adapters.outbound.gurobi.entities.gurobi_strategy_config import (
    GurobiStrategyConfig,
)
from optimizer.application.strategies.incremental.incremental_flow_solver_strategy import (
    IncrementalFlowSolverStrategy,
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


def test_selection_defaults_budgets_and_run_identity(config):
    assert config.solver.strategy == "full"
    raw = config.model_dump()
    raw["solver"].update(
        strategy="incremental",
        insertion_time_limit_s=2,
        repair_time_limit_s=3,
        final_time_limit_s=5,
    )
    selected = ExperimentConfig.model_validate(raw)
    old = next(campaign_specs(config, "main"))
    spec = next(campaign_specs(selected, "main"))
    assert spec.strategy == "incremental"
    assert (
        spec.insertion_time_limit_s,
        spec.repair_time_limit_s,
        spec.final_time_limit_s,
    ) == (2, 3, 5)
    assert fingerprint(spec.as_record()) != fingerprint(old.as_record())
    assert (
        spec.solver_budget_upper_bound_s
        == spec.flow_count * 2 + (spec.flow_count - 1) * 3 + 5
    )
    assert old.solver_budget_upper_bound_s == old.timeout_s
    assert fingerprint(spec.as_record()) != fingerprint(
        replace(spec, repair_time_limit_s=4).as_record()
    )
    for field, value in (
        ("strategy", "unknown"),
        ("repair_time_limit_s", 0),
        ("final_time_limit_s", float("inf")),
    ):
        with pytest.raises(ValueError):
            SolverSettings.model_validate({**config.solver.model_dump(), field: value})


def test_optimizer_factory_reuses_native_configs_and_preserves_independent_final_budget():
    runner, _ = build_solver_strategy(
        MisocpCompactFormulationConfig(0.5),
        GurobiSolverConfig(time_limit_s=10, threads=1, seed=42, mip_gap=0.01),
        strategy_config=GurobiStrategyConfig("incremental", 2, 3, 5),
    )
    assert isinstance(runner, IncrementalFlowSolverStrategy)
    config = runner._config
    assert [
        c.time_limit_s
        for c in (
            config.insertion_solver_config,
            config.repair_solver_config,
            config.final_solver_config,
        )
    ] == [2, 3, 5]
    assert [
        c.solution_limit
        for c in (
            config.insertion_solver_config,
            config.repair_solver_config,
            config.final_solver_config,
        )
    ] == [1, 1, None]
    assert all(
        c.seed == 42 and c.threads == 1 and c.mip_gap == 0.01
        for c in (
            config.insertion_solver_config,
            config.repair_solver_config,
            config.final_solver_config,
        )
    )


def test_cli_selection_and_plan_does_not_profile(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv", ["scalability", "plan", "--strategy", "incremental"]
    )
    cli.main()
    assert "solver budget upper bound=" in capsys.readouterr().out


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", list(Kind))
async def test_native_benchmark_retains_all_stage_metrics_and_csv(
    scenario, config, kind, tmp_path
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
        "incremental",
    )
    result = await execute_solver(scenario, spec, selected)
    assert result["solution_extracted"], result
    assert result["feasible_incumbent"]
    assert result["strategy"] == "incremental"
    assert result["native_solve_count"] == 3
    assert result["global_repair_count"] == 0
    assert result["successful_restricted_insertions"] == 1
    assert result["raw_solver_statistics"] is None
    assert result["last_attempt_solver_statistics"] is not None
    diagnostics = result["strategy_diagnostics"]
    assert [a["flow_count"] for a in diagnostics["attempts"]] == [1, 2, 2]
    assert [a["fixed_flow_count"] for a in diagnostics["attempts"]] == [0, 1, 0]
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
    assert row["strategy"] == "incremental"
    assert row["native_solve_count"] == "3"


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
    records = [{**base, "strategy": strategy} for strategy in ("full", "incremental")]
    rows = analysis.aggregate(records)
    assert len(rows) == 6
    assert {row["strategy"] for row in rows} == {"full", "incremental"}
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
    assert destinations == ["analysis/assignment", "analysis/incremental/assignment"]
