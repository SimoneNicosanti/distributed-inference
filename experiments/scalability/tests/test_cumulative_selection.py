"""Selection and reporting must not change the generated optimization instance."""

import csv
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from scalability_benchmark import analysis, cli
from scalability_benchmark.campaigns import RunSpec, campaign_specs
from scalability_benchmark.configuration import BENCHMARK_OBJECTIVES, SolverSettings
from scalability_benchmark.instances import make_instance
from scalability_benchmark.persistence import ResultStore, read_journal
from scalability_benchmark.randomness import fingerprint
from scalability_benchmark.solver import execute_solver

from optimizer.domain.formulation.compact.misocp.compact_formulation_kind import (
    CompactFormulationKind as Kind,
)

from . import test_optimizer_contract as fixture_module

scenario = fixture_module.scenario
COMPARISON_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts/compare_formulations.py"
)


@pytest.mark.parametrize("kind", list(Kind))
def test_yaml_and_default_selection(config, kind):
    settings = SolverSettings.model_validate(
        {**config.solver.model_dump(), "formulation": kind.value}
    )
    assert settings.formulation is kind
    assert (
        SolverSettings.model_validate(
            config.solver.model_dump(exclude={"formulation"})
        ).formulation
        is Kind.ASSIGNMENT
    )
    assert (
        SolverSettings.model_validate(
            {**config.solver.model_dump(), "formulation": "compact"}
        ).formulation
        is Kind.ASSIGNMENT
    )
    with pytest.raises(ValueError):
        SolverSettings.model_validate(
            {**config.solver.model_dump(), "formulation": "unknown"}
        )


def test_identical_problems_and_distinct_run_ids(scenario):
    settings = scenario.config.solver.model_copy(
        update={"formulation": Kind.CUMULATIVE}
    )
    alternative = scenario.config.model_copy(update={"solver": settings})
    left = list(campaign_specs(scenario.config, "main"))
    right = list(campaign_specs(alternative, "main"))
    assert len(left) == len(right)
    assert {item.edge_count for item in right} == set(range(11))
    assert all(
        replace(a, formulation=b.formulation) == b
        for a, b in zip(left, right, strict=True)
    )
    for first, second in zip(left[:3], right[:3], strict=True):
        assert make_instance(scenario, first) == make_instance(scenario, second)
        assert fingerprint(first.as_record()) != fingerprint(second.as_record())


def test_cli_accepts_cumulative_and_rejects_unknown(monkeypatch, capsys):
    monkeypatch.setattr(
        sys, "argv", ["scalability", "plan", "--formulation", "cumulative"]
    )
    cli.main()
    assert "main:" in capsys.readouterr().out
    monkeypatch.setattr(
        sys, "argv", ["scalability", "plan", "--formulation", "unknown"]
    )
    with pytest.raises(SystemExit):
        cli.main()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", list(Kind))
@pytest.mark.parametrize("objective", BENCHMARK_OBJECTIVES)
async def test_native_benchmark_records_selected_formulation(
    scenario, tmp_path, kind, objective
):
    config = scenario.config.model_copy(
        update={
            "optimization": scenario.config.optimization.model_copy(
                update={"max_components": 1, "priority_count": 1}
            ),
            "solver": scenario.config.solver.model_copy(update={"formulation": kind}),
        }
    )
    spec = RunSpec("main", 42, config.main.loads[0], 1, 1, 0, 0, 5, objective, kind)
    result = await execute_solver(scenario, spec, config)
    assert result["solution_extracted"], result
    assert result["solver_status_name"] == "OPTIMAL"
    assert result["formulation"] == kind.value
    assert result["effective_solver_configuration"]["formulation"] == kind.value
    assert result["requested_solver_configuration"]["formulation"] == kind.value
    assert result["original_variables"] > 0
    assert result["original_linear_nonzeros"] > 0
    assert result["original_constraints"] > 0
    assert result["raw_solver_statistics"]["bound_history"]
    with ResultStore(tmp_path, resume=False) as store:
        store.append(
            {"run_id": "test", "state": "finished", **spec.as_record(), **result}
        )
    with (tmp_path / "summary.csv").open() as handle:
        row = next(csv.DictReader(handle))
    assert row["formulation"] == kind.value
    assert int(row["original_variables"]) == result["original_variables"]


def test_analysis_never_pools_representations(tmp_path, monkeypatch):
    base = {
        "state": "finished",
        "campaign": "main",
        "load": 0.7,
        "objective": "arrival_weighted_response_time",
        "worker_count": 1,
        "flow_count": 1,
        "model_count": 1,
        "termination": "optimal_closed_gap",
        "feasible_incumbent": True,
        "timeout_reached": False,
        "infeasibility_proven": False,
        "final_gap": 0.0,
    }
    records = [
        {
            **base,
            "run_id": kind.value,
            "formulation": kind.value,
            "solver_runtime_s": float(index + 1),
        }
        for index, kind in enumerate(Kind)
    ]
    rows = analysis.aggregate(records)
    assert len(rows) == 6
    assert all(row["completed_count"] == 1 for row in rows)
    observed = []

    def plot(selected, selected_rows, destination):
        observed.append(destination.name)
        assert {row["formulation"] for row in selected_rows} == {destination.name}
        assert {row["formulation"] for row in selected} == {destination.name}

    monkeypatch.setattr(analysis, "_plot_records", plot)
    with ResultStore(tmp_path, resume=False) as store:
        for record in records:
            store.append(record)
    analysis.analyze(tmp_path)
    assert set(observed) == {"assignment", "cumulative"}


@pytest.mark.asyncio
async def test_paired_comparison_reuses_one_scenario(scenario, tmp_path, monkeypatch):
    loader_spec = importlib.util.spec_from_file_location(
        "paired_comparison", COMPARISON_SCRIPT
    )
    module = importlib.util.module_from_spec(loader_spec)
    loader_spec.loader.exec_module(module)
    config = scenario.config.model_copy(
        update={
            "optimization": scenario.config.optimization.model_copy(
                update={"max_components": 1, "priority_count": 1}
            )
        }
    )
    spec = RunSpec("main", 42, config.main.loads[0], 1, 1, 0, 0, 5)
    calls = []

    async def build(config, seed, cache, software):
        calls.append(seed)
        return scenario

    monkeypatch.setattr(module, "load_configuration", lambda path: config)
    monkeypatch.setattr(module, "build_scenario", build)
    monkeypatch.setattr(module, "campaign_specs", lambda config, name: iter([spec]))
    await module.compare(tmp_path / "config.yaml", tmp_path / "paired", 1, 2)
    records = list(read_journal(tmp_path / "paired/runs.jsonl").values())
    assert calls == [42]
    assert len(records) == 4
    assert len({row["scenario_id"] for row in records}) == 1
    assert {row["formulation"] for row in records} == {"assignment", "cumulative"}
    assert all(row["solver_status_name"] == "OPTIMAL" for row in records)
