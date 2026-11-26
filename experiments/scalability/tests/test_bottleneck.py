"""Bounds-only ablations, original-model feasibility and censored reporting."""

import itertools
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from gurobipy import GRB
from pydantic import ValidationError
from scalability_benchmark import instances
from scalability_benchmark.bottleneck import runner
from scalability_benchmark.bottleneck.analysis import (
    aggregate,
    analyze,
    thresholds,
    transitions,
)
from scalability_benchmark.bottleneck.configuration import (
    INFRASTRUCTURES,
    Study,
    cases,
    load_study,
)
from scalability_benchmark.bottleneck.decisions import apply, groups, signature
from scalability_benchmark.bottleneck.instances import instance, jsonable
from scalability_benchmark.bottleneck.metrics import dimensions, parse_log, proof
from scalability_benchmark.bottleneck.model_selection import predefined_models
from scalability_benchmark.bottleneck.reference import obtain
from scalability_benchmark.bottleneck.solver import compiler, solve
from scalability_benchmark.catalog import (
    CatalogEntry,
    selection_order,
    synthetic_accuracies,
)
from scalability_benchmark.persistence import read_journal
from scalability_benchmark.randomness import fingerprint
from test.optimizer.test_cumulative_formulation import dag_problem

from optimizer.application.formulation.compact.misocp.misocp_compact_formulation_builder import (
    MisocpCompactFormulationBuilder,
)
from optimizer.domain.formulation.compact.misocp.compact_formulation_kind import (
    CompactFormulationKind as Kind,
)
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation_config import (
    MisocpCompactFormulationConfig,
)
from optimizer.domain.problem.compact.compact_optimization_problem import ObjectiveKind
from shared.model.model import ModelId
from shared.model.model_variant import ModelVariantId

CONFIG = Path(__file__).resolve().parents[1] / "bottleneck/config/default.yaml"


@pytest.fixture
def study(tmp_path):
    return Study(
        flow_counts=(1, 2),
        seeds=(42,),
        model_count=2,
        max_components=3,
        include_primary_fixed=True,
        output=tmp_path,
        solver={
            "time_limit_seconds": 3,
            "mip_gap": 0.01,
            "threads": 1,
            "seed": 0,
            "measure_presolve": True,
        },
        reference={"validation_seconds": 5},
    )


@pytest_asyncio.fixture
async def generated(study, make_model, monkeypatch, tmp_path):
    _, base = load_study(CONFIG)
    settings = {s.name: s for s in base.catalog.models}
    accuracy = synthetic_accuracies(base, 42)
    entries = tuple(
        CatalogEntry(
            settings[name],
            tmp_path / settings[name].filename,
            ModelVariantId(ModelId(name), "fp32-batch1"),
            accuracy[name],
            {"mock": name},
        )
        for name in selection_order(base, 42)
    )

    async def profile(entry, cache, software_hash):
        return make_model(entry)

    monkeypatch.setattr(instances, "discover_catalog", lambda config, seed: entries)
    monkeypatch.setattr(instances, "profile_entry", profile)
    return await instances.build_scenario(base, 42, tmp_path, "test")


def test_plan_defaults_catalog_and_ofat():
    study, base = load_study(CONFIG)
    assert study.flow_counts == (5, 10, 20, 30, 50, 75, 100)
    assert study.seeds == (42, 43)
    assert base.infrastructure == INFRASTRUCTURES
    assert len(cases(study)) == 14
    extended = study.model_copy(
        update={
            "candidate_model_counts": (1, 2, 5),
            "component_counts": (3, 7),
            "worker_counts": (1, 11),
            "priority_counts": (1, 3),
        }
    )
    for case in cases(extended):
        if case.sweep != "flows":
            assert case.spec.flow_count == study.anchor_flow_count
        if case.sweep == "models":
            assert (
                case.max_components == study.max_components
                and case.priority_count == study.priority_count
            )
            assert case.spec.worker_count == 2
        if case.sweep == "components":
            assert (
                case.spec.model_count == study.model_count
                and case.spec.worker_count == 2
            )


@pytest.mark.parametrize(
    "changes",
    [
        {"flow_counts": (2, 1)},
        {"seeds": (42, 42)},
        {"loads": (1.0,)},
        {"worker_counts": (12,)},
        {"infrastructure_indices": (11,)},
        {"deadline_s": (2, 1)},
    ],
)
def test_invalid_settings(changes):
    with pytest.raises(ValidationError):
        Study(**changes)


def test_constant_nominal_load_nested_inputs_and_candidates(generated, study):
    first, second = (instance(generated, case) for case in cases(study))
    assert set(first.flows) < set(second.flows)
    assert (
        first.variants == second.variants
        and first.execution_times_s == second.execution_times_s
    )
    assert first.connections == second.connections and first.workers == second.workers
    for problem in (first, second):
        assert generated.reference_time_s * sum(
            f.arrival_rate_per_s for f in problem.flows.values()
        ) == pytest.approx(0.8)
        assert all(len(f.configurations) == 2 for f in problem.flows.values())
    for key, value in first.flows.items():
        assert value.max_response_time_s == second.flows[key].max_response_time_s
        assert value.configurations == second.flows[key].configurations
    largest = replace(
        cases(study)[0], spec=replace(cases(study)[0].spec, edge_count=10)
    )
    assert len(instance(generated, largest).workers) == 11


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [Kind.ASSIGNMENT, Kind.CUMULATIVE])
async def test_exact_fixing_and_original_model_feasibility(study, tmp_path, kind):
    study = study.model_copy(update={"formulation": kind})
    problem = dag_problem(priorities=2)
    before = fingerprint(jsonable(problem))
    ref, report = await obtain(
        problem,
        study,
        0.5,
        tmp_path / "refs",
        "tiny",
        model_choices=predefined_models(problem, 42),
    )
    assert ref is not None, report
    assert report["validation"]["native_feasibility"]["validated"]
    assert (
        report["acquisition"]["requested_solver_parameters"]["solution_limit"] is None
    )
    assert report["validation"]["requested_solver_parameters"]["solution_limit"] is None
    data = MisocpCompactFormulationBuilder().build_formulation(
        problem, MisocpCompactFormulationConfig(0.5, kind)
    )
    fixed_sets = []
    for ablation in study.ablations:
        async with compiler().compile_formulation(data) as compiled:
            native = compiled.native_model
            original_signature = signature(compiled)
            original_bounds = {v.VarName: (v.LB, v.UB) for v in native.getVars()}
            intended = groups(compiled)
            names = set(apply(compiled, ablation, ref, warm=True))
            expected = set()
            for label, begin in (
                ("model", "B"),
                ("partition", "C"),
                ("placement", "D"),
                ("priority", "P"),
            ):
                if study.ablations.index(ablation) >= study.ablations.index(begin):
                    expected |= intended[label]
            if ablation == "E":
                expected = intended["all_discrete_decisions"]
            assert names == expected
            for variable in native.getVars():
                if variable.VarName in names:
                    assert variable.LB == variable.UB == ref["values"][variable.VarName]
                    assert variable.VarName in intended["all_discrete_decisions"]
                else:
                    assert original_bounds[variable.VarName] == (
                        variable.LB,
                        variable.UB,
                    )
                assert variable.Start == ref["values"][variable.VarName]
            if ablation == "D":
                assert names.isdisjoint(
                    intended["priority"] | intended["auxiliary_integer"]
                )
            if ablation == "E":
                assert dimensions(native)["free_integer_variables"] == 0
            # Restore bounds: matrix, objective, constraints and original domains identical.
            for variable in native.getVars():
                variable.LB, variable.UB = original_bounds[variable.VarName]
            native.NumStart = 0
            assert signature(compiled) == original_signature
            fixed_sets.append(names)
        result, _ = await solve(
            problem,
            study,
            0.5,
            tmp_path / f"{ablation}.log",
            ablation=ablation,
            reference=ref,
            warm=False,
        )
        assert (
            result["feasible_incumbent"] and result["native_feasibility"]["validated"]
        ), result
        assert result["model_signature"] == ref["model_signature"]
        assert result["requested_solver_parameters"] == {
            "time_limit_seconds": 3,
            "mip_gap": 0.01,
            "threads": 1,
            "seed": 0,
            "solution_limit": None,
        }
    assert all(a <= b for a, b in itertools.pairwise(fixed_sets))
    assert fingerprint(jsonable(problem)) == before
    cached, report = await obtain(
        problem,
        study,
        0.5,
        tmp_path / "refs",
        "tiny",
        model_choices=predefined_models(problem, 42),
    )
    assert (
        cached is not None and report["cache_hit"] and report["acquisition"] is not None
    )
    assert report["validation"]["native_feasibility"]["validated"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "objective",
    [
        ObjectiveKind.ARRIVAL_WEIGHTED_NORMALIZED_DELAY,
        ObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_TIME,
        ObjectiveKind.ARRIVAL_WEIGHTED_NORMALIZED_RESPONSE_TIME,
    ],
)
async def test_objective_semantics_and_zero_tardiness(study, tmp_path, objective):
    problem = dag_problem(priorities=2)
    problem = replace(problem, policy=replace(problem.policy, objective=objective))
    result, _ = await solve(problem, study, 0.5, tmp_path / "objective.log")
    assert result["feasible_incumbent"], result
    assert result["effective_solver_configuration"]["objective"] == objective.value
    if objective == ObjectiveKind.ARRIVAL_WEIGHTED_NORMALIZED_DELAY:
        assert (
            result["zero_tardiness_found"] and result["global_optimality_established"]
        )
    else:
        assert (
            result["time_to_zero_tardiness_s"] is None
            and not result["zero_tardiness_found"]
        )


def test_log_parser_missing_restarted_and_nonobjective_roots():
    assert parse_log("")["root_relaxation_time_s"] is None
    result = parse_log(
        "Variable types: 100 continuous, 20 integer (20 binary)\nPresolve time: 0.12s\nPresolved: 21 rows, 30 columns, 42 nonzeros\nVariable types: 25 continuous, 5 integer (4 binary)\nRoot relaxation: objective -1.20e-03, 10 iterations, 0.03 seconds (0.01 work units)\nRoot relaxation: objective 1, 0 iterations, 0.05 seconds\n"
    )
    assert (
        result["root_relaxation_time_s"] == 0.03
        and result["root_objective_bound"] == -0.0012
    )
    assert result["log_presolved_dimensions"]["binary_variables"] == 4
    assert (
        parse_log("Root relaxation: infeasible, 2 iterations, 0.01 seconds\n")[
            "root_objective_bound"
        ]
        is None
    )
    assert (
        parse_log("Root relaxation: cutoff, 0 iterations, 0.02 seconds\n")[
            "root_relaxation_status"
        ]
        == "cutoff"
    )


@pytest.mark.parametrize(
    "status, solved, censored",
    [
        (GRB.OPTIMAL, True, False),
        (GRB.TIME_LIMIT, False, True),
        (GRB.INFEASIBLE, False, False),
    ],
)
def test_proof_status(status, solved, censored):
    result = proof(
        {
            "solver_status": status,
            "objective_value": 0.2,
            "objective_bound": 0.199,
            "final_gap": 0.005,
        },
        "arrival_weighted_response_time",
    )
    assert (
        result["solved_to_tolerance"] == solved
        and result["runtime_censored"] == censored
    )
    assert not result["global_optimality_established"]


def row(ablation="A", flows=5, seed=42, **changes):
    return {
        "manifest_id": "manifest",
        "scenario_id": f"problem-{flows}-{seed}",
        "reference_id": "ref",
        "seed": seed,
        "objective": "arrival_weighted_response_time",
        "formulation": "cumulative",
        "start_mode": "cold",
        "sweep": "flows",
        "sweep_value": flows,
        "flow_count": flows,
        "model_count": 2,
        "worker_count": 2,
        "max_components": 3,
        "priority_count": 2,
        "load": 0.8,
        "ablation": ablation,
        "available": True,
        "termination": "optimal_closed_gap",
        "solved_to_tolerance": True,
        "runtime_censored": False,
        "timeout_reached": False,
        "feasible_incumbent": True,
        "objective_value": 1.0,
        "final_gap": 0.0,
        "solver_runtime_s": 1.0,
        "state": "finished",
        **changes,
    }


def test_censored_analysis_missing_metrics_thresholds_and_plots(tmp_path):
    records = [
        row(),
        row(
            seed=43,
            termination="timeout_with_incumbent",
            timeout_reached=True,
            runtime_censored=True,
            solved_to_tolerance=False,
            solver_runtime_s=300.0,
            final_gap=0.5,
        ),
        row("B"),
        row("B", seed=43, solver_runtime_s=2.0),
    ]
    groups = aggregate(records)
    a = next(r for r in groups if r["ablation"] == "A")
    assert a["solved_runtime_median_s"] == 1 and a["observed_runtime_median_s"] == 150.5
    assert a["success_rate"] == 0.5 and a["n_censored"] == 1
    pairs = transitions(records)
    assert sum(r["both_solved"] for r in pairs) == 1
    assert (
        next(r for r in pairs if r["seed"] == 43)["runtime_ratio_to_tolerance"] is None
    )
    assert thresholds(groups)[0]["earliest_tested_F"] == 5
    with (tmp_path / "runs.jsonl").open("w") as handle:
        for index, record in enumerate(records):
            handle.write(
                json.dumps({**record, "run_id": str(index), "schema_version": 1}) + "\n"
            )
    analyze(tmp_path)
    assert len(list((tmp_path / "analysis/plots").glob("*.png"))) == 8
    assert "right-censored" in (tmp_path / "analysis/summary.md").read_text()


@pytest.mark.asyncio
async def test_reference_unavailable_preserves_cold_baseline(
    study, generated, monkeypatch, tmp_path
):
    monkeypatch.setattr(runner, "build_scenario", AsyncMock(return_value=generated))
    monkeypatch.setattr(runner, "software_provenance", lambda: {"source_hash": "mock"})
    monkeypatch.setattr(
        runner,
        "obtain",
        AsyncMock(
            return_value=(
                None,
                {
                    "reference_id": None,
                    "wall_time_s": 1,
                    "acquisition": {
                        "termination": "timeout_without_incumbent",
                        "solver_runtime_s": 3,
                        "timeout_reached": True,
                        "feasible_incumbent": False,
                    },
                },
            )
        ),
    )
    solve_mock = AsyncMock(
        return_value=(
            {
                "termination": "timeout_without_incumbent",
                "timeout_reached": True,
                "feasible_incumbent": False,
            },
            None,
        )
    )
    monkeypatch.setattr(runner, "solve", solve_mock)
    await runner.run(study, generated.config, max_instances=1)
    records = list(read_journal(tmp_path / "runs.jsonl").values())
    assert solve_mock.await_count == 1
    assert len(records) == 12
    assert sum(r["termination"] == "reference_unavailable" for r in records) == 10
    baseline = next(
        r for r in records if r["ablation"] == "A" and r["start_mode"] == "cold"
    )
    assert baseline["timeout_reached"]
    assert all(
        r["problem_fingerprint"] == baseline["problem_fingerprint"] for r in records
    )
    assert all(
        key in baseline
        for key in (
            "objective",
            "formulation",
            "relative_error",
            "source_hash",
            "reference_report",
            "deadline_policy",
        )
    )


@pytest.mark.asyncio
async def test_reject_tampered_reference_without_fixing_physical_continuous(
    study, tmp_path
):
    problem = dag_problem(priorities=2)
    reference, _ = await obtain(
        problem,
        study,
        0.5,
        tmp_path / "refs",
        "tamper",
        model_choices=predefined_models(problem, 42),
    )
    assert reference is not None
    stored = tmp_path / "refs/tamper.json"
    record = json.loads(stored.read_text())
    record["candidate"]["model_signature"] = "wrong-original-model"
    stored.write_text(json.dumps(record))
    rejected, report = await obtain(
        problem,
        study,
        0.5,
        tmp_path / "refs",
        "tamper",
        model_choices=predefined_models(problem, 42),
    )
    assert rejected is None and not report["available"]
    assert "signature mismatch" in report["validation"]["error_details"]
    baseline, _ = await solve(problem, study, 0.5, tmp_path / "baseline.log")
    assert baseline["feasible_incumbent"]


@pytest.mark.asyncio
async def test_native_infeasible_and_time_limited_results(study, tmp_path):
    problem = dag_problem(priorities=2)
    impossible = replace(
        problem,
        workers={
            worker: replace(data, available_inference_memory_bytes=1)
            for worker, data in problem.workers.items()
        },
    )
    result, _ = await solve(impossible, study, 0.5, tmp_path / "infeasible.log")
    assert result["infeasibility_proven"] and not result["timeout_reached"]
    assert not result["feasible_incumbent"] and result["objective_value"] is None
    assert result["solver_status"] == GRB.INFEASIBLE
    result, _ = await solve(
        problem, study, 0.5, tmp_path / "timeout.log", limit=1e-9, measure=False
    )
    assert result["solver_status"] == GRB.TIME_LIMIT and result["runtime_censored"]
    assert not result["solved_to_tolerance"] and not result["infeasibility_proven"]


@pytest.mark.asyncio
async def test_naturally_positive_tardiness_retains_soft_deadlines(study, tmp_path):
    problem = dag_problem(priorities=2)
    # Predeclared test workload, independent of solver results: compute service > D.
    problem = replace(
        problem,
        flows={
            key: replace(flow, max_response_time_s=0.0001)
            for key, flow in problem.flows.items()
        },
        policy=replace(
            problem.policy, objective=ObjectiveKind.ARRIVAL_WEIGHTED_NORMALIZED_DELAY
        ),
    )
    result, _ = await solve(problem, study, 0.5, tmp_path / "positive-tardiness.log")
    assert result["feasible_incumbent"] and result["objective_value"] > 0
    assert (
        not result["zero_tardiness_found"]
        and result["time_to_zero_tardiness_s"] is None
    )


@pytest.mark.asyncio
async def test_cancelled_presolve_joins_native_thread_before_disposal():
    import asyncio
    from threading import Event
    from types import SimpleNamespace

    from scalability_benchmark.bottleneck.configuration import SolverSettings
    from scalability_benchmark.bottleneck.solver import presolve_dimensions

    entered, release, finished, terminated = (Event() for _ in range(4))
    disposed = []

    class Presolved:
        def dispose(self):
            assert finished.is_set()
            disposed.append("presolved")

    class Probe:
        Params = SimpleNamespace()

        def presolve(self):
            entered.set()
            release.wait(timeout=5)
            finished.set()
            return Presolved()

        def terminate(self):
            terminated.set()

        def dispose(self):
            assert finished.is_set()
            disposed.append("probe")

    native = SimpleNamespace(copy=lambda: Probe())
    task = asyncio.create_task(presolve_dimensions(native, SolverSettings()))
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        task.cancel()
        assert await asyncio.to_thread(terminated.wait, 1)
        assert not task.done() and not disposed
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert disposed == ["presolved", "probe"]
