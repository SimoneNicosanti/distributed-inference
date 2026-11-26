"""B-derived witnesses do not depend on A or restrict its free model choices."""

import json
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest
from scalability_benchmark.bottleneck import reference as reference_module
from scalability_benchmark.bottleneck import runner
from scalability_benchmark.bottleneck.analysis import aggregate, transitions
from scalability_benchmark.bottleneck.decisions import apply, groups
from scalability_benchmark.bottleneck.instances import jsonable
from scalability_benchmark.bottleneck.model_selection import (
    REFERENCE_POLICY,
    assignment_record,
    predefined_models,
)
from scalability_benchmark.bottleneck.reference import obtain
from scalability_benchmark.bottleneck.solver import compiler, solve
from scalability_benchmark.persistence import read_journal
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

from . import test_bottleneck as fixtures

study = fixtures.study
generated = fixtures.generated


def test_predefined_models_are_nested_independent_of_objective_workers_and_formulation():
    small = dag_problem(flows=2, priorities=2)
    large = dag_problem(flows=8, priorities=2)
    choices = predefined_models(large, 42)
    assert predefined_models(small, 42) == {flow: choices[flow] for flow in small.flows}
    assert choices == predefined_models(large, 42)
    objective = replace(
        large,
        policy=replace(
            large.policy, objective=ObjectiveKind.ARRIVAL_WEIGHTED_NORMALIZED_DELAY
        ),
    )
    assert choices == predefined_models(objective, 42)
    workers = dag_problem(flows=8, workers=1, priorities=2)
    assert choices == predefined_models(workers, 42)
    assert len({c.model_variant_id for c in choices.values()}) == 2
    assert all(
        choice in large.flows[flow].configurations for flow, choice in choices.items()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [Kind.ASSIGNMENT, Kind.CUMULATIVE])
async def test_cold_b_needs_no_reference_and_warm_a_keeps_models_free(
    study, tmp_path, kind
):
    study = study.model_copy(update={"formulation": kind})
    problem = dag_problem(priorities=2)
    choices = predefined_models(problem, 42)
    ref, report = await obtain(
        problem, study, 0.5, tmp_path / "refs", "b-source", model_choices=choices
    )
    assert ref is not None, report
    b = report["acquisition"]
    assert (
        b["feasible_incumbent"]
        and report["validation"]["native_feasibility"]["validated"]
    )
    assert b["requested_solver_parameters"]["solution_limit"] is None
    assert (
        b["requested_solver_parameters"]["time_limit_seconds"]
        == study.solver.time_limit_seconds
    )
    data = MisocpCompactFormulationBuilder().build_formulation(
        problem, MisocpCompactFormulationConfig(0.5, kind)
    )
    for ablation, witness, warm in (
        ("B", None, False),
        ("A", None, False),
        ("A", ref, True),
    ):
        async with compiler().compile_formulation(data) as compiled:
            native = compiled.native_model
            bounds = {v.VarName: (v.LB, v.UB) for v in native.getVars()}
            fixed = set(
                apply(compiled, ablation, witness, model_choices=choices, warm=warm)
            )
            assert fixed == (groups(compiled)["model"] if ablation == "B" else set())
            for (
                flow,
                config,
            ), var in compiled.variables.assignment.configuration_selected.items():
                if ablation == "B":
                    assert var.LB == var.UB == int(config == choices[flow])
                else:
                    assert bounds[var.VarName] == (var.LB, var.UB)
            for var in native.getVars():
                if var.VarName not in fixed:
                    assert bounds[var.VarName] == (var.LB, var.UB)
                if warm:
                    assert var.Start == ref["values"][var.VarName]
    warm_a, _ = await solve(
        problem,
        study,
        0.5,
        tmp_path / "warm-a.log",
        reference=ref,
        model_choices=choices,
        warm=True,
    )
    assert warm_a["feasible_incumbent"] and warm_a["native_feasibility"]["validated"]
    assert warm_a["fixed_variable_names"] == []
    assert warm_a["mip_start_requested"] and warm_a["mip_start_loaded"] is True
    # Validation/reference extraction itself never invokes A, even on cache reload.
    mock = AsyncMock(wraps=reference_module.solve)
    from unittest.mock import patch

    with patch.object(reference_module, "solve", mock):
        cached, cached_report = await obtain(
            problem, study, 0.5, tmp_path / "refs", "b-source", model_choices=choices
        )
    assert cached is not None and cached_report["cache_hit"]
    assert jsonable(cached_report["acquisition"]) == jsonable(b)
    assert [call.kwargs["ablation"] for call in mock.await_args_list] == ["E"]


@pytest.mark.asyncio
async def test_b_timeout_with_incumbent_is_sufficient_for_reference(
    study, tmp_path, monkeypatch
):
    problem = dag_problem()
    candidate = {"model_signature": "test", "values": {}}
    b = {
        "termination": "timeout_with_incumbent",
        "feasible_incumbent": True,
        "solved_to_tolerance": False,
    }
    validation = {"native_feasibility": {"validated": True}}
    mocked = AsyncMock(side_effect=[(b, candidate), (validation, candidate.copy())])
    monkeypatch.setattr(reference_module, "solve", mocked)
    ref, report = await obtain(
        problem,
        study,
        0.5,
        tmp_path,
        "timeout-b",
        model_choices=predefined_models(problem, 42),
    )
    assert ref is not None and report["available"]
    assert mocked.await_args_list[0].kwargs["ablation"] == "B"
    assert not mocked.await_args_list[0].kwargs.get("warm", False)
    assert mocked.await_args_list[1].kwargs["reference"] is candidate


@pytest.mark.asyncio
@pytest.mark.parametrize("old_policy", ["legacy_full_a_reference", REFERENCE_POLICY])
async def test_cache_rejects_legacy_policy_or_different_predefined_models(
    study, tmp_path, monkeypatch, old_policy
):
    problem = dag_problem()
    choices = predefined_models(problem, 42)
    stale = {
        "identity": "same-id",
        "reference_policy": old_policy,
        "model_assignment": [],
        "b_cold": {"termination": "old"},
        "candidate": None,
    }
    (tmp_path / "same-id.json").write_text(json.dumps(stale))
    mocked = AsyncMock(
        return_value=({"termination": "timeout_without_incumbent"}, None)
    )
    monkeypatch.setattr(reference_module, "solve", mocked)
    ref, report = await obtain(
        problem, study, 0.5, tmp_path, "same-id", model_choices=choices
    )
    assert ref is None and not report["cache_hit"]
    assert mocked.await_count == 1 and mocked.await_args.kwargs["ablation"] == "B"


@pytest.mark.asyncio
async def test_runner_reuses_b_cold_once_and_supplies_same_witness_to_warm_a(
    study, generated, monkeypatch, tmp_path
):
    b = {
        "termination": "gap_target_reached",
        "solver_runtime_s": 2.5,
        "feasible_incumbent": True,
        "solved_to_tolerance": True,
        "objective_value": 1.0,
        "final_gap": 0.005,
    }
    ref = {"reference_id": "b-witness", "values": {}}
    report = {"reference_id": "b-witness", "acquisition": b, "wall_time_s": 3.0}
    monkeypatch.setattr(runner, "build_scenario", AsyncMock(return_value=generated))
    monkeypatch.setattr(runner, "software_provenance", lambda: {"source_hash": "mock"})
    source = AsyncMock(return_value=(ref, report))
    monkeypatch.setattr(runner, "obtain", source)
    mocked = AsyncMock(return_value=(dict(b), None))
    monkeypatch.setattr(runner, "solve", mocked)
    await runner.run(study, generated.config, max_instances=1)
    rows = list(read_journal(tmp_path / "runs.jsonl").values())
    assert len(rows) == 12 and mocked.await_count == 11
    assert rows[0]["ablation"] == "B" and rows[0]["start_mode"] == "cold"
    assert (
        rows[0]["solver_runtime_s"] == 2.5 and rows[0]["reused_reference_source_solve"]
    )
    assert all(r["reference_policy"] == REFERENCE_POLICY for r in rows)
    assert all(
        r["reference_id"] == "b-witness"
        and r["model_assignment"] == rows[0]["model_assignment"]
        for r in rows
    )
    choices = source.await_args.kwargs["model_choices"]
    assert assignment_record(choices) == rows[0]["model_assignment"]
    assert not any(
        c.kwargs["ablation"] == "B" and not c.kwargs["warm"]
        for c in mocked.await_args_list
    )
    warm_a = next(
        c
        for c in mocked.await_args_list
        if c.kwargs["ablation"] == "A" and c.kwargs["warm"]
    )
    assert (
        warm_a.kwargs["reference"] is ref and warm_a.kwargs["model_choices"] == choices
    )
    await runner.run(study, generated.config, max_instances=1, resume=True)
    assert mocked.await_count == 11


def test_analysis_never_pairs_different_reference_protocols():
    old = fixtures.row()
    new = fixtures.row("B", reference_policy=REFERENCE_POLICY)
    assert len(aggregate([old, new])) == 2
    assert transitions([old, new]) == []


def test_mip_start_log_evidence_does_not_misclassify_no_new_incumbent():
    from scalability_benchmark.bottleneck.metrics import parse_log

    loaded = parse_log("Loaded user MIP start with objective 6.3241e-02\n")
    assert (
        loaded["mip_start_loaded"] is True and loaded["mip_start_objective"] == 0.063241
    )
    rejected = parse_log("User MIP start violates constraint queue_value by 0.001\n")
    assert rejected["mip_start_loaded"] is False
    ambiguous = parse_log("User MIP start did not produce a new incumbent solution\n")
    assert (
        ambiguous["mip_start_status"] == "no_new_incumbent"
        and ambiguous["mip_start_loaded"] is None
    )
    assert parse_log("")["mip_start_loaded"] is None


def test_warm_only_plan_counts_extra_b_reference_solve(
    study, generated, monkeypatch, capsys
):
    import sys

    from scalability_benchmark.bottleneck import __main__ as cli

    warm = study.model_copy(update={"start_modes": ("warm",)})
    monkeypatch.setattr(cli, "load_study", lambda _: (warm, generated.config))
    monkeypatch.setattr(sys, "argv", ["bottleneck", "plan"])
    cli.main()
    plan = json.loads(capsys.readouterr().out)
    assert plan["extra_reference_b_solves"] == 2
    assert plan["measured_solves"] == 12 and not plan["reference_reuses_measured_b"]
    assert plan["reference_budget_upper_bound_s"] == 2 * (
        study.solver.time_limit_seconds + study.reference.validation_seconds
    )
