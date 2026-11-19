"""Explicit variants share seeded workloads with performance selection."""

import csv
from dataclasses import replace

import pytest
from scalability_benchmark import analysis, cli
from scalability_benchmark.campaigns import MODEL_SELECTIONS, RunSpec, campaign_specs
from scalability_benchmark.configuration import BENCHMARK_OBJECTIVES
from scalability_benchmark.instances import (
    instance_flows,
    make_instance,
    request_record,
)
from scalability_benchmark.persistence import ResultStore, read_journal
from scalability_benchmark.randomness import fingerprint
from scalability_benchmark.solver import execute_solver

from optimizer.domain.formulation.compact.misocp.compact_formulation_kind import (
    CompactFormulationKind,
)
from optimizer.domain.problem.compact.compact_optimization_problem import ObjectiveKind

from . import test_optimizer_contract as fixture_module

scenario = fixture_module.scenario


def _spec(scenario, **changes):
    return replace(
        RunSpec(
            "main",
            42,
            scenario.config.main.loads[0],
            7,
            5,
            0,
            0,
            5,
            model_selection="model_variant",
        ),
        **changes,
    )


def test_campaign_expands_selection_modes_and_preserves_run_identity(config):
    baseline = list(campaign_specs(config, "main"))
    specs = list(
        campaign_specs(
            config,
            "main",
            model_selections=MODEL_SELECTIONS * 2,
            objectives=BENCHMARK_OBJECTIVES,
        )
    )
    assert len(specs) == len(baseline) * 2 * len(BENCHMARK_OBJECTIVES)
    assert {spec.model_selection for spec in specs} == set(MODEL_SELECTIONS)
    assert len({fingerprint(spec.as_record()) for spec in specs}) == len(specs)
    assert specs[0] == baseline[0]
    assert replace(specs[1], model_selection="performance") == specs[0]


@pytest.mark.parametrize("selection", [(), ("unknown",)])
def test_campaign_rejects_invalid_selection(config, selection):
    with pytest.raises(ValueError, match="Select performance or model_variant"):
        list(campaign_specs(config, "main", model_selections=selection))


def test_cli_plan_counts_both_modes(config, monkeypatch, capsys):
    # The live YAML contains pilot indices beyond its reduced worker pool.
    config = config.model_copy(
        update={
            "pilot": config.pilot.model_copy(update={"cases": (config.pilot.cases[0],)})
        }
    )
    monkeypatch.setattr(cli, "load_configuration", lambda path: config)
    monkeypatch.setattr(
        "sys.argv", ["scalability", "plan", "--model-selection", *MODEL_SELECTIONS]
    )
    cli.main()
    count = 2 * len(list(campaign_specs(config, "main")))
    assert f"main: {count} runs;" in capsys.readouterr().out


def test_pinned_variants_are_repeatable_and_only_use_available_models(scenario):
    spec = _spec(scenario)
    requests = instance_flows(scenario, spec)
    assert requests == instance_flows(scenario, spec)
    assert requests[:2] == instance_flows(scenario, replace(spec, flow_count=2))
    assert requests == instance_flows(
        scenario,
        replace(
            spec,
            edge_count=3,
            objective=ObjectiveKind.DEADLINE_VIOLATIONS,
            formulation=CompactFormulationKind.CUMULATIVE,
        ),
    )
    available = {
        model.entry.variant_id for model in scenario.models[: spec.model_count]
    }
    assert {flow.model_variant_id for flow in requests} <= available
    assert len({flow.model_variant_id for flow in requests}) > 1
    for original, pinned in zip(scenario.flows[spec.load], requests, strict=True):
        assert (
            replace(
                pinned,
                performance_requirement=original.performance_requirement,
                model_variant_id=None,
            )
            == original
        )
        assert pinned.performance_requirement is None
    problem = make_instance(scenario, spec)
    for request in requests:
        flow = problem.flows[request.flow_id]
        assert flow.performance_requirement is None
        assert len(flow.configurations) == 1
        assert flow.configurations[0].model_variant_id == request.model_variant_id
    assert set(problem.variants) == {flow.model_variant_id for flow in requests}
    record = request_record(requests[0])
    assert record["minimum_accuracy"] is None and record["dataset"] is None
    assert record["model_variant_id"]["model_id"]["model_name"] in {
        model.entry.settings.name for model in scenario.models[: spec.model_count]
    }


def test_single_available_model_is_assigned_to_every_flow(scenario):
    requests = instance_flows(scenario, _spec(scenario, model_count=1))
    assert {request.model_variant_id for request in requests} == {
        scenario.models[0].entry.variant_id
    }


def test_fixed_selection_ignores_performance_threshold(scenario):
    spec = _spec(scenario)
    flows = tuple(
        replace(
            flow,
            performance_requirement=replace(flow.performance_requirement, value=1.1),
        )
        for flow in scenario.flows[spec.load]
    )
    scenario = replace(scenario, flows={**scenario.flows, spec.load: flows})
    assert make_instance(scenario, spec).flows
    with pytest.raises(ValueError, match="accuracy requirement"):
        make_instance(scenario, replace(spec, model_selection="performance"))


@pytest.mark.asyncio
async def test_campaign_records_pinned_requests_and_resumes_each_mode(
    scenario, tmp_path, monkeypatch
):
    async def build(*args):
        return scenario

    calls = []

    async def solve(scenario, spec, config):
        calls.append(spec.model_selection)
        return {"termination": "optimal", "solver_runtime_s": 0.1}

    monkeypatch.setattr(cli, "build_scenario", build)
    monkeypatch.setattr(cli, "execute_solver", solve)
    monkeypatch.setattr(
        cli, "software_provenance", lambda: {"source_hash": "test", "versions": {}}
    )
    for resume in (False, True):
        await cli.run_campaign(
            scenario.config,
            "main",
            tmp_path,
            resume=resume,
            retry_errors=False,
            limit=2,
            model_selections=MODEL_SELECTIONS,
        )
    assert calls == list(MODEL_SELECTIONS)
    records = list(read_journal(tmp_path / "runs.jsonl").values())
    assert len(records) == 2
    for record in records:
        fixed = record["model_selection"] == "model_variant"
        assert all(
            (flow["model_variant_id"] is not None) == fixed for flow in record["flows"]
        )
        assert all(
            (flow["minimum_accuracy"] is None) == fixed for flow in record["flows"]
        )
    with (tmp_path / "summary.csv").open() as handle:
        assert {row["model_selection"] for row in csv.DictReader(handle)} == set(
            MODEL_SELECTIONS
        )


def test_analysis_separates_modes_and_understands_historical_records(
    tmp_path, monkeypatch
):
    base = {
        "campaign": "main",
        "objective": "arrival_weighted_response_time",
        "load": 0.7,
        "worker_count": 1,
        "flow_count": 1,
        "model_count": 1,
        "termination": "optimal",
        "feasible_incumbent": True,
        "timeout_reached": False,
        "infeasibility_proven": False,
        "formulation": "assignment",
    }
    records = [
        {**base, "run_id": "old", "solver_runtime_s": 1.0},
        {
            **base,
            "run_id": "fixed",
            "model_selection": "model_variant",
            "solver_runtime_s": 9.0,
        },
    ]
    rows = analysis.aggregate(records)
    assert len(rows) == 6
    assert {row["runtime_median_s"] for row in rows} == {1.0, 9.0}
    destinations = []

    def plot(records, rows, destination):
        selections = {analysis.record_model_selection(record) for record in records}
        assert len(selections) == 1
        assert {row["model_selection"] for row in rows} == selections
        destinations.append(destination.relative_to(tmp_path).as_posix())

    monkeypatch.setattr(analysis, "_plot_records", plot)
    with ResultStore(tmp_path, resume=False) as store:
        for record in records:
            store.append({**record, "state": "finished"})
    analysis.analyze(tmp_path)
    assert set(destinations) == {
        "analysis/assignment",
        "analysis/model_variant/assignment",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", list(CompactFormulationKind))
@pytest.mark.parametrize(
    "objective",
    [
        ObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_TIME,
        ObjectiveKind.ARRIVAL_WEIGHTED_ACCURACY,
    ],
)
@pytest.mark.parametrize("strategy", ["full", "incremental"])
async def test_native_solver_accepts_fixed_variant_requests(
    scenario, kind, objective, strategy
):
    config = scenario.config.model_copy(
        update={
            "optimization": scenario.config.optimization.model_copy(
                update={"max_components": 1, "priority_count": 1}
            ),
            "solver": scenario.config.solver.model_copy(update={"output_flag": False}),
        }
    )
    spec = _spec(
        scenario,
        flow_count=2,
        model_count=2,
        objective=objective,
        formulation=kind,
        strategy=strategy,
    )
    result = await execute_solver(scenario, spec, config)
    assert result["solution_extracted"], result
    assert result["model_selection"] == "model_variant"
    assert result["strategy"] == strategy
    assert result["formulation"] == kind.value
    if objective is ObjectiveKind.ARRIVAL_WEIGHTED_ACCURACY:
        assert result["objective_value"] == 0
