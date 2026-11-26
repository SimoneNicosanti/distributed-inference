"""Fixed B6 requests share seeded workloads without sweeping model counts."""

import csv
from dataclasses import replace

import pytest
import pytest_asyncio
import yaml
from scalability_benchmark import analysis, cli, instances, model_variant_cli
from scalability_benchmark.campaigns import (
    FIXED_MODEL_NAME,
    RunSpec,
    campaign_specs,
)
from scalability_benchmark.catalog import CatalogEntry, synthetic_accuracies
from scalability_benchmark.configuration import (
    BENCHMARK_OBJECTIVES,
    default_configuration_path,
    load_configuration,
)
from scalability_benchmark.instances import (
    build_scenario,
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
from shared.model.model import ModelId
from shared.model.model_variant import ModelVariantId

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


@pytest.fixture
def fixed_config():
    return load_configuration(default_configuration_path("model_variant"))


@pytest_asyncio.fixture
async def fixed_scenario(fixed_config, make_model, monkeypatch, tmp_path):
    settings = fixed_config.catalog.models[0]
    entry = CatalogEntry(
        settings,
        tmp_path / settings.filename,
        ModelVariantId(ModelId(settings.name), "fp32-batch1"),
        synthetic_accuracies(fixed_config, 42)[settings.name],
        {settings.filename: "mock-hash"},
    )
    profiled = []

    async def profile(entry, cache, software_hash):
        profiled.append(entry.settings.name)
        return make_model(entry)

    monkeypatch.setattr(instances, "discover_catalog", lambda config, seed: (entry,))
    monkeypatch.setattr(instances, "profile_entry", profile)
    result = await build_scenario(fixed_config, 42, tmp_path, "test-software")
    assert profiled == [FIXED_MODEL_NAME]
    assert len(result.models) == 1
    return result


@pytest.mark.parametrize("mode", ["performance", "model_variant"])
def test_separate_campaigns_preserve_objective_expansion(config, fixed_config, mode):
    selected = config if mode == "performance" else fixed_config
    baseline = list(campaign_specs(selected, "main"))
    specs = list(campaign_specs(selected, "main", objectives=BENCHMARK_OBJECTIVES))
    assert len(specs) == len(baseline) * len(BENCHMARK_OBJECTIVES)
    assert {spec.model_selection for spec in specs} == {mode}
    assert len({fingerprint(spec.as_record()) for spec in specs}) == len(specs)
    assert specs[0] == baseline[0]
    if mode == "model_variant":
        assert {spec.model_count for spec in specs} == {1}
    else:
        assert {spec.model_count for spec in specs} == set(config.main.models)


@pytest.mark.parametrize("campaign", ["main", "sensitivity", "pilot"])
def test_fixed_campaign_has_one_candidate_without_model_fields(fixed_config, campaign):
    raw = yaml.safe_load(default_configuration_path("model_variant").read_text())
    for name in ("main", "sensitivity"):
        assert "models" not in raw[name]
    assert all("models" not in case for case in raw["pilot"]["cases"])
    specs = list(campaign_specs(fixed_config, campaign))
    assert specs and {spec.model_count for spec in specs} == {1}
    assert len(set(specs)) == len(specs)
    assert {spec.model_selection for spec in specs} == {"model_variant"}
    assert tuple(m.name for m in fixed_config.catalog.models) == (FIXED_MODEL_NAME,)


@pytest.mark.parametrize("section", ["main", "sensitivity", "pilot"])
def test_fixed_config_rejects_candidate_sweeps(tmp_path, section):
    raw = yaml.safe_load(default_configuration_path("model_variant").read_text())
    if section == "pilot":
        raw["pilot"]["cases"][0]["models"] = 2
    else:
        raw[section]["models"] = [1, 2]
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_configuration(path)


def test_fixed_config_requires_b6_catalog(tmp_path):
    raw = yaml.safe_load(default_configuration_path("model_variant").read_text())
    raw["catalog"]["models"][0]["name"] = "resnet50"
    raw["flows"]["reference_model"] = "resnet50"
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="requires only efficientnet_b6"):
        load_configuration(path)


@pytest.mark.parametrize("mode", ["performance", "model_variant"])
def test_cli_plans_only_its_own_experiment(mode, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["scalability-" + mode, "plan"])
    entrypoint = cli.main if mode == "performance" else model_variant_cli.main
    entrypoint()
    config = load_configuration(default_configuration_path(mode))
    count = len(list(campaign_specs(config, "main")))
    assert f"main: {count} runs;" in capsys.readouterr().out


@pytest.mark.parametrize("mode", ["performance", "model_variant"])
def test_cli_rejects_other_experiment_config(mode, monkeypatch):
    other = "model_variant" if mode == "performance" else "performance"
    monkeypatch.setattr(
        "sys.argv",
        [
            "scalability-" + mode,
            "plan",
            "--config",
            str(default_configuration_path(other)),
        ],
    )
    with pytest.raises(SystemExit) as error:
        (cli.main if mode == "performance" else model_variant_cli.main)()
    assert error.value.code == 2


@pytest.mark.parametrize("mode", ["performance", "model_variant"])
def test_cli_rejects_combined_mode_flag(mode, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "scalability-" + mode,
            "plan",
            "--model-selection",
            "performance",
            "model_variant",
        ],
    )
    with pytest.raises(SystemExit) as error:
        (cli.main if mode == "performance" else model_variant_cli.main)()
    assert error.value.code == 2


def test_outputs_and_configs_are_independent(config, fixed_config):
    assert config.model_selection == "performance"
    assert fixed_config.model_selection == "model_variant"
    assert config.root / config.output != fixed_config.root / fixed_config.output
    assert config.main.models != fixed_config.main.models
    assert default_configuration_path() != default_configuration_path("model_variant")


def test_every_flow_requests_b6_independently_of_available_prefix(scenario):
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
    b6 = next(
        model
        for model in scenario.models
        if model.entry.settings.name == FIXED_MODEL_NAME
    )
    assert b6 not in scenario.models[:1]
    assert {flow.model_variant_id for flow in requests} == {b6.entry.variant_id}
    for count in (1, 2, len(scenario.models)):
        selected = replace(spec, model_count=count)
        assert requests == instance_flows(scenario, selected)
        assert make_instance(scenario, spec) == make_instance(scenario, selected)
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
    assert record["model_variant_id"]["model_id"]["model_name"] == FIXED_MODEL_NAME


def test_fixed_requests_fail_if_b6_profile_is_missing(scenario):
    scenario = replace(
        scenario,
        models=tuple(
            model
            for model in scenario.models
            if model.entry.settings.name != FIXED_MODEL_NAME
        ),
    )
    with pytest.raises(ValueError, match="exactly one efficientnet_b6"):
        instance_flows(scenario, _spec(scenario, model_count=1))


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
@pytest.mark.parametrize("mode", ["performance", "model_variant"])
async def test_campaign_records_pinned_requests_and_resumes_each_mode(
    scenario, fixed_config, tmp_path, monkeypatch, mode
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
            scenario.config if mode == "performance" else fixed_config,
            "main",
            tmp_path,
            resume=resume,
            retry_errors=False,
            limit=2,
        )
    assert calls == [mode, mode]
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
        if fixed:
            assert record["model_count"] == 1
            assert record["model_names"] == [FIXED_MODEL_NAME]
            assert all(
                flow["model_variant_id"]["model_id"]["model_name"] == FIXED_MODEL_NAME
                for flow in record["flows"]
            )
    with (tmp_path / "summary.csv").open() as handle:
        assert {row["model_selection"] for row in csv.DictReader(handle)} == {mode}


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
@pytest.mark.parametrize("strategy", ["full", "incremental", "incremental-persistent"])
async def test_native_solver_accepts_fixed_variant_requests(
    fixed_scenario, kind, objective, strategy
):
    scenario = fixed_scenario
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
        model_count=1,
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
