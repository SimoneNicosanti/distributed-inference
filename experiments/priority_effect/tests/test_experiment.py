import json
from dataclasses import replace

import pytest
from priority_effect.analysis import aggregate, analyze, compare
from priority_effect.configuration import Config
from priority_effect.scenarios import (
    Spec,
    assign_models,
    calibrate,
    generate,
    jsonable,
    paired_problem,
    planned,
)
from pydantic import ValidationError
from scalability_benchmark.persistence import ResultStore, read_journal
from scalability_benchmark.solver import classify

from optimizer.adapters.outbound.gurobi.entities.gurobi_run_statistics import (
    GurobiRunStatistics,
)
from optimizer.application.ports.outbound.solver_result import (
    SolverOutput,
    SolverTerminationKind,
)
from optimizer.domain.problem.compact.compact_optimization_problem import ObjectiveKind


def test_identical_paired_inputs_and_single_variant(config, prepared):
    problem, metadata = generate(config, prepared, Spec(42, 2, 0.3, "heterogeneous"))
    one, two = (paired_problem(problem, p) for p in (1, 2))
    assert one.flows == two.flows
    assert one.execution_times_s.keys() == two.execution_times_s.keys()
    first, second = jsonable(one), jsonable(two)
    assert first["policy"].pop("priority_count") == 1
    assert second["policy"].pop("priority_count") == 2
    assert first == second
    assert all(len(flow.configurations) == 1 for flow in problem.flows.values())
    assert one.policy.max_components == two.policy.max_components == 5
    assert one.policy.objective.value == "arrival_weighted_response_time"
    assert metadata["reference"]["analytic_checks_pass"]


def test_deterministic_generation_and_solve_order_independence(config, prepared):
    spec = Spec(42, 2, 0.8, "heterogeneous")
    problem, metadata = generate(config, prepared, spec)
    paired_problem(problem, 2)
    other, record = generate(config, prepared, spec)
    assert metadata == record
    assert jsonable(problem) == jsonable(other)
    # Changing the offered load preserves profiles, assignments and weight draws.
    _, lower = generate(config, prepared, replace(spec, load=0.3))
    assert lower["model_assignments"] == metadata["model_assignments"]
    assert lower["arrival_weights"] == metadata["arrival_weights"]
    assert lower["execution"] == metadata["execution"]


@pytest.mark.parametrize(
    "objective",
    [
        ObjectiveKind.ARRIVAL_WEIGHTED_NORMALIZED_DELAY,
        ObjectiveKind.ARRIVAL_WEIGHTED_NORMALIZED_RESPONSE_TIME,
    ],
)
def test_normalized_objectives_preserve_workload_and_reach_problem(
    config, prepared, objective
):
    raw = config.model_dump()
    raw["optimization"]["objective"] = objective
    tardiness = Config.model_validate(raw)
    spec = Spec(42, 2, 0.8, "heterogeneous")
    baseline, original = generate(config, prepared, spec)
    problem, changed = generate(tardiness, prepared, spec)
    assert problem.policy.objective is objective
    assert replace(problem, policy=baseline.policy) == baseline
    for field in ("model_assignments", "arrival_weights", "reference", "execution"):
        assert changed[field] == original[field]


def test_reference_calibration():
    services, weights = (0.01, 0.1, 0.3), (1.0, 2.0, 0.5)
    rates = calibrate(0.7, services, weights)
    assert sum(
        rate * service for rate, service in zip(rates, services, strict=True)
    ) == pytest.approx(0.7)
    assert rates[1] / rates[0] == pytest.approx(2)
    for bad in ((0.0, 1.0), (float("nan"), 1.0)):
        with pytest.raises(ValueError):
            calibrate(0.7, bad, (1.0, 1.0))


@pytest.mark.asyncio
async def test_complete_regeneration_from_seed(config, prepared, tmp_path):
    from priority_effect.scenarios import prepare

    rebuilt = await prepare(config, 42, tmp_path / "fresh-profiles", "unit-fixture")
    spec = Spec(42, 2, 0.3, "heterogeneous")
    _, original = generate(config, prepared, spec)
    _, regenerated = generate(config, rebuilt, spec)
    assert original == regenerated
    different = await prepare(config, 43, tmp_path / "other-seed", "unit-fixture")
    _, changed = generate(config, different, replace(spec, seed=43))
    assert original["problem_fingerprint"] != changed["problem_fingerprint"]
    assert original["network"] != changed["network"]


def test_profile_based_and_explicit_assignments(config, prepared):
    _, heterogeneous = generate(config, prepared, Spec(42, 2, 0.3, "heterogeneous"))
    assert set(heterogeneous["model_assignments"].values()) == {
        "resnet50",
        "efficientnet_b0",
    }
    assert heterogeneous["service_demand_cv"] > 0
    _, homogeneous = generate(config, prepared, Spec(42, 2, 0.3, "homogeneous"))
    assert set(homogeneous["model_assignments"].values()) == {"resnet50"}
    assert homogeneous["service_demand_cv"] == 0
    workload = config.workloads[1].model_copy(
        update={"assignments": {2: ("efficientnet_b0", "resnet50")}}
    )
    assert assign_models(workload, 2, heterogeneous["profiled_model_demands_s"], 9) == (
        "efficientnet_b0",
        "resnet50",
    )


def test_deadlines_reference_queue_and_resources(config, prepared):
    problem, metadata = generate(config, prepared, Spec(42, 2, 0.8, "heterogeneous"))
    reference = metadata["reference"]
    rates = [data.arrival_rate_per_s for data in problem.flows.values()]
    services = list(metadata["reference_services_s"].values())
    expected_wait = sum(
        rate * service**2 for rate, service in zip(rates, services, strict=True)
    ) / (2 * (1 - 0.8))
    assert reference["compute_wait_s"] == pytest.approx(expected_wait)
    for flow, data in problem.flows.items():
        assert data.max_response_time_s == pytest.approx(
            config.deadline.multiplier
            * reference["response_times_s"][str(flow.flow_id)]
        )
    assert not reference["feasibility_certificate"]
    assert reference["offered_reference_load"] == pytest.approx(0.8)
    # Invalid networks are reported, never repaired or excluded.
    slow = replace(
        problem,
        connections=tuple(
            replace(link, throughput_bytes_per_sec=1) for link in problem.connections
        ),
    )
    from priority_effect.reference import reference_estimate

    assert not reference_estimate(slow, 0.1)["resource_and_qos_checks"][
        "network_capacity"
    ]


@pytest.mark.parametrize(
    "kind,incumbent,expected",
    [
        (SolverTerminationKind.INFEASIBLE, False, "proven_infeasible"),
        (SolverTerminationKind.TIME_LIMIT, False, "timeout_without_incumbent"),
        (SolverTerminationKind.TIME_LIMIT, True, "timeout_with_incumbent"),
    ],
)
def test_native_termination_distinctions(kind, incumbent, expected):
    statistics = GurobiRunStatistics(
        native_status=3 if kind == SolverTerminationKind.INFEASIBLE else 9,
        solution_count=int(incumbent),
        relative_gap=0.5 if incumbent else None,
        construction_time_s=1,
        time_limit_s=1,
        mip_gap_target=0.01,
        threads=1,
        seed=0,
    )
    output = SolverOutput(
        termination=kind,
        solved_model=object() if incumbent else None,
        objective_value=1 if incumbent else None,
        objective_bound=0,
        runtime_s=1,
        statistics=statistics,
    )
    from optimizer.domain.problem.compact.compact_optimization_problem import (
        ObjectiveKind,
    )

    result = classify(
        output, None, objective=ObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_TIME
    )
    assert result["termination"] == expected
    assert result["infeasibility_proven"] == (kind == SolverTerminationKind.INFEASIBLE)


def observation(priority=1, **updates):
    return {
        "run_id": f"scenario-p{priority}",
        "state": "finished",
        "scenario_id": "scenario",
        "priority_count": priority,
        "seed": 42,
        "load": 0.7,
        "workload": "mixed",
        "category": "heterogeneous",
        "flow_count": 2,
        "service_demand_cv": 0.5,
        "termination": "gap_target_reached",
        "solver_status_name": "OPTIMAL",
        "feasible_incumbent": True,
        "solution_extracted": True,
        "objective_value": 10.0 if priority == 1 else 8.0,
        "arrival_weighted_mean_response_time_s": 0.5 if priority == 1 else 0.4,
        "objective_bound": 9.9 if priority == 1 else 7.99,
        "final_gap": 0.01,
        "solver_runtime_s": 2.0 * priority,
        "construction_time_s": 0.1 * priority,
        **updates,
    }


def test_paired_improvement_bounds_and_negative_results():
    pair = compare(observation(), observation(2), 0.01)
    assert pair["improvement_pct"] == pytest.approx(20)
    assert pair["absolute_response_reduction_s"] == pytest.approx(0.1)
    assert pair["delta_solver_runtime_s"] == 2
    assert pair["certified_improvement"]
    assert pair["objective_reduction_lower_bound"] == pytest.approx(1.9)
    negative = compare(
        observation(),
        observation(
            2,
            objective_value=11,
            objective_bound=1,
            arrival_weighted_mean_response_time_s=0.55,
            final_gap=0.9,
        ),
        0.01,
    )
    assert negative["improvement_pct"] < 0
    assert negative["comparison_quality"] == "provisional_incumbents"
    assert not negative["certified_improvement"]
    no_bound = compare(
        observation(objective_bound=None, final_gap=None), observation(2), 0.01
    )
    assert not no_bound["certified_improvement"]


@pytest.mark.parametrize("missing", [None, float("inf"), float("nan")])
def test_unavailable_objectives_excluded(config, tmp_path, missing):
    # JSON journals reject nonfinite numbers; numerical comparison also rejects them.
    assert not compare(observation(objective_value=missing), observation(2), 0.01)[
        "valid_pair"
    ]
    with ResultStore(tmp_path, resume=False) as store:
        store.append(
            observation(
                objective_value=None,
                feasible_incumbent=False,
                termination="timeout_without_incumbent",
            )
        )
        store.append(observation(2))
    report = analyze(tmp_path, plots=False)
    assert report["run_count"] == 2
    assert report["valid_pair_count"] == 0
    assert sum(row["count"] for row in report["feasibility"]) == 2
    assert report["aggregates"][0]["valid_pair_count"] == 0


def test_persistence_resume_and_incomplete_pair(tmp_path):
    with ResultStore(tmp_path, resume=False) as store:
        store.append(observation())
        store.append(observation(2, state="started"))
    with ResultStore(tmp_path, resume=True) as store:
        assert store.completed("scenario-p1")
        assert not store.completed("scenario-p2")
        store.append(observation(2, termination="solver_error"))
    with ResultStore(tmp_path, resume=True, retry_errors=True) as store:
        assert not store.completed("scenario-p2")
        store.append(observation(2))
    assert len(read_journal(tmp_path / "runs.jsonl")) == 2
    assert analyze(tmp_path, plots=False)["valid_pair_count"] == 1
    with pytest.raises(FileExistsError), ResultStore(tmp_path, resume=False):
        pass


def test_configuration_and_pilot_dimensions(config):
    assert len(planned(config, pilot=True)) == 4
    raw = config.model_dump()
    raw["priority_counts"] = (2, 1)
    with pytest.raises(ValidationError):
        Config.model_validate(raw)
    pairs = [compare(observation(), observation(2), 0.01)]
    assert aggregate(pairs)[0]["seed_count"] == 1
    assert json.loads(json.dumps(jsonable(planned(config))))


def test_offline_figures(tmp_path):
    for priority in (1, 2):
        record = observation(
            priority,
            weighted_decomposition={
                "computation_s": 0.1,
                "waiting_s": 0.2,
                "transmission_s": 0.1,
                "propagation_s": 0,
                "batch_fill_s": 0,
            },
        )
        with ResultStore(tmp_path, resume=priority == 2) as store:
            store.append(record)
    analyze(tmp_path)
    assert len(list((tmp_path / "analysis").glob("*.png"))) == 6


@pytest.mark.asyncio
async def test_execute_pairs_resume_and_configuration_guard(
    config, prepared, tmp_path, monkeypatch
):
    from priority_effect import cli

    config = config.model_copy(update={"output": tmp_path})
    calls = []

    async def fake_prepare(*args):
        return prepared

    async def fake_solve(problem, settings, *, reference=False, iis_path=None):
        calls.append((problem, settings.solver, reference))
        result = observation(problem.policy.priority_count)
        for name in (
            "run_id",
            "state",
            "scenario_id",
            "seed",
            "load",
            "workload",
            "category",
            "flow_count",
            "service_demand_cv",
        ):
            result.pop(name)
        return result

    monkeypatch.setattr(cli, "prepare", fake_prepare)
    monkeypatch.setattr(cli, "solve", fake_solve)
    monkeypatch.setattr(cli, "provenance", lambda: {"source_hash": "fixture"})
    specs = (Spec(42, 2, 0.3, "heterogeneous"),)
    await cli.execute(config, specs, resume=False, retry_errors=False, iis=False)
    assert len(calls) == 3
    experimental = [call for call in calls if not call[2]]
    one, two = sorted(experimental, key=lambda call: call[0].policy.priority_count)
    assert one[1] == two[1]
    assert jsonable(paired_problem(one[0], 2)) == jsonable(two[0])
    journal = (tmp_path / "runs.jsonl").read_bytes()
    await cli.execute(config, specs, resume=True, retry_errors=False, iis=False)
    assert len(calls) == 3
    assert (tmp_path / "runs.jsonl").read_bytes() == journal
    changed = config.model_copy(update={"tight_gap": 0.5})
    with pytest.raises(ValueError, match="different configuration"):
        await cli.execute(changed, specs, resume=True, retry_errors=False, iis=False)


@pytest.mark.asyncio
async def test_execute_joint_strategies_same_scenario_and_resume(
    config, prepared, tmp_path, monkeypatch
):
    from priority_effect import cli

    settings = config.model_copy(
        update={
            "output": tmp_path,
            "strategies": ("full_p1", "full_p2"),
        }
    )
    calls = []

    async def fake_prepare(*args):
        return prepared

    async def fake_full(problem, config, *, reference=False, iis_path=None):
        calls.append(
            (
                "reference" if reference else f"full_p{problem.policy.priority_count}",
                problem,
            )
        )
        return {
            "termination": "gap_target_reached",
            "feasible_incumbent": True,
            "solution_extracted": True,
            "objective_value": 1.0,
            "arrival_weighted_mean_response_time_s": 1.0,
            "wall_time_s": 0.1,
        }

    monkeypatch.setattr(cli, "prepare", fake_prepare)
    monkeypatch.setattr(cli, "solve", fake_full)
    monkeypatch.setattr(cli, "provenance", lambda: {"source_hash": "fixture"})
    specs = (Spec(42, 2, 0.3, "heterogeneous"),)
    await cli.execute(settings, specs, resume=False, retry_errors=False, iis=False)
    assert len(calls) == 3
    assert all(
        paired_problem(problem, 1) == paired_problem(calls[0][1], 1)
        for _, problem in calls
    )
    records = list(read_journal(tmp_path / "runs.jsonl").values())
    assert len({r["scenario_id"] for r in records}) == 1
    assert len({r["shared_problem_fingerprint"] for r in records}) == 1
    assert {r["strategy"] for r in records} == set(settings.strategies)
    original = (tmp_path / "runs.jsonl").read_bytes()
    await cli.execute(settings, specs, resume=True, retry_errors=False, iis=False)
    assert len(calls) == 3 and (tmp_path / "runs.jsonl").read_bytes() == original


def test_provenance_hashes_loaded_optimizer_sources(tmp_path, monkeypatch):
    from priority_effect import cli

    root = tmp_path / "optimizer"
    source = root / "domain/problem/compact/compact_optimization_problem.py"
    source.parent.mkdir(parents=True)
    source.write_text("value = 1\n")
    (root / "extra.py").write_text("value = 2\n")
    prefix = "services/optimizer/src/optimizer/"
    monkeypatch.setattr(cli.optimizer_problem, "__file__", str(source))
    monkeypatch.setattr(
        cli,
        "software_provenance",
        lambda: {"source_hash": "old", "source_files": {prefix + "old.py": "obsolete"}},
    )
    result = cli.provenance()
    assert prefix + "extra.py" in result["source_files"]
    assert prefix + "old.py" not in result["source_files"]
    assert result["optimizer_source_root"] == str(root)


def test_reused_heterogeneous_models_are_explicit_and_opt_in(config, prepared):
    from collections import Counter

    raw = config.model_dump()
    raw["flows"] = (4,)
    heterogeneous = raw["workloads"][1]
    names = tuple(heterogeneous["models"][:2]) * 2
    heterogeneous["assignments"] = {4: names}
    with pytest.raises(ValidationError, match="distinct models"):
        Config.model_validate(raw)
    heterogeneous["distinct_models_per_flow"] = False
    reused = Config.model_validate(raw)
    problem, metadata = generate(reused, prepared, Spec(42, 4, 0.8, "heterogeneous"))
    assert len(problem.flows) == 4
    assert Counter(metadata["model_assignments"].values()) == Counter(names)
    assert assign_models(reused.workloads[1], 4, {}, 42) == names
    assert paired_problem(problem, 1).flows == paired_problem(problem, 2).flows
    heterogeneous["assignments"] = {}
    with pytest.raises(ValidationError, match="explicit assignments"):
        Config.model_validate(raw)
    heterogeneous["assignments"] = {4: (names[0],) * 4}
    with pytest.raises(ValidationError, match="at least two models"):
        Config.model_validate(raw)
