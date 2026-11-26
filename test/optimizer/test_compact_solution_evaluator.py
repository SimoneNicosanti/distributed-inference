"""Independent queue, DAG, network, objective and solution-contract checks."""

from dataclasses import replace

import pytest

from optimizer.application.evaluation.compact_solution_evaluator import (
    evaluate_decisions,
    validate_solution,
)
from optimizer.application.evaluation.queueing import QueueingMode, bin_upper
from optimizer.application.formulation.compact.misocp.misocp_compact_formulation_builder import (
    MisocpCompactFormulationBuilder,
)
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation_config import (
    MisocpCompactFormulationConfig,
)
from optimizer.domain.problem.compact.compact_optimization_problem import ObjectiveKind

from .compact_fixtures import decisions, problem


def prepared(instance):
    return MisocpCompactFormulationBuilder().build_formulation(
        instance, MisocpCompactFormulationConfig(0.1)
    )


def test_single_worker_exact_wait_and_service_second_moment():
    p = problem(variants=1, workers=1, components=1, priorities=1)
    report = evaluate_decisions(prepared(p), decisions(p))
    assert report.valid
    service, arrival = 0.03, 0.2
    wait = arrival * service**2 / (2 * (1 - arrival * service))
    assert report.solution.flow_solutions[0].mean_response_time_s == pytest.approx(
        service + wait
    )
    assert report.queue_wait_s[next(iter(p.workers)), 1] == pytest.approx(wait)
    assert validate_solution(prepared(p), report.solution).valid


def test_multiclass_wait_uses_all_classes_in_numerator_and_prefix():
    p = problem(flows=2, variants=1, workers=1, components=1)
    ds = decisions(p)
    flows = tuple(ds)
    ds[flows[1]] = replace(ds[flows[1]], priorities=(2,))
    report = evaluate_decisions(prepared(p), ds)
    assert report.valid
    worker = next(iter(p.workers))
    numerator = 2 * 0.2 * 0.03**2
    assert report.queue_wait_s[worker, 1] == pytest.approx(
        numerator / (2 * (1 - 0.006))
    )
    assert report.queue_wait_s[worker, 2] == pytest.approx(
        numerator / (2 * (1 - 0.006) * (1 - 0.012))
    )
    invalid = {f: replace(d, priorities=(2,)) for f, d in ds.items()}
    assert not evaluate_decisions(prepared(p), invalid).valid


def test_fanout_tensor_once_per_cut_and_shared_source_ps_queue():
    p = problem(flows=2)
    data = prepared(p)
    report = evaluate_decisions(data, decisions(p, split=True, variant=1))
    assert report.valid
    flows = report.solution.flow_solutions
    assert flows[0].cuts[0].tensor_keys == ("t0",)
    assert flows[0].cuts[1].tensor_keys == ("t3",)
    assert flows[0].components[0].mean_transmission_time_s == pytest.approx(
        0.001 / (1 - 2 * 0.2 * 0.001)
    )
    assert flows[0].components[1].mean_transmission_time_s == pytest.approx(
        0.0002 / (1 - 2 * 0.2 * 0.0002)
    )
    assert sum(c.transmission_latency_s for c in flows[0].components) == 0.002
    # Every used worker, even one hosting only I/O, reserves full activation bound.
    assert all(amount >= 500 for amount in report.memory_bytes.values())


def test_legacy_bins_are_conservative_and_ties_choose_lower_bin():
    assert bin_upper(0.1, (0.0, 0.1, 0.2)) == 0.1
    p = problem(flows=2)
    data, ds = prepared(p), decisions(p, split=True, variant=1)
    native = evaluate_decisions(data, ds)
    legacy = evaluate_decisions(data, ds, mode=QueueingMode.LEGACY_EQUIVALENT)
    assert native.valid and legacy.valid
    assert legacy.solution.objective_value >= native.solution.objective_value


@pytest.mark.parametrize("kind", list(ObjectiveKind))
def test_all_objectives_and_deadline_semantics(kind):
    p = problem(
        variants=1,
        workers=1,
        components=1,
        priorities=1,
        objective=kind,
        deadline=0.001,
    )
    report = evaluate_decisions(prepared(p), decisions(p))
    hard = kind in (
        ObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_TIME,
        ObjectiveKind.ARRIVAL_WEIGHTED_NORMALIZED_RESPONSE_TIME,
        ObjectiveKind.ARRIVAL_WEIGHTED_ACCURACY,
    )
    assert report.valid is not hard
    if kind == ObjectiveKind.DEADLINE_VIOLATIONS:
        assert report.solution.objective_value == 1
    if kind == ObjectiveKind.ARRIVAL_WEIGHTED_NORMALIZED_DELAY:
        assert report.solution.objective_value > 0


def test_unstable_queue_is_finite_but_rejected():
    p = problem(variants=1, workers=1, components=1, priorities=1, rate=100)
    report = evaluate_decisions(prepared(p), decisions(p))
    assert not report.valid
    assert any("stability" in key for key in report.violations)
    assert report.solution is not None


@pytest.mark.parametrize(
    "fault",
    (
        "precedence",
        "empty_component",
        "unsupported",
        "relay",
        "memory",
        "link",
        "same_worker",
    ),
)
def test_structural_and_resource_violations(fault):
    p = problem()
    ds = decisions(p, split=True, variant=1)
    flow = next(iter(ds))
    decision = ds[flow]
    if fault == "precedence":
        ds[flow] = replace(
            decision, component_by_layer={**decision.component_by_layer, "join": 1}
        )
    elif fault == "empty_component":
        ds[flow] = replace(
            decision,
            component_by_layer={
                **decision.component_by_layer,
                **dict.fromkeys(("a", "b", "join"), 1),
            },
        )
    elif fault == "unsupported":
        times = {k: dict(v) for k, v in p.execution_times_s.items()}
        del times[decision.configuration, decision.workers[1]]["a"]
        p = replace(p, execution_times_s=times)
    elif fault == "relay":
        request = p.flows[flow]
        p = replace(
            p,
            flows={
                flow: replace(
                    request,
                    local_worker_config=replace(
                        request.local_worker_config,
                        worker_id=decision.workers[1],
                        is_inference_runner=False,
                    ),
                )
            },
        )
    elif fault == "memory":
        p = replace(
            p,
            workers={
                s: replace(w, available_inference_memory_bytes=10)
                for s, w in p.workers.items()
            },
        )
    elif fault == "link":
        p = replace(p, connections=())
    else:
        ds[flow] = replace(decision, workers=(decision.workers[0],) * 3)
    assert not evaluate_decisions(prepared(p), ds).valid


def test_batch_arrivals_and_fill_preserve_units():
    p = problem(variants=1, workers=1, components=1, priorities=1, batch=4)
    report = evaluate_decisions(prepared(p), decisions(p))
    assert report.valid
    flow = report.solution.flow_solutions[0]
    assert flow.mean_batch_fill_time_s == 7.5
    assert report.compute_utilization[next(iter(p.workers))] == pytest.approx(
        (0.2 / 4) * (0.03 * 4)
    )


def test_validator_detects_timing_and_objective_tampering():
    p = problem(variants=1, workers=1, components=1, priorities=1)
    data = prepared(p)
    report = evaluate_decisions(data, decisions(p))
    solution = report.solution
    assert not validate_solution(
        data, replace(solution, objective_value=solution.objective_value + 0.01)
    ).valid
    flow = solution.flow_solutions[0]
    bad = replace(flow.components[0], mean_execution_queue_time_s=0.0)
    assert not validate_solution(
        data, replace(solution, flow_solutions=(replace(flow, components=(bad,)),))
    ).valid
