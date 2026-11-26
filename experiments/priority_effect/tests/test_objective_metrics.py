"""Native solver metrics for the component-priority objective alternatives."""

from dataclasses import replace

import pytest
from priority_effect.configuration import Config
from priority_effect.solver import solve
from test.optimizer.test_cumulative_formulation import dag_problem

from optimizer.domain.problem.compact.compact_optimization_problem import ObjectiveKind


@pytest.mark.asyncio
@pytest.mark.parametrize("deadline", [None, 0.001])
async def test_normalized_tardiness_native_metrics(config, deadline):
    raw = config.model_dump()
    raw["optimization"].update(objective="arrival_weighted_normalized_delay")
    settings = Config.model_validate(raw)
    problem = dag_problem(variants=1, flows=2, components=3, priorities=2)
    problem = replace(
        problem,
        policy=replace(
            problem.policy, objective=ObjectiveKind.ARRIVAL_WEIGHTED_NORMALIZED_DELAY
        ),
    )
    if deadline is not None:
        problem = replace(
            problem,
            flows={
                f: replace(v, max_response_time_s=deadline)
                for f, v in problem.flows.items()
            },
        )
    result = await solve(problem, settings)
    assert result["solution_extracted"], result
    assert result["objective"] == "arrival_weighted_normalized_delay"
    if deadline is None:
        assert result["objective_value"] == pytest.approx(0, abs=1e-7)
        assert result["time_to_zero_tardiness_s"] is not None
    else:
        assert result["objective_value"] > 0
        assert result["time_to_zero_tardiness_s"] is None
    assert result["objective_recomputed"] == pytest.approx(
        result["aggregate_normalized_tardiness"], abs=1e-7
    )
    assert result["objective_recomputed"] == pytest.approx(
        result["objective_value"], rel=1e-4, abs=1e-6
    )
    assert (
        result["native_objective_scaling"]
        == "arrival_weighted_mean_normalized_tardiness"
    )


@pytest.mark.asyncio
async def test_deadline_saturation_metrics_and_hard_deadlines(config):
    raw = config.model_dump()
    raw["optimization"].update(
        objective="arrival_weighted_normalized_response_time",
    )
    settings = Config.model_validate(raw)
    problem = dag_problem(variants=1, flows=2, components=3, priorities=2)
    problem = replace(
        problem,
        policy=replace(
            problem.policy,
            objective=ObjectiveKind.ARRIVAL_WEIGHTED_NORMALIZED_RESPONSE_TIME,
        ),
    )
    result = await solve(problem, settings)
    assert result["solution_extracted"], result
    expected = sum(
        f["arrival_rate_per_s"] * f["response_time_s"] / f["deadline_s"]
        for f in result["flow_metrics"]
    ) / sum(f["arrival_rate_per_s"] for f in result["flow_metrics"])
    assert 0 < result["objective_value"] <= 1
    assert result["objective_recomputed"] == pytest.approx(expected, abs=1e-6)
    assert result["objective_recomputed"] == pytest.approx(
        result["objective_value"], abs=1e-6
    )
    assert (
        result["native_objective_scaling"] == "arrival_weighted_mean_deadline_fraction"
    )
    impossible = replace(
        problem,
        flows={
            f: replace(v, max_response_time_s=1e-12) for f, v in problem.flows.items()
        },
    )
    rejected = await solve(impossible, settings)
    assert rejected["termination"] == "proven_infeasible", rejected
