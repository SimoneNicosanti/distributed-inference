"""Component priorities retain full/incremental execution after selector removal."""

import sys
from dataclasses import replace

import pytest
from pydantic import ValidationError
from scalability_benchmark import cli
from scalability_benchmark.campaigns import RunSpec
from scalability_benchmark.configuration import ExperimentConfig
from scalability_benchmark.solver import execute_solver

from optimizer.domain.formulation.compact.misocp.compact_formulation_kind import (
    CompactFormulationKind as Kind,
)

from . import test_optimizer_contract as fixtures

scenario = fixtures.scenario


def test_removed_selector_is_rejected(config):
    raw = config.model_dump()
    raw["optimization"]["priority_formulation"] = "flow"
    with pytest.raises(ValidationError, match="priority_formulation"):
        ExperimentConfig.model_validate(raw)


def test_removed_cli_selector_is_rejected(monkeypatch, capsys):
    monkeypatch.setattr(
        sys, "argv", ["scalability", "plan", "--priority-formulation", "flow"]
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", list(Kind))
@pytest.mark.parametrize("strategy", ["full", "incremental", "incremental-persistent"])
async def test_component_solver_retains_statistics_and_observations(
    scenario, kind, strategy
):
    config = scenario.config.model_copy(
        update={
            "optimization": scenario.config.optimization.model_copy(
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
        10,
        formulation=kind,
        strategy=strategy,
    )
    outcome = await execute_solver(scenario, spec, config)
    assert outcome["solution_extracted"], outcome
    assert outcome["component_priority_variables"] == 4
    assert len(outcome["flow_metrics"]) == 2
    assert outcome["model_dimensions"]["quadratic_constraints"] > 0
    assert outcome["worker_priority_utilization"]
    assert replace(spec, strategy="full").formulation is kind
