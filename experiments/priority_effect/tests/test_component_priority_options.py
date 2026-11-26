"""Only the component-priority experiment interfaces remain selectable."""

import sys
from pathlib import Path

import pytest
from priority_effect import cli
from priority_effect.configuration import Config, load_config
from pydantic import ValidationError


@pytest.mark.parametrize("option", ["priority_assignment", "priority_formulation"])
def test_removed_priority_options_are_rejected(config, option):
    raw = config.model_dump()
    raw["optimization"][option] = "flow"
    with pytest.raises(ValidationError, match=option):
        Config.model_validate(raw)


@pytest.mark.parametrize(
    "strategy", ["full_p2_flow", "full_p2_native_flow", "two_stage_priority"]
)
def test_removed_strategies_are_rejected(config, strategy):
    raw = config.model_dump()
    raw["strategies"] = [strategy]
    with pytest.raises(ValidationError, match="strategies"):
        Config.model_validate(raw)


def test_removed_cli_selector_is_rejected(monkeypatch, capsys):
    monkeypatch.setattr(
        sys, "argv", ["priority-effect", "plan", "--priority-formulation", "flow"]
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_packaged_component_configurations():
    paths = sorted(
        (Path(__file__).resolve().parents[1] / "config").glob(
            "component-priority*.yaml"
        )
    )
    assert paths
    for path in paths:
        config = load_config(path)
        assert set(config.strategies) <= {"full_p1", "full_p2"}
        assert (
            not {"priority_assignment", "priority_formulation"}
            & config.optimization.model_dump().keys()
        )


@pytest.mark.parametrize(
    "option",
    [
        "total_time_limit_s",
        "stage_a_time_limit_s",
        "lift_time_limit_s",
        "fallback_full_p2",
    ],
)
def test_removed_staged_solver_options_are_rejected(config, option):
    raw = config.model_dump()
    raw["solver"][option] = False if option == "fallback_full_p2" else 30
    with pytest.raises(ValidationError, match=option):
        Config.model_validate(raw)
