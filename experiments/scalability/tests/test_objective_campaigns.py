import sys
from dataclasses import replace

import pytest
from scalability_benchmark import cli
from scalability_benchmark.campaigns import campaign_specs
from scalability_benchmark.configuration import ExperimentConfig
from scalability_benchmark.randomness import fingerprint

from controller.domain.optimization.solving.solver_config import ObjectiveKind


def test_each_instance_has_independent_runs_for_all_objectives(
    config: ExperimentConfig,
) -> None:
    baseline = list(campaign_specs(config, "pilot"))
    expanded = list(campaign_specs(config, "pilot", objectives=tuple(ObjectiveKind)))
    assert len(expanded) == 4 * len(baseline)
    for index, original in enumerate(baseline):
        group = expanded[4 * index : 4 * index + 4]
        assert tuple(spec.objective for spec in group) == tuple(ObjectiveKind)
        assert all(
            replace(spec, objective=original.objective) == original for spec in group
        )
        ids = {
            fingerprint(
                {
                    "scenario_id": "same",
                    "specification": spec.as_record(),
                    "software": "same",
                }
            )
            for spec in group
        }
        assert len(ids) == 4


def test_subset_selection_is_deduplicated_and_preserves_order(
    config: ExperimentConfig,
) -> None:
    selected = (
        ObjectiveKind.DEADLINE_VIOLATIONS,
        ObjectiveKind.ARRIVAL_WEIGHTED_ACCURACY,
    )
    specs = list(campaign_specs(config, "pilot", objectives=selected + selected))
    assert tuple(spec.objective for spec in specs[:2]) == selected
    assert len(specs) == 2 * len(list(campaign_specs(config, "pilot")))
    with pytest.raises(ValueError, match="at least one"):
        list(campaign_specs(config, "pilot", objectives=()))


@pytest.mark.parametrize(
    "selection,multiplier",
    [
        ("all", 4),
        ("deadline_violations", 1),
        ("deadline_violations arrival_weighted_accuracy", 2),
    ],
)
def test_plan_cli_reports_expanded_budget(
    config: ExperimentConfig,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    selection: str,
    multiplier: int,
) -> None:
    monkeypatch.setattr(cli, "load_configuration", lambda path: config)
    monkeypatch.setattr(
        sys, "argv", ["scalability", "plan", "--objectives", *selection.split()]
    )
    cli.main()
    output = capsys.readouterr().out
    count = len(list(campaign_specs(config, "main"))) * multiplier
    assert f"main: {count} runs;" in output


def test_cli_rejects_all_mixed_with_individual_objectives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["scalability", "plan", "--objectives", "all", "deadline_violations"],
    )
    with pytest.raises(SystemExit) as failure:
        cli.main()
    assert failure.value.code == 2
