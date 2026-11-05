from itertools import pairwise
from pathlib import Path

import pytest
from pydantic import ValidationError
from scalability_benchmark.campaigns import campaign_specs
from scalability_benchmark.catalog import (
    discover_catalog,
    selection_order,
    synthetic_accuracies,
)
from scalability_benchmark.configuration import ExperimentConfig
from scalability_benchmark.randomness import fingerprint, stream

from controller.domain.optimization.solving.solver_config import ObjectiveKind
from shared.model.model_variant import PerformanceMetricType


def test_default_campaign_contract(config: ExperimentConfig) -> None:
    main = list(campaign_specs(config, "main"))
    sensitivity = list(campaign_specs(config, "sensitivity"))
    assert len(main) == 330 and len(sensitivity) == 198
    assert config.seeds == (42,) and not config.sensitivity.enabled
    assert {spec.timeout_s for spec in main + sensitivity} == {300}
    assert config.solver.relative_gap == 0.01
    assert config.optimization.max_components == 7
    assert [spec.worker_count for spec in main[:11]] == list(range(1, 12))
    assert [
        (spec.seed, spec.load, spec.flow_count, spec.model_count, spec.worker_count)
        for spec in main
    ] == sorted(
        (spec.seed, spec.load, spec.flow_count, spec.model_count, spec.worker_count)
        for spec in main
    )
    assert len({fingerprint(spec.as_record()) for spec in main + sensitivity}) == 528
    assert {spec.timeout_s for spec in campaign_specs(config, "pilot")} == {60}


def test_accuracy_and_selection_are_independent(config: ExperimentConfig) -> None:
    accuracy = synthetic_accuracies(config, 42)
    order = selection_order(config, 42)
    assert order == selection_order(config, 42)
    assert set(order) == set(accuracy)
    ranked = sorted(config.catalog.models, key=lambda item: item.accuracy_rank)
    values = [accuracy[item.name].value for item in ranked]
    assert all(left < right for left, right in pairwise(values))
    assert all(
        metric.type == PerformanceMetricType.ACCURACY and metric.value > 0
        for metric in accuracy.values()
    )
    for count in range(1, 6):
        assert order[:count] == order[: count + 1][:-1]
    stream(42, "network", "extra-worker").random()
    assert accuracy == synthetic_accuracies(config, 42)


@pytest.mark.parametrize(
    "field,value", [("seeds", []), ("infrastructure", [[0, 0]]), ("unknown", True)]
)
def test_reject_inconsistent_configuration(
    config: ExperimentConfig, field: str, value: object
) -> None:
    raw = config.model_dump()
    raw[field] = value
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(raw)


def test_compact_is_default_and_legacy_is_rejected(config: ExperimentConfig) -> None:
    raw = config.model_dump()
    raw["solver"].pop("formulation")
    assert ExperimentConfig.model_validate(raw).solver.formulation == "compact"
    raw["solver"]["formulation"] = "legacy"
    with pytest.raises(ValidationError, match="formulation"):
        ExperimentConfig.model_validate(raw)


@pytest.mark.parametrize("objective", list(ObjectiveKind))
def test_all_four_objectives_are_accepted(
    config: ExperimentConfig, objective: ObjectiveKind
) -> None:
    raw = config.model_dump()
    raw["optimization"]["objective"] = objective.value
    assert ExperimentConfig.model_validate(raw).optimization.objective is objective


def test_missing_models_never_download(
    config: ExperimentConfig, tmp_path: Path
) -> None:
    local = config.model_copy(update={"root": tmp_path})
    with pytest.raises(FileNotFoundError, match=r"export_models\.py"):
        discover_catalog(local, 42)
    assert list(tmp_path.iterdir()) == []
