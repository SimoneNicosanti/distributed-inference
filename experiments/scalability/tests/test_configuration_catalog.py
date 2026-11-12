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
from scalability_benchmark.configuration import BENCHMARK_OBJECTIVES, ExperimentConfig
from scalability_benchmark.randomness import fingerprint, stream

from controller.domain.optimization.solving.formulation.compact.compact_objective_kind import (
    CompactObjectiveKind,
)
from shared.model.model_variant import PerformanceMetricType


def test_default_campaign_contract(config: ExperimentConfig) -> None:
    main = list(campaign_specs(config, "main"))
    sensitivity = list(campaign_specs(config, "sensitivity"))
    assert len(main) == 770 and len(sensitivity) == 198
    assert len(config.catalog.models) == 10 and config.flows.pool_size == 7
    assert config.main.flows == tuple(range(1, 8))
    assert config.main.models == tuple(range(1, 11))
    assert config.main.loads == (0.7,)
    assert config.pilot.load == 0.7
    assert config.flows.deadline_s == (0.2, 3.0)
    assert config.sensitivity.loads == (0.3, 0.8)
    assert config.sensitivity.flows == (1, 4, 7)
    assert config.sensitivity.models == (1, 5, 10)
    pilot = list(campaign_specs(config, "pilot"))
    assert len(pilot) == 6
    assert any(spec.flow_count == 7 and spec.model_count == 10 for spec in pilot)
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
    assert len({fingerprint(spec.as_record()) for spec in main + sensitivity}) == 968
    assert (
        len(list(campaign_specs(config, "main", objectives=BENCHMARK_OBJECTIVES)))
        == 3080
    )
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
    for count in range(1, len(config.catalog.models)):
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


@pytest.mark.parametrize("campaign", ["main", "sensitivity", "pilot"])
@pytest.mark.parametrize("dimension", ["flows", "models"])
def test_reject_counts_outside_configured_pools(
    config: ExperimentConfig, campaign: str, dimension: str
) -> None:
    raw = config.model_dump()
    maximum = (
        config.flows.pool_size if dimension == "flows" else len(config.catalog.models)
    )
    if campaign == "pilot":
        raw["pilot"]["cases"][0][dimension] = maximum + 1
    else:
        raw[campaign][dimension] = [maximum + 1]
    with pytest.raises(ValidationError, match="outside supported pool"):
        ExperimentConfig.model_validate(raw)


def test_model_pool_requires_entries_and_unique_identity(
    config: ExperimentConfig,
) -> None:
    for models in ([], [config.catalog.models[0].model_dump()] * 10):
        raw = config.model_dump()
        raw["catalog"]["models"] = models
        with pytest.raises(ValidationError):
            ExperimentConfig.model_validate(raw)


def test_compact_is_default_and_legacy_is_rejected(config: ExperimentConfig) -> None:
    raw = config.model_dump()
    raw["solver"].pop("formulation")
    assert ExperimentConfig.model_validate(raw).solver.formulation == "compact"
    raw["solver"]["formulation"] = "legacy"
    with pytest.raises(ValidationError, match="formulation"):
        ExperimentConfig.model_validate(raw)


@pytest.mark.parametrize("objective", BENCHMARK_OBJECTIVES)
def test_all_four_objectives_are_accepted(
    config: ExperimentConfig, objective: CompactObjectiveKind
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
