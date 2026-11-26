from typing import Any

import pytest
from scalability_benchmark.aggregate_time_plots import (
    empirical_cdf,
    grouped_times,
    summarize_times,
)


def observation(
    runtime: object,
    first: object = None,
    *,
    timeout: bool = False,
    error: bool = False,
    wall: object = None,
    workers: int = 1,
    flows: int = 1,
) -> dict[str, Any]:
    return {
        "solver_runtime_s": runtime,
        "wall_time_s": wall,
        "time_to_first_incumbent_s": first,
        "timeout_reached": timeout,
        "termination": "solver_error" if error else "other_termination",
        "worker_count": workers,
        "flow_count": flows,
        "model_count": 1,
    }


def test_timeout_and_missing_times_have_distinct_denominators() -> None:
    records = [
        observation(1, 0),
        observation(300, 2, timeout=True),
        observation(300, timeout=True),
        observation(0.5, error=True),
    ]
    runtime = summarize_times(records, "solver_runtime_s")
    first = summarize_times(records, "time_to_first_incumbent_s")
    assert runtime.total == first.total == 4
    assert runtime.values == (1.0, 300.0, 300.0)
    assert runtime.median_s == 300
    assert runtime.timeouts == (300.0, 300.0)
    assert first.values == (0.0, 2.0)
    assert first.median_s == 1
    assert first.timeouts == (2.0,)


@pytest.mark.parametrize("value", [None, True, -1, float("nan"), float("inf")])
def test_invalid_or_missing_times_are_not_zero_samples(value: object) -> None:
    for metric in ("solver_runtime_s", "time_to_first_incumbent_s", "wall_time_s"):
        summary = summarize_times([observation(value, value, wall=value)], metric)
        assert summary.total == 1
        assert summary.values == ()
        assert summary.median_s is None


def test_cdf_preserves_zero_and_tied_observations() -> None:
    x, y = empirical_cdf((2, 0, 0))
    assert x == [0, 0, 0, 2]
    assert y == [0, 1 / 3, 2 / 3, 1]
    assert empirical_cdf(()) == ([], [])


def test_pairwise_cells_keep_observation_coverage_and_missing_combinations() -> None:
    records = [
        observation(1, 0, workers=1, flows=1),
        observation(300, workers=1, flows=1, timeout=True),
        observation(4, 2, workers=2, flows=2),
    ]
    cells = grouped_times(
        records, "time_to_first_incumbent_s", ("worker_count", "flow_count")
    )
    assert cells[(1, 1)].total == 2
    assert cells[(1, 1)].values == (0.0,)
    assert cells[(1, 1)].median_s == 0
    assert cells[(2, 2)].median_s == 2
    assert (1, 2) not in cells


def test_unknown_metric_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown time metric"):
        summarize_times([], "typo")


def test_wall_time_uses_recorded_elapsed_time_and_keeps_coverage() -> None:
    records = [
        observation(1, wall=0),
        observation(3, wall=30, timeout=True),
        observation(300, timeout=True),
        observation(0.5, wall=2, error=True),
    ]
    summary = summarize_times(records, "wall_time_s")
    assert summary.total == 4
    assert summary.values == (0.0, 30.0)
    assert summary.median_s == 15
    assert summary.timeouts == (30.0,)


def test_analysis_renders_wall_time_boxplots_in_separate_experiment_group(
    tmp_path,
) -> None:
    from scalability_benchmark.analysis import analyze
    from scalability_benchmark.persistence import ResultStore

    record = {
        **observation(1, 0.1, wall=3),
        "run_id": "wall-time",
        "state": "finished",
        "campaign": "main",
        "objective": "arrival_weighted_normalized_delay",
        "load": 0.7,
        "formulation": "cumulative",
        "strategy": "incremental",
        "model_selection": "model_variant",
        "feasible_incumbent": True,
        "infeasibility_proven": False,
        "final_gap": 0,
    }
    with ResultStore(tmp_path, resume=False) as store:
        store.append(record)
        store.append({**record, "run_id": "missing-wall-time", "wall_time_s": None})
    analyze(tmp_path)
    destination = tmp_path / "analysis/model_variant/incremental/cumulative"
    prefix = "main-arrival_weighted_normalized_delay-load-0.7"
    for kind in ("time-boxplots", "wall-time-boxplots", "time-cdf", "time-heatmaps"):
        for extension in ("png", "pdf"):
            assert (destination / f"{prefix}-{kind}.{extension}").stat().st_size > 0
