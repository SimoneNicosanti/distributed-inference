import csv
from pathlib import Path
from typing import Any

import pytest
from scalability_benchmark.analysis import (
    aggregate,
    incumbent_points,
    time_to_first_incumbent,
)
from scalability_benchmark.persistence import ResultStore, read_journal


def record(
    run_id: str,
    termination: str = "timeout_without_incumbent",
    runtime: float | None = 300,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "state": "finished",
        "campaign": "main",
        "seed": 42,
        "load": 0.6,
        "flow_count": 1,
        "model_count": 1,
        "worker_count": 1,
        "termination": termination,
        "solver_runtime_s": runtime,
        "feasible_incumbent": False,
        "final_gap": None,
        "timeout_reached": termination.startswith("timeout"),
        "infeasibility_proven": False,
    }


def test_resume_preserves_completed_timeouts_and_retries_incomplete(
    tmp_path: Path,
) -> None:
    with ResultStore(tmp_path, resume=False) as store:
        store.append(record("complete"))
        store.append({"run_id": "unfinished", "state": "started"})
        store.append(record("error", "solver_error", None))
    with (tmp_path / "runs.jsonl").open("ab") as handle:
        handle.write(b'{"run_id":"torn')
    with ResultStore(tmp_path, resume=True) as store:
        assert store.completed("complete")
        assert not store.completed("unfinished")
        assert store.completed("error")
    with ResultStore(tmp_path, resume=True, retry_errors=True) as store:
        assert not store.completed("error")
    assert (tmp_path / "runs.jsonl").read_bytes().endswith(b"\n")
    with (tmp_path / "summary.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 2
    with pytest.raises(FileExistsError), ResultStore(tmp_path, resume=False):
        pass


def test_committed_corruption_is_not_silently_discarded(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    path.write_bytes(b'{"invalid":\n')
    with pytest.raises(ValueError, match="Corrupt committed"):
        read_journal(path, repair=True)
    assert path.read_bytes() == b'{"invalid":\n'


def test_only_one_writer_can_enter(tmp_path: Path) -> None:
    with (
        ResultStore(tmp_path, resume=True),
        pytest.raises(BlockingIOError),
        ResultStore(tmp_path, resume=True),
    ):
        pass


def test_aggregation_keeps_timeouts_and_missing_data_explicit() -> None:
    fast = record("fast", "optimal_closed_gap", 1)
    fast.update(feasible_incumbent=True, final_gap=0)
    rows = aggregate([fast, record("timeout"), record("error", "solver_error", None)])
    assert len(rows) == 3
    for row in rows:
        assert row["completed_count"] == 3
        assert row["runtime_observed_count"] == 2
        assert row["runtime_mean_s"] == 150.5
        assert row["timeout_rate"] == 1 / 3
        assert row["gap_observed_count"] == 1


def test_aggregation_never_pools_different_objectives() -> None:
    first = {
        **record("first", "optimal_closed_gap", 1),
        "objective": "deadline_violations",
    }
    second = {
        **record("second", "optimal_closed_gap", 9),
        "objective": "arrival_weighted_accuracy",
    }
    rows = aggregate([first, second])
    assert len(rows) == 6
    assert all(row["completed_count"] == 1 for row in rows)
    assert {row["objective"]: row["runtime_mean_s"] for row in rows} == {
        "deadline_violations": 1,
        "arrival_weighted_accuracy": 9,
    }


def test_historical_objective_metadata_is_preserved() -> None:
    historic = record("historic")
    historic["requested_solver_configuration"] = {
        "objective": "arrival_weighted_response_slack"
    }
    assert all(
        row["objective"] == "arrival_weighted_response_slack"
        for row in aggregate([historic])
    )


def test_first_incumbent_summary_preserves_zero_and_missing_observations() -> None:
    zero = {**record("zero"), "time_to_first_incumbent_s": 0.0}
    later = {**record("later"), "time_to_first_incumbent_s": 2.0}
    invalid = {**record("invalid"), "time_to_first_incumbent_s": float("nan")}
    rows = aggregate([zero, later, invalid, record("historic")])
    assert time_to_first_incumbent(zero) == 0.0
    assert time_to_first_incumbent(invalid) is None
    for row in rows:
        assert row["first_incumbent_observed_count"] == 2
        assert row["first_incumbent_median_s"] == 1.0
    assert all(
        row["first_incumbent_median_s"] is None
        for row in aggregate([record("historic")])
    )


def test_incumbent_points_keep_missing_values_and_final_fraction() -> None:
    traced = {
        **record("traced"),
        "raw_solver_statistics": {
            "incumbent_history": [
                {"runtime_s": 0.0, "objective_value": None},
                {"runtime_s": 1.0, "objective_value": 5.0},
                {"runtime_s": 2.0, "objective_value": 0.0},
                {"runtime_s": 2.3, "objective_value": 0.0},
            ]
        },
    }
    assert incumbent_points(traced) == [(0, None), (1, 5), (2, 0), (2.3, 0)]
    assert incumbent_points(record("historic")) == []
    assert incumbent_points({"raw_solver_statistics": None}) == []


def test_offline_plot_generation(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from scalability_benchmark.analysis import analyze

    with ResultStore(tmp_path, resume=False) as store:
        store.append(record("plot"))
    analyze(tmp_path)
    assert (tmp_path / "analysis/aggregated.csv").is_file()
    assert len(list((tmp_path / "analysis").glob("*.pdf"))) == 3
    assert len(list((tmp_path / "analysis").glob("*.png"))) == 3


def test_offline_incumbent_plot_generation(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from scalability_benchmark.analysis import analyze

    traced = {
        **record("traced", "timeout_with_incumbent", 2.3),
        "objective": "deadline_violations",
        "feasible_incumbent": True,
        "time_to_first_incumbent_s": 0.2,
        "raw_solver_statistics": {
            "incumbent_history": [
                {"runtime_s": 0.0, "objective_value": None},
                {"runtime_s": 1.0, "objective_value": 1.0},
                {"runtime_s": 2.0, "objective_value": 0.0},
                {"runtime_s": 2.3, "objective_value": 0.0},
            ]
        },
    }
    with ResultStore(tmp_path, resume=False) as store:
        store.append(traced)
        store.append(record("historic"))
    analyze(tmp_path)
    destination = tmp_path / "analysis"
    for extension in ("pdf", "png"):
        assert (
            destination
            / f"main-deadline_violations-load-0.6-time-to-first-incumbent.{extension}"
        ).is_file()
        assert (destination / f"incumbents/traced.{extension}").is_file()
        assert not (destination / f"incumbents/historic.{extension}").exists()
    with (destination / "aggregated.csv").open(newline="") as handle:
        observed = [
            row
            for row in csv.DictReader(handle)
            if row["objective"] == "deadline_violations"
        ]
    assert len(observed) == 3
    assert all(row["first_incumbent_observed_count"] == "1" for row in observed)
