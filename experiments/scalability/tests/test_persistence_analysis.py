import csv
from pathlib import Path
from typing import Any

import pytest
from scalability_benchmark.analysis import aggregate
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


def test_offline_plot_generation(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from scalability_benchmark.analysis import analyze

    with ResultStore(tmp_path, resume=False) as store:
        store.append(record("plot"))
    analyze(tmp_path)
    assert (tmp_path / "analysis/aggregated.csv").is_file()
    assert len(list((tmp_path / "analysis").glob("*.pdf"))) == 3
    assert len(list((tmp_path / "analysis").glob("*.png"))) == 3
