"""Single-writer, fsynced JSONL journal; CSV is an atomic derived snapshot."""

import csv
import fcntl
import json
import os
from pathlib import Path
from types import TracebackType
from typing import Any, BinaryIO, Self

SUMMARY_FIELDS = (
    "run_id",
    "scenario_id",
    "campaign",
    "objective",
    "seed",
    "load",
    "flow_count",
    "model_count",
    "edge_count",
    "cloud_count",
    "worker_count",
    "timeout_s",
    "solver_runtime_s",
    "wall_time_s",
    "construction_time_s",
    "solver_status",
    "solver_status_name",
    "termination",
    "feasible_incumbent",
    "solution_extracted",
    "objective_value",
    "objective_bound",
    "final_gap",
    "timeout_reached",
    "infeasibility_proven",
    "error_type",
    "error_details",
)


def atomic_json(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_journal(path: Path, *, repair: bool = False) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return latest
    with path.open("r+b" if repair else "rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                # Only a final, uncommitted line may be discarded. Corruption in
                # the middle of the journal is always an error, never skipped.
                if repair:
                    handle.truncate(offset)
                    handle.flush()
                    os.fsync(handle.fileno())
                break
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Corrupt committed JSONL record at byte {offset}"
                ) from error
            if not isinstance(record, dict) or record.get("schema_version") != 1:
                raise ValueError(f"Unsupported journal record at byte {offset}")
            if not isinstance(record.get("run_id"), str) or record.get("state") not in {
                "started",
                "finished",
            }:
                raise ValueError(f"Invalid journal event at byte {offset}")
            latest[record["run_id"]] = record
    return latest


def write_summary(path: Path, latest: dict[str, dict[str, Any]]) -> None:
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore"
        )
        writer.writeheader()
        for record in latest.values():
            if record["state"] == "finished":
                writer.writerow(record)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


class ResultStore:
    def __init__(
        self, directory: Path, *, resume: bool, retry_errors: bool = False
    ) -> None:
        self.directory = directory
        self.resume = resume
        self.retry_errors = retry_errors
        self.latest: dict[str, dict[str, Any]] = {}
        self._lock: BinaryIO | None = None

    def __enter__(self) -> Self:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = (self.directory / ".writer.lock").open("a+b")
        try:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            path = self.directory / "runs.jsonl"
            if path.exists() and path.stat().st_size and not self.resume:
                raise FileExistsError(
                    f"{path} already contains runs; use --resume or another --output"
                )
            self.latest = read_journal(path, repair=True)
        except BaseException:
            self._lock.close()
            self._lock = None
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            write_summary(self.directory / "summary.csv", self.latest)
        finally:
            if self._lock is not None:
                self._lock.close()
                self._lock = None

    def completed(self, run_id: str) -> bool:
        previous = self.latest.get(run_id)
        if (
            previous is None
            or previous["state"] != "finished"
            or previous.get("termination") == "interrupted"
        ):
            return False
        return not (self.retry_errors and previous.get("termination") == "solver_error")

    def append(self, record: dict[str, object]) -> None:
        if self._lock is None:
            raise RuntimeError("Result store must be entered before writing")
        event = {**record, "schema_version": 1}
        payload = (json.dumps(event, sort_keys=True, allow_nan=False) + "\n").encode()
        with (self.directory / "runs.jsonl").open("ab") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        run_id = event["run_id"]
        assert isinstance(run_id, str)
        self.latest[run_id] = event
        if event["state"] == "finished":
            write_summary(self.directory / "summary.csv", self.latest)
