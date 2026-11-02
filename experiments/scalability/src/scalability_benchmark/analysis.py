"""Offline descriptive analysis; timeouts remain in runtime distributions."""

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from scalability_benchmark.persistence import read_journal, write_summary

DIMENSIONS = {
    "worker_count": "Workers",
    "flow_count": "Inference flows",
    "model_count": "Candidate models",
}


def finite_number(value: object) -> float | None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        return None
    return float(value)


def aggregate(records: list[dict[str, Any]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float, str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for dimension in DIMENSIONS:
            grouped[
                record["campaign"], record["load"], dimension, record[dimension]
            ].append(record)
    rows: list[dict[str, object]] = []
    for (campaign, load, dimension, size), group in sorted(grouped.items()):
        runtimes = [
            value
            for record in group
            if record["termination"] != "solver_error"
            if (value := finite_number(record.get("solver_runtime_s"))) is not None
        ]
        gaps = [
            value
            for record in group
            if record["feasible_incumbent"]
            if (value := finite_number(record.get("final_gap"))) is not None
        ]
        count = len(group)
        rows.append(
            {
                "campaign": campaign,
                "load": load,
                "dimension": dimension,
                "size": size,
                "completed_count": count,
                "runtime_observed_count": len(runtimes),
                "gap_observed_count": len(gaps),
                "runtime_median_s": median(runtimes) if runtimes else None,
                "runtime_mean_s": sum(runtimes) / len(runtimes) if runtimes else None,
                "runtime_min_s": min(runtimes) if runtimes else None,
                "runtime_max_s": max(runtimes) if runtimes else None,
                "incumbent_count": sum(
                    bool(record["feasible_incumbent"]) for record in group
                ),
                "feasible_discovery_rate": sum(
                    bool(record["feasible_incumbent"]) for record in group
                )
                / count,
                "timeout_rate": sum(bool(record["timeout_reached"]) for record in group)
                / count,
                "proven_infeasibility_rate": sum(
                    bool(record["infeasibility_proven"]) for record in group
                )
                / count,
                "error_rate": sum(
                    record["termination"] == "solver_error" for record in group
                )
                / count,
                "gap_median": median(gaps) if gaps else None,
            }
        )
    return rows


def analyze(directory: Path) -> None:
    latest = read_journal(directory / "runs.jsonl")
    records = [record for record in latest.values() if record["state"] == "finished"]
    if not records:
        raise ValueError(f"No completed runs in {directory}")
    destination = directory / "analysis"
    destination.mkdir(parents=True, exist_ok=True)
    write_summary(destination / "runs.csv", latest)
    rows = aggregate(records)
    with (destination / "aggregated.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Import only for analysis: scenario construction and solves do not need plotting.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 10, "axes.grid": True, "grid.alpha": 0.25, "savefig.dpi": 200}
    )
    campaigns = sorted({(record["campaign"], record["load"]) for record in records})
    for campaign, load in campaigns:
        selected = [
            record
            for record in records
            if record["campaign"] == campaign and record["load"] == load
        ]
        for dimension, label in DIMENSIONS.items():
            subset = [
                row
                for row in rows
                if row["campaign"] == campaign
                and row["load"] == load
                and row["dimension"] == dimension
            ]
            figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
            for timeout, marker, legend in (
                (False, "o", "Before timeout"),
                (True, "x", "Timeout"),
            ):
                observed = [
                    record
                    for record in selected
                    if record["timeout_reached"] == timeout
                    and record["termination"] != "solver_error"
                    and finite_number(record.get("solver_runtime_s")) is not None
                ]
                axes[0].scatter(
                    [record[dimension] for record in observed],
                    [record["solver_runtime_s"] for record in observed],
                    marker=marker,
                    label=legend,
                    alpha=0.65,
                )
            axes[0].set_ylabel("Solver runtime (s)")
            for metric, legend in (
                ("feasible_discovery_rate", "Feasible incumbent"),
                ("timeout_rate", "Timeout"),
                ("proven_infeasibility_rate", "Proven infeasible"),
                ("error_rate", "Error"),
            ):
                axes[1].plot(
                    [row["size"] for row in subset],
                    [row[metric] for row in subset],
                    marker="o",
                    label=legend,
                )
            axes[1].set_ylabel("Fraction of completed attempts")
            axes[1].set_ylim(-0.02, 1.02)
            valid_gaps = [
                record
                for record in selected
                if record["feasible_incumbent"]
                and finite_number(record.get("final_gap")) is not None
            ]
            axes[2].scatter(
                [record[dimension] for record in valid_gaps],
                [100 * record["final_gap"] for record in valid_gaps],
                alpha=0.65,
                label="Available incumbent gaps",
            )
            axes[2].set_ylabel("Final relative gap (%)")
            for axis in axes:
                axis.set_xlabel(label)
                axis.set_xticks(sorted({record[dimension] for record in selected}))
                axis.legend(fontsize=8)
            figure.suptitle(f"{campaign}; reference load={load}; other factors pooled")
            for extension in ("pdf", "png"):
                figure.savefig(
                    destination / f"{campaign}-load-{load}-{dimension}.{extension}"
                )
            plt.close(figure)
