"""Offline descriptive analysis; timeouts remain in runtime distributions."""

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from scalability_benchmark.aggregate_time_plots import plot_aggregate_times
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


def record_objective(record: dict[str, Any]) -> str:
    requested = record.get("requested_solver_configuration") or {}
    return str(record.get("objective") or requested.get("objective") or "unspecified")


def time_to_first_incumbent(record: dict[str, Any]) -> float | None:
    value = finite_number(record.get("time_to_first_incumbent_s"))
    return value if value is not None and value >= 0 else None


def incumbent_points(record: dict[str, Any]) -> list[tuple[float, float | None]]:
    """Preserve missing samples so plotting cannot bridge unavailable values."""
    statistics = record.get("raw_solver_statistics")
    if not isinstance(statistics, dict):
        return []
    history = statistics.get("incumbent_history")
    if not isinstance(history, list):
        return []
    points = []
    for sample in history:
        if not isinstance(sample, dict):
            continue
        timestamp = finite_number(sample.get("runtime_s"))
        if timestamp is not None and timestamp >= 0:
            points.append((timestamp, finite_number(sample.get("objective_value"))))
    return points


def aggregate(records: list[dict[str, Any]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float, str, str, int], list[dict[str, Any]]] = defaultdict(
        list
    )
    for record in records:
        for dimension in DIMENSIONS:
            grouped[
                record["campaign"],
                record["load"],
                record_objective(record),
                dimension,
                record[dimension],
            ].append(record)
    rows: list[dict[str, object]] = []
    for (campaign, load, objective, dimension, size), group in sorted(grouped.items()):
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
        first_incumbents = [
            value
            for record in group
            if (value := time_to_first_incumbent(record)) is not None
        ]
        count = len(group)
        rows.append(
            {
                "campaign": campaign,
                "objective": objective,
                "load": load,
                "dimension": dimension,
                "size": size,
                "completed_count": count,
                "runtime_observed_count": len(runtimes),
                "gap_observed_count": len(gaps),
                "first_incumbent_observed_count": len(first_incumbents),
                "first_incumbent_median_s": (
                    median(first_incumbents) if first_incumbents else None
                ),
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
    campaigns = sorted(
        {
            (record["campaign"], record["load"], record_objective(record))
            for record in records
        }
    )
    for campaign, load, objective in campaigns:
        selected = [
            record
            for record in records
            if record["campaign"] == campaign
            and record["load"] == load
            and record_objective(record) == objective
        ]
        plot_runtime_progression(selected, rows, destination, campaign, load, objective)
        plot_aggregate_times(
            selected,
            destination,
            f"{campaign}-{objective}-load-{load}",
            f"{campaign}; objective={objective}; reference load={load}",
            DIMENSIONS,
        )
        # for record in selected:
        #     plot_incumbent_history(record, destination)
        for dimension, label in DIMENSIONS.items():
            subset = [
                row
                for row in rows
                if row["campaign"] == campaign
                and row["objective"] == objective
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
            figure.suptitle(
                f"{campaign}; objective={objective}; reference load={load}; other factors pooled"
            )
            for extension in ("pdf", "png"):
                figure.savefig(
                    destination
                    / f"{campaign}-{objective}-load-{load}-{dimension}.{extension}"
                )
            plt.close(figure)


def plot_first_incumbent(
    records: list[dict[str, Any]],
    rows: list[dict[str, object]],
    destination: Path,
    campaign: str,
    load: float,
    objective: str,
) -> None:
    observed = [
        record for record in records if time_to_first_incumbent(record) is not None
    ]
    if not observed:
        return
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for axis, (dimension, label) in zip(axes, DIMENSIONS.items(), strict=True):
        for timeout, marker, legend in (
            (False, "o", "Before timeout"),
            (True, "x", "Timeout with incumbent"),
        ):
            subset = [
                record for record in observed if record["timeout_reached"] == timeout
            ]
            axis.scatter(
                [record[dimension] for record in subset],
                [time_to_first_incumbent(record) for record in subset],
                marker=marker,
                label=legend,
                alpha=0.65,
            )
        summaries = [
            row
            for row in rows
            if row["campaign"] == campaign
            and row["load"] == load
            and row["objective"] == objective
            and row["dimension"] == dimension
            and row["first_incumbent_median_s"] is not None
        ]
        axis.plot(
            [row["size"] for row in summaries],
            [row["first_incumbent_median_s"] for row in summaries],
            color="black",
            marker="_",
            label="Observed median",
        )
        axis.set_xlabel(label)
        axis.set_ylabel("Time to first incumbent (s)")
        axis.set_xticks(sorted({record[dimension] for record in records}))
        axis.set_ylim(bottom=0)
        axis.legend(fontsize=8)
    figure.suptitle(
        f"{campaign}; objective={objective}; reference load={load}; "
        f"observed={len(observed)}/{len(records)}; other factors pooled"
    )
    for extension in ("pdf", "png"):
        figure.savefig(
            destination
            / f"{campaign}-{objective}-load-{load}-time-to-first-incumbent.{extension}"
        )
    plt.close(figure)


def plot_incumbent_history(record: dict[str, Any], destination: Path) -> None:
    points = incumbent_points(record)
    if not any(value is not None for _, value in points):
        return
    import matplotlib.pyplot as plt

    directory = destination / "incumbents"
    directory.mkdir(exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 4), constrained_layout=True)
    axis.step(
        [timestamp for timestamp, _ in points],
        [value if value is not None else math.nan for _, value in points],
        where="post",
        marker=".",
        markersize=3,
        label="Incumbent (1 s samples + termination)",
    )
    first = time_to_first_incumbent(record)
    if first is not None:
        axis.axvline(first, linestyle=":", color="black", label="First incumbent")
    axis.set_xlabel("Solver runtime (s)")
    axis.set_ylabel("Objective value")
    axis.set_xlim(left=0)
    axis.legend(fontsize=8)
    axis.set_title(
        f"{record['campaign']}; objective={record_objective(record)}; load={record['load']}\n"
        f"workers={record['worker_count']}; flows={record['flow_count']}; "
        f"models={record['model_count']}; seed={record['seed']}\n"
        f"{record['run_id']}"
    )
    for extension in ("pdf", "png"):
        figure.savefig(directory / f"{record['run_id']}.{extension}")
    plt.close(figure)


def plot_runtime_progression(
    records: list[dict[str, Any]],
    rows: list[dict[str, object]],
    destination: Path,
    campaign: str,
    load: float,
    objective: str,
) -> None:
    """Show runtime scaling with pooled observed medians and explicit timeouts."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    observed = [
        record
        for record in records
        if record["termination"] != "solver_error"
        and finite_number(record.get("solver_runtime_s")) is not None
    ]
    for axis, (dimension, label) in zip(axes, DIMENSIONS.items(), strict=True):
        for timeout, marker, color, legend in (
            (False, "o", "tab:blue", "Before timeout"),
            (True, "x", "tab:orange", "Timeout"),
        ):
            subset = [
                record for record in observed if record["timeout_reached"] == timeout
            ]
            axis.scatter(
                [record[dimension] for record in subset],
                [record["solver_runtime_s"] for record in subset],
                marker=marker,
                color=color,
                alpha=0.4,
                label=legend,
            )
        summaries = [
            row
            for row in rows
            if row["campaign"] == campaign
            and row["objective"] == objective
            and row["load"] == load
            and row["dimension"] == dimension
        ]
        axis.plot(
            [row["size"] for row in summaries],
            [
                row["runtime_median_s"]
                if row["runtime_median_s"] is not None
                else math.nan
                for row in summaries
            ],
            color="black",
            marker="o",
            linewidth=2,
            label="Observed median (including timeouts)",
        )
        axis.set_xlabel(label)
        axis.set_ylabel("Solver runtime (s)")
        axis.set_xticks(sorted({record[dimension] for record in records}))
        axis.set_ylim(bottom=0)
        axis.legend(fontsize=8)
    figure.suptitle(
        f"{campaign}; objective={objective}; reference load={load}; "
        f"runtimes observed={len(observed)}/{len(records)}; other factors pooled"
    )
    for extension in ("pdf", "png"):
        figure.savefig(
            destination
            / f"{campaign}-{objective}-load-{load}-runtime-progression.{extension}"
        )
    plt.close(figure)
