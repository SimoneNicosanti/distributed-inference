"""Descriptive plots of persisted paired observations; no significance claims."""

from collections import defaultdict
from pathlib import Path
from typing import Any

from scalability_benchmark.analysis import finite_number


def render(
    records: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    feasibility: list[dict[str, Any]],
    destination: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    def save(fig: Any, name: str) -> None:
        fig.tight_layout()
        fig.savefig(destination / f"{name}.png", dpi=160)
        plt.close(fig)

    valid = [pair for pair in pairs if pair["valid_pair"]]
    groups = defaultdict(list)
    for pair in valid:
        groups[pair["workload"], pair["flow_count"]].append(pair)
    fig, ax = plt.subplots()
    for (workload, flows), group in sorted(groups.items()):
        label = f"{workload}, F={flows}"
        line = sorted(
            [
                row
                for row in summaries
                if row["workload"] == workload
                and row["flow_count"] == flows
                and row["valid_pair_count"]
            ],
            key=lambda row: row["load"],
        )
        (plotted,) = ax.plot(
            [row["load"] for row in line],
            [row["median_improvement_pct"] for row in line],
            label=label,
        )
        for quality, marker in ((True, "o"), (False, "x")):
            points = [pair for pair in group if pair["tight_gaps"] == quality]
            ax.scatter(
                [p["load"] for p in points],
                [p["improvement_pct"] for p in points],
                color=plotted.get_color(),
                marker=marker,
            )
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set(
        xlabel="Offered reference load",
        ylabel="P=2 improvement (%)",
        title="Paired observations and medians; x = provisional",
    )
    if groups:
        ax.legend(fontsize="small")
    save(fig, "improvement_vs_load")

    fig, ax = plt.subplots()
    for (workload, flows), group in sorted(groups.items()):
        for priority in (1, 2):
            key = f"p{priority}_response_s"
            for load in sorted({p["load"] for p in group}):
                values = [p[key] for p in group if p["load"] == load]
                ax.scatter(
                    [load] * len(values),
                    values,
                    alpha=0.4,
                    marker="o" if priority == 1 else "x",
                )
            from statistics import median

            loads = sorted({p["load"] for p in group})
            ax.plot(
                loads,
                [median(p[key] for p in group if p["load"] == load) for load in loads],
                linestyle="-" if priority == 1 else "--",
                label=f"{workload}, F={flows}, P={priority}",
            )
    ax.set(xlabel="Offered reference load", ylabel="Arrival-weighted response time (s)")
    if groups:
        ax.legend(fontsize="x-small")
    save(fig, "response_vs_load")

    fig, ax = plt.subplots()
    for workload in sorted({p["workload"] for p in valid}):
        group = [p for p in valid if p["workload"] == workload]
        ax.scatter(
            [p["service_demand_cv"] for p in group],
            [p["improvement_pct"] for p in group],
            label=workload,
        )
    ax.set(xlabel="CV of reference service demands", ylabel="P=2 improvement (%)")
    ax.axhline(0, color="grey", linewidth=0.8)
    if valid:
        ax.legend()
    save(fig, "heterogeneity")

    valid_ids = {p["scenario_id"] for p in valid}
    paired_records = [r for r in records if r["scenario_id"] in valid_ids]
    fig, ax = plt.subplots(figsize=(10, 5))
    labels: list[str] = []
    for record in sorted(
        paired_records,
        key=lambda r: (
            r["workload"],
            r["flow_count"],
            r["load"],
            r["seed"],
            r["priority_count"],
        ),
    ):
        x = len(labels)
        labels.append(
            f"{record['workload']}\nF={record['flow_count']} L={record['load']}\nseed={record['seed']} P={record['priority_count']}"
        )
        bottom = 0
        for index, metric in enumerate(
            (
                "computation_s",
                "waiting_s",
                "transmission_s",
                "propagation_s",
                "batch_fill_s",
            )
        ):
            value = record["weighted_decomposition"][metric]
            ax.bar(
                x,
                value,
                bottom=bottom,
                color=f"C{index}",
                label=metric if x == 0 else None,
            )
            bottom += value
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6)
    ax.set_ylabel("Weighted response contributions (s)")
    if labels:
        ax.legend(fontsize="small")
    save(fig, "decomposition")

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    lookup = {(r["scenario_id"], r["priority_count"]): r for r in records}
    for ax, metric in zip(
        axes.flat,
        (
            "construction_time_s",
            "solver_runtime_s",
            "time_to_first_incumbent_s",
            "final_gap",
        ),
        strict=True,
    ):
        time_values = [
            (
                finite_number(lookup[p["scenario_id"], 1].get(metric)),
                finite_number(lookup[p["scenario_id"], 2].get(metric)),
            )
            for p in pairs
            if (p["scenario_id"], 1) in lookup and (p["scenario_id"], 2) in lookup
        ]
        timing_points = [
            (a, b) for a, b in time_values if a is not None and b is not None
        ]
        ax.scatter([a for a, _ in timing_points], [b for _, b in timing_points])
        maximum = max((max(a, b) for a, b in timing_points), default=1)
        ax.plot([0, maximum], [0, maximum], color="grey", linewidth=0.8)
        ax.set(xlabel="P=1", ylabel="P=2", title=metric)
    save(fig, "solver_overhead")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(feasibility)), [row["count"] for row in feasibility])
    ax.set_xticks(
        range(len(feasibility)),
        [f"P={row['priority_count']} {row['termination']}" for row in feasibility],
        rotation=35,
        ha="right",
        fontsize=8,
    )
    ax.set_ylabel("Attempts (including unsuccessful runs)")
    save(fig, "feasibility")
