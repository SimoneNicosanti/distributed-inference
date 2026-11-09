"""Descriptive time distributions; missing observations are never imputed."""

import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import median
from typing import Any

TIME_METRICS = {
    "time_to_first_incumbent_s": "Time to first incumbent (s)",
    "solver_runtime_s": "Observed solver runtime (s)",
}


@dataclass(frozen=True)
class TimeDistribution:
    total: int
    non_timeouts: tuple[float, ...]
    timeouts: tuple[float, ...]

    @property
    def values(self) -> tuple[float, ...]:
        return self.non_timeouts + self.timeouts

    @property
    def median_s(self) -> float | None:
        return median(self.values) if self.values else None


def summarize_times(records: list[dict[str, Any]], metric: str) -> TimeDistribution:
    if metric not in TIME_METRICS:
        raise ValueError(f"Unknown time metric: {metric}")
    non_timeouts = []
    timeouts = []
    for record in records:
        if metric == "solver_runtime_s" and record["termination"] == "solver_error":
            continue
        value = record.get(metric)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
        ):
            continue
        target = timeouts if record["timeout_reached"] else non_timeouts
        target.append(float(value))
    return TimeDistribution(len(records), tuple(non_timeouts), tuple(timeouts))


def empirical_cdf(values: tuple[float, ...]) -> tuple[list[float], list[float]]:
    if not values:
        return [], []
    ordered = sorted(values)
    return [0.0, *ordered], [
        0.0,
        *(index / len(ordered) for index in range(1, len(ordered) + 1)),
    ]


def grouped_times(
    records: list[dict[str, Any]], metric: str, dimensions: tuple[str, ...]
) -> dict[tuple[int, ...], TimeDistribution]:
    groups: dict[tuple[int, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[tuple(record[dimension] for dimension in dimensions)].append(record)
    return {
        key: summarize_times(group, metric) for key, group in sorted(groups.items())
    }


def plot_aggregate_times(
    records: list[dict[str, Any]],
    destination: Path,
    prefix: str,
    title: str,
    dimensions: dict[str, str],
) -> None:
    """Called for one campaign/objective/load; all other factors are pooled."""
    import matplotlib.pyplot as plt

    for kind, draw in (
        ("time-boxplots", _boxplots),
        ("time-cdf", _cdf),
        ("time-heatmaps", _heatmaps),
    ):
        figure = draw(records, dimensions)
        figure.suptitle(f"{title}; other factors pooled")
        for extension in ("pdf", "png"):
            figure.savefig(destination / f"{prefix}-{kind}.{extension}")
        plt.close(figure)


def _boxplots(records: list[dict[str, Any]], dimensions: dict[str, str]) -> Any:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for row, (metric, label) in zip(axes, TIME_METRICS.items(), strict=True):
        for axis, (dimension, dimension_label) in zip(
            row, dimensions.items(), strict=True
        ):
            groups = grouped_times(records, metric, (dimension,))
            observed = [
                (index, group)
                for index, group in enumerate(groups.values())
                if group.values
            ]
            if observed:
                axis.boxplot(
                    [group.values for _, group in observed],
                    positions=[index for index, _ in observed],
                    widths=0.5,
                    manage_ticks=False,
                )
            else:
                axis.text(
                    0.5, 0.5, "No observations", transform=axis.transAxes, ha="center"
                )
            for index, group in enumerate(groups.values()):
                axis.text(
                    index,
                    0.98,
                    f"{len(group.values)}/{group.total}",
                    transform=axis.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=7,
                )
                if group.timeouts:
                    axis.scatter(
                        [index] * len(group.timeouts),
                        group.timeouts,
                        marker="x",
                        color="tab:orange",
                        alpha=0.5,
                    )
            axis.set_xticks(list(range(len(groups))), [str(key[0]) for key in groups])
            axis.set_xlabel(dimension_label)
            axis.set_ylabel(label)
            axis.set_title(
                "Observed/total above; orange crosses = timeouts", fontsize=9
            )
            axis.set_ylim(bottom=0)
    return figure


def _cdf(records: list[dict[str, Any]], dimensions: dict[str, str]) -> Any:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    for axis, (metric, label) in zip(axes, TIME_METRICS.items(), strict=True):
        group = summarize_times(records, metric)
        for values, color, legend in (
            (group.values, "black", "All observations"),
            (group.non_timeouts, "tab:blue", "Before timeout"),
            (group.timeouts, "tab:orange", "Timeout runs"),
        ):
            if values:
                x, y = empirical_cdf(values)
                axis.step(
                    x, y, where="post", color=color, label=f"{legend} (n={len(values)})"
                )
        if not group.values:
            axis.text(
                0.5, 0.5, "No observations", transform=axis.transAxes, ha="center"
            )
        else:
            axis.legend(fontsize=8)
        axis.set_xlabel(label)
        axis.set_ylabel("Fraction within observed group")
        axis.set_xlim(left=0)
        axis.set_ylim(0, 1.02)
        axis.set_title(
            f"Observed {len(group.values)}/{group.total}; missing {group.total - len(group.values)}"
        )
    figure.supxlabel(
        "Conditional empirical distributions; timeout runtimes are not times to proven optimality",
        fontsize=9,
    )
    return figure


def _heatmaps(records: list[dict[str, Any]], dimensions: dict[str, str]) -> Any:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    pairs = list(combinations(dimensions, 2))
    for row, (metric, label) in zip(axes, TIME_METRICS.items(), strict=True):
        distribution = summarize_times(records, metric)
        maximum = max(distribution.values, default=1.0) or 1.0
        for axis, (horizontal, vertical) in zip(row, pairs, strict=True):
            groups = grouped_times(records, metric, (horizontal, vertical))
            xs = sorted({record[horizontal] for record in records})
            ys = sorted({record[vertical] for record in records})
            cells = []
            for y in ys:
                cells.append(
                    [
                        groups[(x, y)].median_s
                        if (x, y) in groups and groups[(x, y)].median_s is not None
                        else math.nan
                        for x in xs
                    ]
                )
            image = axis.imshow(
                cells,
                origin="lower",
                aspect="auto",
                vmin=0,
                vmax=maximum,
                cmap="viridis",
            )
            for row_index, y in enumerate(ys):
                for column_index, x in enumerate(xs):
                    group = groups.get((x, y))
                    if group is None:
                        text = "—"
                    elif group.median_s is None:
                        text = f"NA\n0/{group.total}"
                    else:
                        text = (
                            f"{group.median_s:.2g}\n{len(group.values)}/{group.total}"
                        )
                    axis.text(
                        column_index,
                        row_index,
                        text,
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="black"
                        if group is None or group.median_s is None
                        else "white",
                        bbox={"facecolor": "black", "alpha": 0.25, "edgecolor": "none"}
                        if group is not None and group.median_s is not None
                        else None,
                    )
            axis.set_xticks(list(range(len(xs))), [str(x) for x in xs])
            axis.set_yticks(list(range(len(ys))), [str(y) for y in ys])
            axis.set_xlabel(dimensions[horizontal])
            axis.set_ylabel(dimensions[vertical])
            axis.set_title("Median seconds; observed/total", fontsize=9)
            axis.grid(False)
            figure.colorbar(image, ax=axis, label=label)
    return figure
