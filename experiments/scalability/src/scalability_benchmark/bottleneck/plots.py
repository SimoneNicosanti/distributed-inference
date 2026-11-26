"""Eight diagnostic figure families, exported as PNG/PDF like the main study."""

from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from scalability_benchmark.bottleneck.analysis import family
from scalability_benchmark.randomness import fingerprint


def plots(
    rows: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    destination: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.mkdir(parents=True, exist_ok=True)
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in aggregates:
        families[row["family_id"]].append(row)
    metrics = (
        (
            "runtime",
            "solved_runtime_median_s",
            "Time to configured tolerance, solved subset (s)",
        ),
        (
            "first-incumbent",
            "first_incumbent_median_s",
            "First incumbent, observed subset (s)",
        ),
        ("success", "success_rate", "Fraction solved to configured tolerance"),
        ("gap", "gap_median", "Final relative gap, feasible subset"),
        (
            "presolved-binaries",
            "presolved_binary_median",
            "Presolve-probe binary count",
        ),
        ("root-time", "root_median_s", "First root relaxation time, log (s)"),
    )

    def save(figure: Any, name: str) -> None:
        for extension in ("png", "pdf"):
            figure.savefig(destination / f"{name}.{extension}")
        plt.close(figure)

    for family_id, group in families.items():
        first = group[0]
        title = f"{first['objective']} / {first['start_mode']} / {first['sweep']} / {family_id[:10]}"
        for name, metric, label in metrics:
            figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
            for ablation in sorted({r["ablation"] for r in group}):
                selected = sorted(
                    [r for r in group if r["ablation"] == ablation],
                    key=lambda r: r["sweep_value"],
                )
                axis.plot(
                    [r["sweep_value"] for r in selected],
                    [
                        r[metric] if r[metric] is not None else float("nan")
                        for r in selected
                    ],
                    marker="o",
                    label=ablation,
                )
                if name in ("runtime", "gap"):
                    low, high = (
                        ("solved_runtime_q25_s", "solved_runtime_q75_s")
                        if name == "runtime"
                        else ("gap_q25", "gap_q75")
                    )
                    observed = [r for r in selected if r[low] is not None]
                    axis.fill_between(
                        [r["sweep_value"] for r in observed],
                        [r[low] for r in observed],
                        [r[high] for r in observed],
                        alpha=0.12,
                    )
                if name == "runtime":
                    censored = [
                        r
                        for r in rows
                        if r.get("runtime_censored")
                        and r["ablation"] == ablation
                        and r["start_mode"] == first["start_mode"]
                        and r["manifest_id"] == first["manifest_id"]
                        and r["objective"] == first["objective"]
                        and r["sweep"] == first["sweep"]
                    ]
                    # family() also excludes cross-configuration pooling.
                    censored = [
                        r for r in censored if fingerprint(family(r)) == family_id
                    ]
                    axis.scatter(
                        [r["sweep_value"] for r in censored],
                        [r["solver_runtime_s"] for r in censored],
                        marker="^",
                        facecolors="none",
                        edgecolors="black",
                        label=f"{ablation} censored" if censored else None,
                    )
            axis.set(xlabel=first["sweep"], ylabel=label, title=title)
            axis.legend()
            if not any(r[metric] is not None for r in group):
                axis.text(
                    0.5,
                    0.5,
                    "Measurement unavailable",
                    transform=axis.transAxes,
                    ha="center",
                )
            save(figure, f"{family_id[:12]}-{name}")
        figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
        selected = [r for r in comparisons if r["family_id"] == family_id]
        for transition in sorted({r["transition"] for r in selected}):
            ratio_values: dict[int, list[float]] = defaultdict(list)
            for row in selected:
                if row["transition"] == transition and row["both_solved"]:
                    ratio_values[row["sweep_value"]].append(
                        row["runtime_ratio_to_tolerance"]
                    )
            axis.plot(
                sorted(ratio_values),
                [median(ratio_values[x]) for x in sorted(ratio_values)],
                marker="o",
                label=transition,
            )
        axis.axhline(1, color="grey", linestyle="--")
        axis.set(
            xlabel=first["sweep"],
            ylabel="Conditional runtime ratio, both solved only",
            title=title,
        )
        if any(r["both_solved"] for r in selected):
            axis.legend()
        else:
            axis.text(
                0.5,
                0.5,
                "No completed pairs; ratios unavailable",
                transform=axis.transAxes,
                ha="center",
            )
        save(figure, f"{family_id[:12]}-transitions")
        # Representative: largest observed native duration, then its paired ablations.
        candidates = [
            r
            for r in rows
            if r.get("available")
            and r.get("solver_runtime_s") is not None
            and fingerprint(family(r)) == family_id
        ]
        figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
        if candidates:
            representative = max(candidates, key=lambda r: r["solver_runtime_s"])
            subset = sorted(
                [
                    r
                    for r in candidates
                    if r["scenario_id"] == representative["scenario_id"]
                ],
                key=lambda r: r["ablation"],
            )
            labels = [
                r["ablation"] + (" censored" if r.get("runtime_censored") else "")
                for r in subset
            ]
            bottoms = [0.0] * len(subset)
            for metric, label in (
                ("solver_presolve_time_s", "Log presolve"),
                ("root_relaxation_time_s", "First root relaxation"),
                ("remaining_native_runtime_s", "Remaining native runtime"),
            ):
                values = [
                    float(r[metric])
                    if all(
                        r.get(k) is not None
                        for k in (
                            "solver_presolve_time_s",
                            "root_relaxation_time_s",
                            "remaining_native_runtime_s",
                        )
                    )
                    else 0
                    for r in subset
                ]
                axis.bar(labels, values, bottom=bottoms, label=label)
                bottoms = [a + b for a, b in zip(bottoms, values, strict=True)]
            missing = [
                r if r.get("remaining_native_runtime_s") is None else None
                for r in subset
            ]
            axis.bar(
                labels,
                [r["solver_runtime_s"] if r else 0 for r in missing],
                color="none",
                edgecolor="grey",
                hatch="//",
                label="Total only; breakdown unavailable",
            )
            axis.legend()
            axis.set_title(
                f"{title}\nRepresentative F={representative['flow_count']}, seed={representative['seed']}"
            )
        else:
            axis.text(
                0.5,
                0.5,
                "No native solve measurements",
                transform=axis.transAxes,
                ha="center",
            )
        axis.set_ylabel("Native runtime (s)")
        save(figure, f"{family_id[:12]}-breakdown")
