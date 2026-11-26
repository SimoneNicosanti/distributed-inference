"""Conditional diagnostics: timeouts are censored, never exact solve times."""

import csv
import json
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from statistics import median
from typing import Any

from scalability_benchmark.analysis import finite_number
from scalability_benchmark.persistence import atomic_json, read_journal
from scalability_benchmark.randomness import fingerprint


def numbers(rows: list[dict[str, Any]], name: str) -> list[float]:
    return [
        value for row in rows if (value := finite_number(row.get(name))) is not None
    ]


def med(rows: list[dict[str, Any]], name: str) -> float | None:
    values = numbers(rows, name)
    return median(values) if values else None


def quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (position - low) * (ordered[high] - ordered[low])


def family(row: dict[str, Any]) -> dict[str, Any]:
    axis = row["sweep"]
    return {
        "reference_policy": row.get("reference_policy", "legacy_full_a_reference"),
        **{
            key: row[key]
            for key in (
                "manifest_id",
                "objective",
                "formulation",
                "start_mode",
                "sweep",
                "load",
                "flow_count",
                "model_count",
                "worker_count",
                "max_components",
                "priority_count",
            )
            if key
            != {
                "flows": "flow_count",
                "models": "model_count",
                "workers": "worker_count",
                "components": "max_components",
                "priorities": "priority_count",
            }[axis]
        },
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(fingerprint(family(row)), row["ablation"], row["sweep_value"])].append(
            row
        )
    result = []
    for (family_id, ablation, value), group in sorted(buckets.items()):
        available = [r for r in group if r.get("available")]
        solved = [r for r in available if r.get("solved_to_tolerance")]
        times = numbers(solved, "solver_runtime_s")
        gaps = numbers(available, "final_gap")
        presolved = [
            {
                "binary": (r.get("presolved_model_dimensions") or {}).get(
                    "binary_variables"
                ),
                "free_integer": (r.get("presolved_model_dimensions") or {}).get(
                    "free_integer_variables"
                ),
            }
            for r in available
        ]
        result.append(
            {
                **family(group[0]),
                "family_id": family_id,
                "ablation": ablation,
                "sweep_value": value,
                "n": len(group),
                "n_available": len(available),
                "n_solved": len(solved),
                "n_censored": sum(bool(r.get("runtime_censored")) for r in available),
                "availability_rate": len(available) / len(group),
                "success_rate": len(solved) / len(group),
                "timeout_rate": sum(bool(r.get("timeout_reached")) for r in group)
                / len(group),
                "feasible_rate": sum(bool(r.get("feasible_incumbent")) for r in group)
                / len(group),
                "strict_feasible_rate": sum(
                    bool((r.get("native_feasibility") or {}).get("validated"))
                    for r in group
                )
                / len(group),
                "zero_tardiness_rate": sum(
                    bool(r.get("zero_tardiness_found")) for r in group
                )
                / len(group),
                "solved_runtime_median_s": median(times) if times else None,
                "solved_runtime_q25_s": quantile(times, 0.25),
                "solved_runtime_q75_s": quantile(times, 0.75),
                "observed_runtime_median_s": med(available, "solver_runtime_s"),
                "first_incumbent_median_s": med(available, "time_to_first_incumbent_s"),
                "n_first_incumbent_observed": len(
                    numbers(available, "time_to_first_incumbent_s")
                ),
                "zero_tardiness_median_s": med(available, "time_to_zero_tardiness_s"),
                "n_zero_tardiness_time_observed": len(
                    numbers(available, "time_to_zero_tardiness_s")
                ),
                "gap_median": median(gaps) if gaps else None,
                "gap_q25": quantile(gaps, 0.25),
                "gap_q75": quantile(gaps, 0.75),
                "n_gap_observed": len(gaps),
                "objective_median": med(available, "objective_value"),
                "bound_median": med(available, "objective_bound"),
                "presolved_binary_median": med(presolved, "binary"),
                "presolved_free_integer_median": med(presolved, "free_integer"),
                "root_median_s": med(available, "root_relaxation_time_s"),
                "presolve_median_s": med(available, "solver_presolve_time_s"),
            }
        )
    return result


def transitions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    paired: dict[Any, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        # Same original problem, reference, mode, study and sweep point only.
        key = (
            row.get("reference_policy", "legacy_full_a_reference"),
            row["manifest_id"],
            row["scenario_id"],
            row["reference_id"],
            row["start_mode"],
            row["sweep"],
            row["sweep_value"],
        )
        paired[key][row["ablation"]] = row
    results = []
    for group in paired.values():
        order = (
            ("A", "B", "C", "D", "P", "E")
            if "P" in group
            else ("A", "B", "C", "D", "E")
        )
        for left, right in pairwise(order):
            if left not in group or right not in group:
                continue
            a, b = group[left], group[right]
            ta, tb = (
                finite_number(a.get("solver_runtime_s")),
                finite_number(b.get("solver_runtime_s")),
            )
            comparable = bool(
                a.get("solved_to_tolerance")
                and b.get("solved_to_tolerance")
                and ta is not None
                and tb is not None
                and tb > 0
            )
            results.append(
                {
                    **family(a),
                    "family_id": fingerprint(family(a)),
                    "seed": a["seed"],
                    "sweep_value": a["sweep_value"],
                    "transition": f"{left}->{right}",
                    "scenario_id": a["scenario_id"],
                    "reference_id": a["reference_id"],
                    "left_termination": a["termination"],
                    "right_termination": b["termination"],
                    "both_solved": comparable,
                    "left_solved": bool(a.get("solved_to_tolerance")),
                    "right_solved": bool(b.get("solved_to_tolerance")),
                    "runtime_ratio_to_tolerance": ta / tb
                    if comparable and ta is not None and tb is not None
                    else None,
                    "runtime_difference_s": ta - tb
                    if comparable and ta is not None and tb is not None
                    else None,
                    "objective_difference": a["objective_value"] - b["objective_value"]
                    if comparable
                    else None,
                }
            )
    return results


def thresholds(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in groups:
        if row["sweep"] == "flows":
            buckets[(row["family_id"], row["ablation"])].append(row)
    result = []
    for (family_id, ablation), values in buckets.items():
        previous = None
        for row in sorted(values, key=lambda r: r["sweep_value"]):
            reason = None
            if row["n_available"] and row["timeout_rate"] >= 0.5:
                reason = "timeout_rate_at_least_half"
            elif (
                previous
                and previous["solved_runtime_median_s"]
                and row["solved_runtime_median_s"] is not None
                and row["solved_runtime_median_s"]
                >= 5 * previous["solved_runtime_median_s"]
            ):
                reason = "solved_subset_median_at_least_5x_previous_tested_F"
            if reason:
                result.append(
                    {
                        "family_id": family_id,
                        "ablation": ablation,
                        "earliest_tested_F": row["sweep_value"],
                        "reason": reason,
                        "n_at_threshold": row["n"],
                        "preliminary": row["n_available"] < 2,
                    }
                )
                break
            previous = row
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def summary_text(
    groups: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    detected: list[dict[str, Any]],
) -> str:
    lines = [
        "# Bottleneck diagnostics",
        "",
        "Conditional ablations; runtimes are not additive decision costs. Restricted objectives may differ.",
        "Solve-time medians include only runs solved to the configured tolerance. Timeouts are right-censored; their durations are reported separately.",
        "Missing root/presolve/callback measurements are unavailable, not zero. E fixes discretization auxiliaries as well as priorities; use P to separate them.",
        "Warm-start preparation costs are separate from A runtime; cold B is measured once and reused as the reference source under the predefined-model protocol.",
        "Reference protocols: "
        + ", ".join(sorted({r["reference_policy"] for r in groups})),
        "",
    ]
    ratios: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for item in comparisons:
        if item["both_solved"]:
            ratios[(item["family_id"], item["transition"], item["sweep_value"])].append(
                item["runtime_ratio_to_tolerance"]
            )
    if ratios:
        best, values = max(ratios.items(), key=lambda item: median(item[1]))
        label = (
            "largest measured conditional reduction"
            if median(values) > 1
            else "no median conditional reduction observed"
        )
        lines.append(
            f"{label}: {best[1]}, tested axis value {best[2]}, median runtime ratio {median(values):.3g} ({len(values)} paired seeds; family {best[0][:12]})."
        )
        selected_pairs = [
            item
            for item in comparisons
            if item["both_solved"]
            and (item["family_id"], item["transition"], item["sweep_value"]) == best
        ]
        difference = med(selected_pairs, "objective_difference")
        if difference is not None:
            lines.append(
                f"For those pairs, median final objective difference (left minus right) is {difference:+.6g}; runtime reduction does not imply equal solution quality."
            )
    else:
        lines.append("No paired completed solves: no runtime ratio can be estimated.")
    for item in detected:
        lines.append(
            f"Earliest tested difficulty flag: {item['ablation']} at F={item['earliest_tested_F']}: {item['reason']} (n={item['n_at_threshold']}; family {item['family_id'][:12]})."
        )
    if not detected:
        lines.append(
            "No tested point meets the declared difficulty rules (timeout fraction >= 0.5, or solved-subset median >= 5x previous tested F). This does not establish scalability beyond tested sizes."
        )
    if groups and min(r["n"] for r in groups) < 2:
        lines.append(
            "Single-seed groups are preliminary; seed dispersion cannot be estimated."
        )
    lines.extend(
        [
            "",
            "## Measurements",
            "",
            "|Family|Axis|Value|Mode|Ablation|Available|Solved|Timeouts|Median solved s|First incumbent s|Gap median|Objective median|Bound median|",
            "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in groups:

        def fmt(value: float | None) -> str:
            return "NA" if value is None else f"{value:.4g}"

        lines.append(
            f"|{row['family_id'][:12]}|{row['sweep']}|{row['sweep_value']}|{row['start_mode']}|{row['ablation']}|{row['n_available']}/{row['n']}|{row['n_solved']}|{row['timeout_rate']:.0%}|{fmt(row['solved_runtime_median_s'])}|{fmt(row['first_incumbent_median_s'])}|{fmt(row['gap_median'])}|{fmt(row['objective_median'])}|{fmt(row['bound_median'])}|"
        )
    return "\n".join(lines) + "\n"


def analyze(directory: Path) -> None:
    rows = [
        r
        for r in read_journal(directory / "runs.jsonl").values()
        if r["state"] == "finished"
    ]
    destination = directory / "analysis"
    destination.mkdir(parents=True, exist_ok=True)
    groups, comparisons = aggregate(rows), transitions(rows)
    detected = thresholds(groups)
    write_csv(destination / "runs.csv", rows)
    write_csv(destination / "aggregate.csv", groups)
    write_csv(destination / "transitions.csv", comparisons)
    atomic_json(
        destination / "thresholds.json",
        {
            "rules": {
                "timeout_fraction": 0.5,
                "previous_tested_solved_median_ratio": 5,
            },
            "thresholds": detected,
        },
    )
    (destination / "summary.md").write_text(summary_text(groups, comparisons, detected))
    from scalability_benchmark.bottleneck.plots import plots

    plots(rows, groups, comparisons, destination / "plots")
