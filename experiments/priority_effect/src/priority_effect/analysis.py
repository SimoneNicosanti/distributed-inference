"""Offline paired comparisons. Missing objectives never become zero observations."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from scalability_benchmark.analysis import finite_number
from scalability_benchmark.persistence import atomic_json, read_journal


def _strategy_name(record: dict[str, Any]) -> str:
    name = record.get("strategy") or f"full_p{record['priority_count']}"
    if name not in {"full_p1", "full_p2"}:
        raise ValueError(f"Unsupported priority experiment strategy: {name}")
    return str(name)


def usable(record: dict[str, Any]) -> bool:
    return bool(
        record.get("state") == "finished"
        and record.get("feasible_incumbent")
        and record.get("solution_extracted")
        and finite_number(record.get("objective_value")) is not None
        and finite_number(record.get("arrival_weighted_mean_response_time_s"))
        is not None
    )


def difference(left: dict[str, Any], right: dict[str, Any], name: str) -> float | None:
    a, b = finite_number(left.get(name)), finite_number(right.get(name))
    return b - a if a is not None and b is not None else None


def compare(
    one: dict[str, Any], two: dict[str, Any], tight_gap: float
) -> dict[str, Any]:
    if (
        one["scenario_id"] != two["scenario_id"]
        or one["priority_count"] != 1
        or two["priority_count"] != 2
    ):
        raise ValueError("Comparison requires the same scenario with P=1 then P=2")
    row = {
        name: one.get(name)
        for name in (
            "scenario_id",
            "seed",
            "load",
            "workload",
            "category",
            "flow_count",
            "service_demand_cv",
        )
    }
    row.update(
        {
            "valid_pair": usable(one) and usable(two),
            "p1_termination": one.get("termination"),
            "p2_termination": two.get("termination"),
        }
    )
    if not row["valid_pair"]:
        return row
    first, second = (
        one["arrival_weighted_mean_response_time_s"],
        two["arrival_weighted_mean_response_time_s"],
    )
    row.update(
        {
            "absolute_response_reduction_s": first - second,
            "improvement_pct": 100 * (first - second) / first if first > 0 else None,
            "objective_improvement_pct": 100
            * (one["objective_value"] - two["objective_value"])
            / one["objective_value"]
            if one["objective_value"] > 0
            else None,
            "p1_response_s": first,
            "p2_response_s": second,
            "p1_objective": one["objective_value"],
            "p2_objective": two["objective_value"],
            "p1_bound": one.get("objective_bound"),
            "p2_bound": two.get("objective_bound"),
            "p1_gap": one.get("final_gap"),
            "p2_gap": two.get("final_gap"),
        }
    )
    for metric in (
        "solver_runtime_s",
        "construction_time_s",
        "time_to_first_incumbent_s",
        "final_gap",
    ):
        row[f"delta_{metric}"] = difference(one, two, metric)
    tight = all(
        (gap := finite_number(record.get("final_gap"))) is not None
        and 0 <= gap <= tight_gap
        for record in (one, two)
    )
    row["tight_gaps"] = tight
    lower_one, lower_two = (
        finite_number(one.get("objective_bound")),
        finite_number(two.get("objective_bound")),
    )
    valid_bounds = all(
        record.get("solver_status_name")
        in {
            "OPTIMAL",
            "TIME_LIMIT",
            "NODE_LIMIT",
            "WORK_LIMIT",
            "SOLUTION_LIMIT",
            "INTERRUPTED",
            "USER_OBJ_LIMIT",
        }
        for record in (one, two)
    )
    valid_bounds = (
        valid_bounds
        and lower_one is not None
        and lower_two is not None
        and 0 <= lower_one <= one["objective_value"] + 1e-8
        and 0 <= lower_two <= two["objective_value"] + 1e-8
    )
    certified = False
    if valid_bounds and lower_one is not None and lower_two is not None:
        row["objective_reduction_lower_bound"] = lower_one - two["objective_value"]
        row["objective_reduction_upper_bound"] = one["objective_value"] - lower_two
        row["improvement_lower_bound_pct"] = (
            100 * (1 - two["objective_value"] / lower_one) if lower_one > 0 else None
        )
        row["improvement_upper_bound_pct"] = (
            100 * (1 - lower_two / one["objective_value"])
            if one["objective_value"] > 0
            else None
        )
        certified = lower_one > two["objective_value"] + 1e-8
    row["certified_improvement"] = certified
    row["comparison_quality"] = (
        "certified_improvement"
        if certified
        else "tight_gap_incumbents"
        if tight
        else "provisional_incumbents"
    )
    first_flows = {f["flow_id"]: f for f in one.get("flow_metrics", [])}
    row["flow_changes"] = [
        {
            "flow_id": f["flow_id"],
            "delta_response_time_s": f["response_time_s"]
            - first_flows[f["flow_id"]]["response_time_s"],
            "delta_waiting_s": f["decomposition"]["waiting_s"]
            - first_flows[f["flow_id"]]["decomposition"]["waiting_s"],
        }
        for f in two.get("flow_metrics", [])
        if f["flow_id"] in first_flows
    ]
    row["utilization_changes"] = {
        resource: {
            worker: value - one.get(resource, {}).get(worker, 0)
            for worker, value in two.get(resource, {}).items()
        }
        for resource in ("compute_utilization", "network_interface_utilization")
    }
    row["delta_weighted_waiting_s"] = difference(
        one.get("weighted_decomposition", {}),
        two.get("weighted_decomposition", {}),
        "waiting_s",
    )
    return row


def aggregate(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for pair in pairs:
        groups[
            pair["workload"], pair["category"], pair["flow_count"], pair["load"]
        ].append(pair)
    rows = []
    for (workload, category, flows, load), group in sorted(groups.items()):
        valid = [
            pair
            for pair in group
            if pair["valid_pair"] and pair.get("improvement_pct") is not None
        ]
        rows.append(
            {
                "workload": workload,
                "category": category,
                "flow_count": flows,
                "load": load,
                "scenario_count": len(group),
                "valid_pair_count": len(valid),
                "seed_count": len({pair["seed"] for pair in valid}),
                "median_improvement_pct": median(
                    pair["improvement_pct"] for pair in valid
                )
                if valid
                else None,
                "median_response_reduction_s": median(
                    pair["absolute_response_reduction_s"] for pair in valid
                )
                if valid
                else None,
                "tight_pair_count": sum(pair["tight_gaps"] for pair in valid),
                "certified_improvement_count": sum(
                    pair["certified_improvement"] for pair in valid
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    names = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
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
    temporary.replace(path)


def analyze(
    directory: Path, *, plots: bool = True, tight_gap: float = 0.01
) -> dict[str, Any]:
    records = list(read_journal(directory / "runs.jsonl").values())
    if not records:
        raise ValueError("No persisted runs to analyze")
    output = directory / "analysis"
    output.mkdir(exist_ok=True)
    by_scenario: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    counts: Counter[tuple[int, str]] = Counter()
    for record in records:
        strategy = _strategy_name(record)
        by_scenario[record["scenario_id"]][strategy] = record
        termination = (
            record.get("termination", "unfinished")
            if record["state"] == "finished"
            else "unfinished"
        )
        if strategy in {"full_p1", "full_p2"}:
            counts[record["priority_count"], termination] += 1
    pairs = []
    for variants in by_scenario.values():
        if "full_p1" in variants and "full_p2" in variants:
            pairs.append(compare(variants["full_p1"], variants["full_p2"], tight_gap))
        else:
            record = next(iter(variants.values()))
            pairs.append(
                {
                    **{
                        name: record.get(name)
                        for name in (
                            "scenario_id",
                            "seed",
                            "load",
                            "workload",
                            "category",
                            "flow_count",
                            "service_demand_cv",
                        )
                    },
                    "valid_pair": False,
                }
            )
    summaries = aggregate(pairs)
    feasibility = [
        {"priority_count": p, "termination": termination, "count": count}
        for (p, termination), count in sorted(counts.items())
    ]
    write_csv(output / "runs.csv", records)
    write_csv(output / "pairs.csv", pairs)
    write_csv(output / "aggregates.csv", summaries)
    write_csv(output / "feasibility.csv", feasibility)
    report = {
        "run_count": len(records),
        "scenario_count": len(by_scenario),
        "valid_pair_count": sum(pair["valid_pair"] for pair in pairs),
        "tight_gap_threshold": tight_gap,
        "feasibility": feasibility,
        "aggregates": summaries,
        "pairs": pairs,
    }
    atomic_json(output / "report.json", report)
    if plots:
        from priority_effect.plots import render

        render(records, pairs, summaries, feasibility, output)
    return report
