"""Reuse the measured cold B solution as the common feasible reference."""

import asyncio
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from optimizer.domain.problem.compact.compact_optimization_problem import (
    CompactOptimizationProblem,
)
from scalability_benchmark.bottleneck.configuration import Study
from scalability_benchmark.bottleneck.model_selection import (
    REFERENCE_POLICY,
    ModelChoices,
    assignment_record,
)
from scalability_benchmark.bottleneck.solver import solve
from scalability_benchmark.persistence import atomic_json
from scalability_benchmark.randomness import fingerprint


async def obtain(
    problem: CompactOptimizationProblem,
    study: Study,
    relative_error: float,
    directory: Path,
    identity: str,
    *,
    model_choices: ModelChoices,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    await asyncio.to_thread(directory.mkdir, parents=True, exist_ok=True)
    path = directory / f"{identity}.json"
    started = perf_counter()
    assignment = assignment_record(model_choices)
    details: dict[str, Any] = {
        "identity": identity,
        "reference_policy": REFERENCE_POLICY,
        "source_ablation": "B",
        "model_assignment": assignment,
        "cache_hit": False,
        "acquisition": None,
        "validation": None,
        "available": False,
    }
    candidate = None
    if path.exists():
        try:
            cached = json.loads(path.read_text())
            if (
                cached["identity"] == identity
                and cached["reference_policy"] == REFERENCE_POLICY
                and cached["model_assignment"] == assignment
                and cached.get("b_cold") is not None
            ):
                candidate = cached["candidate"]
                details["acquisition"] = cached["b_cold"]
                details["cache_hit"] = True
        except (ValueError, KeyError, TypeError) as error:
            details["cache_error"] = str(error)
    if details["acquisition"] is None:
        print(
            f"  cold B: predefined variants; native budget {study.solver.time_limit_seconds:g}s",
            flush=True,
        )
        details["acquisition"], candidate = await solve(
            problem,
            study,
            relative_error,
            directory / f"{identity}-b-cold.log",
            ablation="B",
            model_choices=model_choices,
            capture=True,
        )
    cache = {
        "identity": identity,
        "reference_policy": REFERENCE_POLICY,
        "model_assignment": assignment,
        "b_cold": details["acquisition"],
        "candidate": candidate,
    }
    # Persist B even without an incumbent: resume must not silently rerun a
    # completed measured solve. A changed budget/policy receives a new identity.
    atomic_json(path, cache)
    reference = None
    if candidate is not None:
        details["validation"], validated = await solve(
            problem,
            study,
            relative_error,
            directory / f"{identity}-validation.log",
            ablation="E",
            model_choices=model_choices,
            reference=candidate,
            warm=True,
            limit=study.reference.validation_seconds,
            capture=True,
            measure=False,
        )
        if validated is not None and details["validation"].get(
            "native_feasibility", {}
        ).get("validated"):
            reference = validated
            reference["reference_id"] = fingerprint(
                {"identity": identity, "values": reference["values"]}
            )
            details["available"] = True
    details["wall_time_s"] = perf_counter() - started
    details["reference_id"] = reference["reference_id"] if reference else None
    atomic_json(path, {**cache, "reference": reference})
    atomic_json(directory / f"{identity}-report.json", details)
    return reference, details
