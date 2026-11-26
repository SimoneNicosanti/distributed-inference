"""Explicit model sizes; an optional separate presolve probe is not solver timing."""

import asyncio
from time import perf_counter
from typing import Any

from scalability_benchmark.model_metrics import dimensions
from scalability_benchmark.solver import CompactGurobiModel


async def model_metrics(
    compiled: CompactGurobiModel,
    *,
    probe: bool,
    time_limit_s: float,
    threads: int | None,
    seed: int,
    relative_gap: float,
) -> dict[str, Any]:
    native = compiled.native_model
    result: dict[str, Any] = {
        "model_dimensions": dimensions(native),
        "component_priority_variables": len(
            compiled.variables.compute_queue.component_priority
        ),
        "presolved_model_dimensions": None,
        "presolve_probe_wall_s": None,
        "solver_presolve_time_s": None,
        "root_relaxation_time_s": None,
    }
    if probe:
        started = perf_counter()
        old_parameters = {
            "TimeLimit": native.Params.TimeLimit,
            "Threads": native.Params.Threads,
            "Seed": native.Params.Seed,
            "MIPGap": native.Params.MIPGap,
        }
        presolved = None
        try:
            native.Params.TimeLimit = min(30.0, time_limit_s)
            if threads is not None:
                native.Params.Threads = threads
            native.Params.Seed = seed
            native.Params.MIPGap = relative_gap
            presolved = await asyncio.to_thread(native.presolve)
            result["presolved_model_dimensions"] = dimensions(presolved)
        except Exception as error:
            result["presolve_probe_error"] = f"{type(error).__name__}: {error}"
        finally:
            if presolved is not None:
                presolved.dispose()
            for name, value in old_parameters.items():
                native.setParam(name, value)
            result["presolve_probe_wall_s"] = perf_counter() - started
    return result
