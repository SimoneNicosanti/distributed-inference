"""Compile the unchanged production model; change only selected bounds/starts."""

import asyncio
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import gurobipy as gp

from optimizer.adapters.outbound.gurobi.entities.gurobi_solver_config import (
    GurobiSolverConfig,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.compact_variables import (
    CompactVariables,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.gurobi_misocp_compact_formulation_compiler import (
    GurobiMisocpCompactFormulationModelBuilder,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.gurobi_formulation_compiler import (
    GurobiFormulationCompiler,
)
from optimizer.adapters.outbound.gurobi.result_translation.formulations.compact.gurobi_misocp_compact_solved_compiled_formulation_translator import (
    GurobiMisocpCompactSolvedCompiledFormulationTranslator,
)
from optimizer.adapters.outbound.gurobi.solver.gurobi_solver import GurobiSolver
from optimizer.application.formulation.compact.misocp.misocp_compact_formulation_builder import (
    MisocpCompactFormulationBuilder,
)
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation import (
    MisocpCompactFormulation,
)
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation_config import (
    MisocpCompactFormulationConfig,
)
from optimizer.domain.problem.compact.compact_optimization_problem import (
    CompactOptimizationProblem,
)
from scalability_benchmark.bottleneck.configuration import SolverSettings, Study
from scalability_benchmark.bottleneck.decisions import (
    apply,
    groups,
    reference_values,
    signature,
)
from scalability_benchmark.bottleneck.instances import jsonable
from scalability_benchmark.bottleneck.metrics import (
    dimensions,
    parse_log,
    proof,
    quality,
)
from scalability_benchmark.bottleneck.model_selection import ModelChoices
from scalability_benchmark.solver import (
    CompactSolverOutput,
    _solution_observations,
    classify,
)


def compiler() -> GurobiFormulationCompiler[MisocpCompactFormulation, CompactVariables]:
    return GurobiFormulationCompiler(
        GurobiMisocpCompactFormulationModelBuilder(), non_convex=0, output_flag=False
    )


async def presolve_dimensions(
    native: gp.Model, settings: SolverSettings
) -> dict[str, Any]:
    probe = None
    try:
        # Probe a copy: no solver state, starts or tuning leaks into the measured model.
        probe = native.copy()
        probe.Params.TimeLimit = min(
            settings.time_limit_seconds, settings.presolve_probe_seconds
        )
        probe.Params.Threads = settings.threads
        probe.Params.Seed = settings.seed
        probe.Params.MIPGap = settings.mip_gap
        probe.Params.OutputFlag = 0
        work = asyncio.create_task(asyncio.to_thread(probe.presolve))
        try:
            presolved = await asyncio.shield(work)
        except asyncio.CancelledError:
            probe.terminate()
            await GurobiSolver._wait_for_optimization(work)
            if not work.cancelled() and work.exception() is None:
                work.result().dispose()
            raise
        try:
            return {
                "presolved_model_dimensions": dimensions(presolved),
                "presolve_probe_error": None,
            }
        finally:
            presolved.dispose()
    except Exception as error:
        return {
            "presolved_model_dimensions": None,
            "presolve_probe_error": f"{type(error).__name__}: {error}",
        }
    finally:
        if probe is not None:
            probe.dispose()


async def solve(
    problem: CompactOptimizationProblem,
    study: Study,
    relative_error: float,
    log_path: Path,
    *,
    ablation: str = "A",
    reference: dict[str, Any] | None = None,
    model_choices: ModelChoices | None = None,
    warm: bool = False,
    limit: float | None = None,
    capture: bool = False,
    measure: bool = True,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    started = perf_counter()
    output: CompactSolverOutput | None = None
    failure = None
    candidate = None
    record: dict[str, Any] = {
        "presolved_model_dimensions": None,
        "presolve_probe_wall_s": None,
        "solver_presolve_time_s": None,
        "root_relaxation_time_s": None,
        "root_objective_bound": None,
        "native_solve_started": False,
        "mip_start_requested": warm,
    }
    settings = study.solver
    time_limit = limit if limit is not None else settings.time_limit_seconds
    await asyncio.to_thread(log_path.parent.mkdir, parents=True, exist_ok=True)
    # An interrupted retry must not append to the previous attempt's log.
    await asyncio.to_thread(log_path.write_text, "")
    try:
        formulation = MisocpCompactFormulationBuilder().build_formulation(
            problem, MisocpCompactFormulationConfig(relative_error, study.formulation)
        )
        async with compiler().compile_formulation(formulation) as compiled:
            native = compiled.native_model
            original_signature = signature(compiled)
            record["model_signature"] = original_signature
            record["original_dimensions"] = dimensions(native)
            record["decision_group_counts"] = {
                k: len(v) for k, v in groups(compiled).items()
            }
            record["fixed_variable_names"] = apply(
                compiled, ablation, reference, warm=warm, model_choices=model_choices
            )
            record["bound_fixed_dimensions"] = dimensions(native)
            record["construction_time_s"] = perf_counter() - started
            if measure and settings.measure_presolve:
                probe_started = perf_counter()
                record.update(await presolve_dimensions(native, settings))
                record["presolve_probe_wall_s"] = perf_counter() - probe_started
            native.Params.LogToConsole = int(settings.console)
            native.Params.LogFile = str(log_path)
            record["native_solve_started"] = True
            output = await GurobiSolver[
                MisocpCompactFormulation, CompactVariables
            ]().solve_compiled_formulation(
                compiled,
                GurobiSolverConfig(
                    output_flag=True,
                    time_limit_s=time_limit,
                    mip_gap=settings.mip_gap,
                    threads=settings.threads,
                    seed=settings.seed,
                    record_incumbent_history=True,
                ),
            )
            native.Params.LogFile = ""
            record.update(
                parse_log(await asyncio.to_thread(log_path.read_text, errors="replace"))
            )
            record["native_feasibility"] = quality(native)
            # Include inherited gurobi.env settings so external tuning is visible.
            record["native_parameters"] = {
                name: native.getParamInfo(name)[2]
                for name in (
                    "TimeLimit",
                    "MIPGap",
                    "MIPGapAbs",
                    "Threads",
                    "Seed",
                    "SolutionLimit",
                    "NonConvex",
                    "DegenMoves",
                    "Presolve",
                    "Method",
                    "NumericFocus",
                    "FeasibilityTol",
                    "IntFeasTol",
                    "OptimalityTol",
                )
            }
            if output.solved_model is not None:
                if capture:
                    candidate = {
                        "model_signature": original_signature,
                        "values": reference_values(compiled),
                    }
                try:
                    solution = await GurobiMisocpCompactSolvedCompiledFormulationTranslator().translate_solved_formulation_model(
                        compiled
                    )
                    record.update(_solution_observations(problem, solution))
                    record["solution"] = jsonable(solution)
                    record["worker_utilization"] = {
                        worker: sum(classes.values())
                        for worker, classes in record[
                            "worker_priority_utilization"
                        ].items()
                    }
                    record["network_utilization"] = {
                        str(worker.id): variable.X
                        for worker, variable in compiled.variables.network.network_utilization.items()
                    }
                    record["solution_extracted"] = True
                except Exception as error:
                    record["solution_extracted"] = False
                    record["translation_error"] = f"{type(error).__name__}: {error}"
    except Exception as error:
        failure = error
    classification = classify(
        output,
        failure,
        objective=problem.policy.objective,
        formulation=study.formulation,
    )
    # classify calls the post-fixing dimensions 'original'; keep explicit pre-fix names above.
    record = {**classification, **record}
    if "construction_time_s" not in record or record["construction_time_s"] is None:
        record["construction_time_s"] = perf_counter() - started
    record.update(proof(record, problem.policy.objective.value))
    record["raw_solver_statistics"] = asdict(output.statistics) if output else None
    record["wall_time_s"] = perf_counter() - started
    record["log_path"] = str(log_path)
    record["requested_solver_parameters"] = {
        "time_limit_seconds": time_limit,
        "mip_gap": settings.mip_gap,
        "threads": settings.threads,
        "seed": settings.seed,
        "solution_limit": None,
    }
    # Continuous presolve may solve without a MIPSOL callback. Do not fabricate its time.
    record["first_incumbent_time_observed"] = (
        record.get("time_to_first_incumbent_s") is not None
    )
    presolve, root, runtime = (
        record.get(k)
        for k in (
            "solver_presolve_time_s",
            "root_relaxation_time_s",
            "solver_runtime_s",
        )
    )
    record["remaining_native_runtime_s"] = (
        runtime - presolve - root
        if presolve is not None
        and root is not None
        and runtime is not None
        and runtime >= presolve + root
        else None
    )
    return record, candidate
