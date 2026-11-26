"""Production compiler/solver/translator with experiment-only observation hooks."""

import asyncio
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

from gurobipy import GRB, GurobiError
from scalability_benchmark.analysis import finite_number
from scalability_benchmark.randomness import fingerprint
from scalability_benchmark.solver import (
    CompactGurobiModel,
    CompactSolverOutput,
    classify,
)

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
    ObjectiveKind,
)
from optimizer.domain.solution.optimization_solution import OptimizationSolution
from priority_effect.configuration import Config
from priority_effect.model_metrics import model_metrics
from priority_effect.reference import reference_component, reference_workers
from priority_effect.scenarios import jsonable


def fix_reference(compiled: CompactGurobiModel) -> None:
    """Fix only the separate reference check, never either experimental solve."""
    problem = compiled.formulation.problem
    device, edge = reference_workers(problem)
    assignment = compiled.variables.assignment
    for (_, key, layer, component), term in assignment.layer_component.items():
        variant = problem.variants[key.model_variant_id]
        compiled.native_model.addConstr(
            term == int(component == reference_component(variant, layer))
        )
    for (_, component, worker), term in assignment.component_worker.items():
        compiled.native_model.addConstr(
            term
            == int(component <= 3 and worker == (edge if component == 2 else device))
        )
    for (
        _,
        component,
        worker,
        priority,
    ), term in compiled.variables.compute_queue.component_priority.items():
        compiled.native_model.addConstr(
            term
            == int(
                priority == 1
                and component <= 3
                and worker == (edge if component == 2 else device)
            )
        )


def solution_metrics(
    problem: CompactOptimizationProblem,
    solution: OptimizationSolution,
    compiled: CompactGurobiModel,
) -> dict[str, Any]:
    flows: list[dict[str, Any]] = []
    compute = {str(worker.id): 0.0 for worker in problem.workers}
    class_load = {
        str(worker.id): {
            str(priority): 0.0
            for priority in range(1, problem.policy.priority_count + 1)
        }
        for worker in problem.workers
    }
    counts = {
        str(priority): 0 for priority in range(1, problem.policy.priority_count + 1)
    }
    execution_priorities = set()
    total_rate = sum(data.arrival_rate_per_s for data in problem.flows.values())
    weighted = dict.fromkeys(
        (
            "computation_s",
            "waiting_s",
            "transmission_s",
            "propagation_s",
            "batch_fill_s",
        ),
        0.0,
    )
    for flow in solution.flow_solutions:
        data = problem.flows[flow.flow_id]
        components = []
        configuration = next(
            c
            for c in compiled.formulation.configurations_by_flow[flow.flow_id]
            if c.model_variant_id == flow.model_variant_id
            and (
                flow.input_shapes is None
                or c.input_shapes.input_shapes == flow.input_shapes
            )
        )
        service_rate = compiled.formulation.service_arrival_rate_per_s[
            flow.flow_id, configuration
        ]
        for component in flow.components:
            worker, priority = str(component.worker_id.id), str(component.priority)
            utilization = service_rate * component.mean_execution_time_s
            compute[worker] += utilization
            class_load[worker][priority] += utilization
            counts[priority] += 1
            if component.mean_execution_time_s > 0:
                execution_priorities.add(priority)
            components.append(
                {
                    "index": component.component_index,
                    "worker": worker,
                    "priority": component.priority,
                    "layers": component.layer_keys,
                    "service_time_s": component.mean_execution_time_s,
                    "waiting_time_s": component.mean_execution_queue_time_s,
                    "transmission_response_s": component.mean_transmission_time_s,
                    "propagation_s": component.transmission_latency_s,
                }
            )
        decomposition = {
            "computation_s": sum(c.mean_execution_time_s for c in flow.components),
            "waiting_s": sum(c.mean_execution_queue_time_s for c in flow.components),
            "transmission_s": sum(c.mean_transmission_time_s for c in flow.components),
            "propagation_s": sum(c.transmission_latency_s for c in flow.components),
            "batch_fill_s": flow.mean_batch_fill_time_s,
        }
        for name, value in decomposition.items():
            weighted[name] += data.arrival_rate_per_s * value / total_rate
        tardiness = max(0.0, flow.mean_response_time_s - data.max_response_time_s)
        flows.append(
            {
                "flow_id": str(flow.flow_id.flow_id),
                "priorities_used": sorted({c.priority for c in flow.components}),
                "model_variant": str(flow.model_variant_id),
                "arrival_rate_per_s": data.arrival_rate_per_s,
                "response_time_s": flow.mean_response_time_s,
                "deadline_s": data.max_response_time_s,
                "response_deadline_ratio": flow.mean_response_time_s
                / data.max_response_time_s,
                "tardiness_s": tardiness,
                "normalized_tardiness": tardiness / data.max_response_time_s,
                "decomposition": decomposition,
                "components": components,
            }
        )
    weighted_response = (
        sum(f["arrival_rate_per_s"] * f["response_time_s"] for f in flows) / total_rate
    )
    weighted_tardiness = (
        sum(f["arrival_rate_per_s"] * f["normalized_tardiness"] for f in flows)
        / total_rate
    )
    tardiness_objective = (
        problem.policy.objective is ObjectiveKind.ARRIVAL_WEIGHTED_NORMALIZED_DELAY
    )
    normalized_response_objective = (
        problem.policy.objective
        is ObjectiveKind.ARRIVAL_WEIGHTED_NORMALIZED_RESPONSE_TIME
    )
    weighted_deadline_fraction = (
        sum(f["arrival_rate_per_s"] * f["response_deadline_ratio"] for f in flows)
        / total_rate
    )
    network = {
        str(worker.id): float(value.X)
        for worker, value in compiled.variables.network.network_utilization.items()
    }
    memory = {}
    for host_worker in problem.workers:
        value = sum(
            term.X
            for (
                *_,
                host,
            ), term in compiled.variables.memory.component_weight_mb.items()
            if host == host_worker
        )
        value += compiled.variables.memory.max_activation_mb[host_worker].X
        memory[str(host_worker.id)] = (
            value
            * 1_000_000
            / problem.workers[host_worker].available_inference_memory_bytes
        )
    return {
        "arrival_weighted_mean_response_time_s": weighted_response,
        "total_arrival_rate_per_s": total_rate,
        "flow_metrics": flows,
        "aggregate_normalized_tardiness": weighted_tardiness,
        "arrival_weighted_mean_deadline_fraction": weighted_deadline_fraction,
        "priority_component_counts": counts,
        "per_worker_priority_load": class_load,
        "both_priority_classes_used": all(counts.get(str(p), 0) > 0 for p in (1, 2)),
        "both_compute_priority_classes_used": {"1", "2"} <= execution_priorities,
        "compute_utilization": compute,
        "network_interface_utilization": network,
        "memory_constraint_fraction": memory,
        "weighted_decomposition": weighted,
        "resource_limit_proximity": {
            "compute": max(compute.values()) / problem.policy.compute_utilization_limit,
            "network": max(network.values()) / problem.policy.network_utilization_limit,
            "memory": max(memory.values()),
            "deadline": max(f["response_deadline_ratio"] for f in flows),
        },
        "objective_recomputed": weighted_tardiness
        if tardiness_objective
        else weighted_deadline_fraction
        if normalized_response_objective
        else weighted_response,
        "arrival_weighted_response_time_sum": weighted_response * total_rate,
        "native_objective_scaling": "arrival_weighted_mean_normalized_tardiness"
        if tardiness_objective
        else "arrival_weighted_mean_deadline_fraction"
        if normalized_response_objective
        else "arrival_weighted_mean_response_time_s",
    }


async def solve(
    problem: CompactOptimizationProblem,
    config: Config,
    *,
    reference: bool = False,
    iis_path: Path | None = None,
) -> dict[str, Any]:
    started = perf_counter()
    output: CompactSolverOutput | None = None
    failure = None
    metrics: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    extracted = False
    construction = None
    solver_wall_offset = None
    native_runtime = None
    native_started = False
    limit = config.solver.timeout_s
    try:
        formulation = MisocpCompactFormulationBuilder().build_formulation(
            problem,
            MisocpCompactFormulationConfig(
                relative_error=config.optimization.relative_error,
                representation=config.solver.formulation,
            ),
        )
        compiler = GurobiFormulationCompiler(
            GurobiMisocpCompactFormulationModelBuilder(),
            non_convex=0,
            output_flag=config.solver.output_flag,
        )
        async with compiler.compile_formulation(formulation) as compiled:
            native = compiled.native_model
            diagnostics["native_construction_time_s"] = compiled.construction_time_s
            diagnostics["effective_degen_moves"] = native.Params.DegenMoves
            if reference:
                fix_reference(compiled)
            native.update()
            construction = perf_counter() - started
            diagnostics.update(
                await model_metrics(
                    compiled,
                    probe=config.solver.measure_presolve_dimensions,
                    time_limit_s=limit,
                    threads=config.solver.threads,
                    seed=config.solver.seed,
                    relative_gap=config.solver.relative_gap,
                )
            )
            solver_wall_offset = perf_counter() - started
            native_started = True
            try:
                output = await GurobiSolver[
                    MisocpCompactFormulation, CompactVariables
                ]().solve_compiled_formulation(
                    compiled,
                    GurobiSolverConfig(
                        time_limit_s=limit,
                        mip_gap=config.solver.relative_gap,
                        threads=config.solver.threads,
                        seed=config.solver.seed,
                        output_flag=config.solver.output_flag,
                        record_incumbent_history=True,
                    ),
                )
            finally:
                with suppress(AttributeError, RuntimeError, GurobiError):
                    native_runtime = float(native.Runtime)
            if output.solved_model is not None:
                candidate = await GurobiMisocpCompactSolvedCompiledFormulationTranslator().translate_solved_formulation_model(
                    compiled
                )
                diagnostics["native_feasibility"] = {
                    "bound_violation": native.BoundVio,
                    "constraint_violation": native.ConstrVio,
                    "integrality_violation": native.IntVio,
                    "feasibility_tolerance": native.Params.FeasibilityTol,
                    "integrality_tolerance": native.Params.IntFeasTol,
                }
                diagnostics["native_feasibility_validated"] = (
                    native.BoundVio <= native.Params.FeasibilityTol
                    and native.ConstrVio <= native.Params.FeasibilityTol
                    and native.IntVio <= native.Params.IntFeasTol
                )
                metrics = solution_metrics(problem, candidate, compiled)
                metrics["solution"] = jsonable(candidate)
                metrics["active_component_count"] = sum(
                    len(f.components) for f in candidate.flow_solutions
                )
                metrics["distinct_priority_classes_used"] = len(
                    {c.priority for f in candidate.flow_solutions for c in f.components}
                )
                extracted = True
            elif (
                output.statistics.native_status == GRB.INFEASIBLE
                and iis_path is not None
            ):
                iis_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    await asyncio.to_thread(native.computeIIS)
                    native.write(str(iis_path))
                    diagnostics["iis_path"] = str(iis_path)
                except Exception as error:
                    diagnostics["iis_error"] = f"{type(error).__name__}: {error}"
    except Exception as error:
        failure = error
    result = classify(
        output,
        failure,
        objective=problem.policy.objective,
        formulation=config.solver.formulation,
    )
    runtime = output.runtime_s if output else native_runtime if native_started else 0.0
    first = finite_number(result.get("time_to_first_incumbent_s"))
    record = {
        **result,
        **metrics,
        **diagnostics,
        "priority_count": problem.policy.priority_count,
        "reference_check": reference,
        "solution_extracted": extracted,
        "construction_time_s": construction
        if construction is not None
        else perf_counter() - started,
        "solver_wall_offset_s": solver_wall_offset,
        "wall_time_s": perf_counter() - started,
        "solver_runtime_s": runtime,
        "native_solve_started": native_started,
        "raw_solver_statistics": asdict(output.statistics) if output else None,
        "requested_solver_configuration": {
            **config.solver.model_dump(mode="json"),
            "time_limit_s": limit,
        },
        "feasibility_certificate": bool(reference and extracted),
        "objective": problem.policy.objective.value,
        "formulation": config.solver.formulation.value,
        "problem_fingerprint": fingerprint(jsonable(problem)),
        "time_to_first_feasible_complete_solution_s": solver_wall_offset + first
        if solver_wall_offset is not None and first is not None and extracted
        else None,
    }
    history = (record["raw_solver_statistics"] or {}).get("incumbent_history", [])
    final_objective = finite_number(record.get("objective_value"))
    best_samples = [
        sample
        for sample in history
        if final_objective is not None
        and sample.get("objective_value") is not None
        and abs(sample["objective_value"] - final_objective) <= 1e-8
    ]
    record["time_to_best_observed_solution_s"] = (
        solver_wall_offset + best_samples[0]["runtime_s"]
        if solver_wall_offset is not None and best_samples and extracted
        else None
    )
    record["incumbent_history_sampling_s"] = 1.0
    return record
