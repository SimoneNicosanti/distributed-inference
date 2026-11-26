"""Backend-independent feasibility and common-queue objective observations."""

from typing import Any

from optimizer.application.evaluation.compact_solution_evaluator import (
    decisions_from_solution,
    evaluate_decisions,
    validate_solution,
)
from optimizer.application.evaluation.queueing import QueueingMode
from optimizer.application.formulation.compact.misocp.misocp_compact_formulation_builder import (
    MisocpCompactFormulationBuilder,
)
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation_config import (
    MisocpCompactFormulationConfig,
)
from optimizer.domain.problem.compact.compact_optimization_problem import (
    CompactOptimizationProblem,
)
from optimizer.domain.solution.optimization_solution import OptimizationSolution
from scalability_benchmark.randomness import fingerprint
from scalability_benchmark.serialization import jsonable


def validation_observations(
    problem: CompactOptimizationProblem,
    solution: OptimizationSolution,
    mode: QueueingMode,
    relative_error: float,
) -> dict[str, Any]:
    data = MisocpCompactFormulationBuilder().build_formulation(
        problem,
        MisocpCompactFormulationConfig(relative_error),
    )
    report = validate_solution(
        data,
        solution,
        mode=mode,
        allow_epigraph_slack=mode is QueueingMode.LEGACY_EQUIVALENT,
    )
    native = evaluate_decisions(
        data, decisions_from_solution(data, solution), mode=QueueingMode.NATIVE
    )
    return {
        "problem_fingerprint": fingerprint(jsonable(problem)),
        "solution": jsonable(solution),
        "objective_value_recomputed": report.solution.objective_value
        if report.solution
        else None,
        "underlying_native_objective": native.solution.objective_value
        if native.solution
        else None,
        "independently_validated": report.valid,
        "max_constraint_violation": report.max_violation,
        "constraint_violation_metrics": dict(report.violations),
        "validation_tolerance": report.tolerance,
        "validation_mode": mode.value,
    }
