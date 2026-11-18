"""Reproducible small native comparison; not a scalability campaign."""

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from statistics import median

import gurobipy as gp
from test.optimizer.test_cumulative_formulation import dag_problem, formulation

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
from optimizer.adapters.outbound.gurobi.solver.gurobi_solver import GurobiSolver
from optimizer.application.ports.outbound.solver_result import SolverTerminationKind
from optimizer.domain.formulation.compact.misocp.compact_formulation_kind import (
    CompactFormulationKind as Kind,
)
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation import (
    MisocpCompactFormulation,
)


async def measure(repetitions: int) -> dict[str, object]:
    problem = dag_problem(variants=1, workers=2, components=3)
    compiler = GurobiFormulationCompiler(
        GurobiMisocpCompactFormulationModelBuilder(), non_convex=0
    )
    records = []
    for repetition in range(repetitions):
        order = list(Kind)
        if repetition % 2:
            order.reverse()
        for kind in order:
            async with compiler.compile_formulation(
                formulation(problem, kind)
            ) as compiled:
                result = await GurobiSolver[
                    MisocpCompactFormulation, CompactVariables
                ]().solve_compiled_formulation(
                    compiled,
                    GurobiSolverConfig(
                        time_limit_s=10,
                        threads=1,
                        seed=0,
                        mip_gap=0,
                        record_incumbent_history=True,
                    ),
                )
                if result.termination is not SolverTerminationKind.OPTIMAL:
                    raise RuntimeError(
                        f"Comparison requires proven optima: {result.termination}"
                    )
                effective_parameters = {
                    name: compiled.native_model.getParamInfo(name)[2]
                    for name in (
                        "Threads",
                        "Seed",
                        "MIPGap",
                        "MIPGapAbs",
                        "TimeLimit",
                        "NonConvex",
                        "DegenMoves",
                        "PreMIQCPForm",
                        "Presolve",
                        "Method",
                        "MIQCPMethod",
                        "FeasibilityTol",
                        "OptimalityTol",
                        "BarQCPConvTol",
                        "NumericFocus",
                    )
                }
                records.append(
                    {
                        "formulation": kind.value,
                        "repetition": repetition,
                        "objective": result.objective_value,
                        "runtime_s": result.runtime_s,
                        "effective_parameters": effective_parameters,
                        **asdict(result.statistics),
                    }
                )
    summaries = []
    for kind in Kind:
        selected = [record for record in records if record["formulation"] == kind.value]
        summaries.append(
            {
                "formulation": kind.value,
                "original_variables": selected[0]["original_variables"],
                "binary_variables": selected[0]["binary_variables"],
                "linear_constraints": selected[0]["linear_constraints"],
                "linear_nonzeros": selected[0]["linear_nonzeros"],
                "objective": selected[0]["objective"],
                "median_construction_time_s": median(
                    record["construction_time_s"] for record in selected
                ),
                "median_solver_runtime_s": median(
                    record["runtime_s"] for record in selected
                ),
            }
        )
    return {
        "gurobi_version": gp.gurobi.version(),
        "fixture": "one flow, one 8-layer branching/skip DAG, two outputs, two CPU/GPU workers, K=3, P=1",
        "repetitions": repetitions,
        "gurobi_settings": {"threads": 1, "seed": 0, "mip_gap": 0, "time_limit_s": 10},
        "summaries": summaries,
        "runs": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    result = asyncio.run(measure(args.repetitions))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summaries"], indent=2))


if __name__ == "__main__":
    main()
