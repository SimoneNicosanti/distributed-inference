"""Native dimensions plus explicitly identified, optional Gurobi log metrics."""

import math
import re
from typing import Any

import gurobipy as gp
from gurobipy import GRB

from scalability_benchmark.model_metrics import dimensions as model_dimensions

FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def dimensions(model: gp.Model) -> dict[str, Any]:
    variables = model.getVars()
    return {
        **model_dimensions(model),
        "free_binary_variables": sum(
            v.VType == GRB.BINARY and v.LB != v.UB for v in variables
        ),
        "free_integer_variables": sum(
            v.VType in (GRB.BINARY, GRB.INTEGER) and v.LB != v.UB for v in variables
        ),
        "identified_soc_constraints": sum(
            c.QCName.startswith(("queue_soc[", "network_soc["))
            for c in model.getQConstrs()
        ),
    }


def parse_log(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "solver_presolve_time_s": None,
        "root_relaxation_time_s": None,
        "root_objective_bound": None,
        "root_relaxation_status": None,
        "log_presolved_dimensions": None,
        "mip_start_status": None,
        "mip_start_loaded": None,
        "mip_start_objective": None,
    }
    start = re.search(
        rf"^\s*Loaded user MIP start(?: \d+)? with objective\s+({FLOAT})(?:\s|$)",
        text,
        re.M,
    )
    if start:
        result.update(
            mip_start_status="loaded",
            mip_start_loaded=True,
            mip_start_objective=float(start[1]),
        )
    elif re.search(r"User MIP start(?: \d+)? violates constraint", text):
        result.update(mip_start_status="violates_constraint", mip_start_loaded=False)
    elif "did not produce a new incumbent solution" in text:
        # This can also mean another incumbent is already better, not infeasibility.
        result["mip_start_status"] = "no_new_incumbent"
    presolve = re.search(rf"^\s*Presolve time:\s*({FLOAT})s\s*$", text, re.M)
    if presolve:
        result["solver_presolve_time_s"] = float(presolve[1])
    # First root only: a root relaxation may be restarted or repeated by the solver.
    root = re.search(r"^\s*Root relaxation:\s*(.*)$", text, re.M)
    if root:
        line = root[1]
        timing = re.search(rf",\s*({FLOAT})\s+seconds\b", line)
        objective = re.match(rf"objective\s+({FLOAT})(?:,|\s|$)", line)
        result["root_relaxation_time_s"] = float(timing[1]) if timing else None
        result["root_objective_bound"] = float(objective[1]) if objective else None
        result["root_relaxation_status"] = (
            "objective" if objective else line.split(",", 1)[0].strip()
        )
    size = re.search(
        r"^\s*Presolved:\s*(\d+) rows,\s*(\d+) columns,\s*(\d+) nonzeros", text, re.M
    )
    if size:
        result["log_presolved_dimensions"] = dict(
            zip(
                ("linear_rows", "variables", "linear_nonzeros"),
                map(int, size.groups()),
                strict=True,
            )
        )
        types = re.search(
            r"Variable types:\s*(\d+) continuous,\s*(\d+) integer\s*\((\d+) binary\)",
            text[size.end() :],
        )
        if types:
            result["log_presolved_dimensions"].update(
                continuous_variables=int(types[1]),
                integer_variables=int(types[2]) - int(types[3]),
                binary_variables=int(types[3]),
            )
    return result


def quality(native: gp.Model) -> dict[str, Any]:
    if native.SolCount == 0:
        return {"validated": False}
    bound, constraint, integer = native.BoundVio, native.ConstrVio, native.IntVio
    return {
        "validated": bound <= native.Params.FeasibilityTol
        and constraint <= native.Params.FeasibilityTol
        and integer <= native.Params.IntFeasTol,
        "bound_violation": bound,
        "constraint_violation": constraint,
        "integrality_violation": integer,
        "feasibility_tolerance": native.Params.FeasibilityTol,
        "integrality_tolerance": native.Params.IntFeasTol,
    }


def proof(record: dict[str, Any], objective: str) -> dict[str, Any]:
    incumbent, bound = record.get("objective_value"), record.get("objective_bound")
    exact_gap = record.get("final_gap") == 0
    closed = (
        incumbent is not None
        and bound is not None
        and math.isclose(incumbent, bound, rel_tol=0, abs_tol=1e-9)
    )
    zero = (
        objective == "arrival_weighted_normalized_delay"
        and incumbent is not None
        and abs(incumbent) <= 1e-9
    )
    status = record.get("solver_status")
    return {
        "solved_to_tolerance": status == GRB.OPTIMAL,
        "global_optimality_established": status == GRB.OPTIMAL
        and (exact_gap or closed),
        "zero_tardiness_found": zero,
        "runtime_censored": status
        in (
            GRB.TIME_LIMIT,
            GRB.NODE_LIMIT,
            GRB.INTERRUPTED,
            GRB.WORK_LIMIT,
            GRB.MEM_LIMIT,
        ),
    }
