"""Native model dimensions shared by scalability and priority experiments."""

import gurobipy as gp


def dimensions(model: gp.Model) -> dict[str, int]:
    return {
        "variables": model.NumVars,
        "binary_variables": model.NumBinVars,
        "integer_variables": model.NumIntVars - model.NumBinVars,
        "continuous_variables": model.NumVars - model.NumIntVars,
        "linear_constraints": model.NumConstrs,
        "quadratic_constraints": model.NumQConstrs,
        "general_constraints": model.NumGenConstrs,
        "linear_nonzeros": model.NumNZs,
        "quadratic_constraint_nonzeros": model.NumQCNZs,
    }
