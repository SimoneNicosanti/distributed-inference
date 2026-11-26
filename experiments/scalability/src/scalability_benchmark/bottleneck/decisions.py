"""Identify decisions through compiler containers, never mathematical name guesses."""

import math
from typing import Any

from gurobipy import GRB

from scalability_benchmark.bottleneck.model_selection import ModelChoices
from scalability_benchmark.randomness import fingerprint
from scalability_benchmark.solver import CompactGurobiModel


def signature(compiled: CompactGurobiModel) -> str:
    native = compiled.native_model
    native.update()
    variables = native.getVars()
    names = [v.VarName for v in variables]
    if len(set(names)) != len(names):
        raise ValueError(
            "Reference serialization requires unique native variable names"
        )
    return fingerprint(
        {
            "native_fingerprint": native.Fingerprint,
            "variables": [
                (v.VarName, v.VType, repr(v.LB), repr(v.UB), repr(v.Obj))
                for v in variables
            ],
        }
    )


def groups(compiled: CompactGurobiModel) -> dict[str, set[str]]:
    assignment = compiled.variables.assignment
    model = {v.VarName for v in assignment.configuration_selected.values()}
    partition = {v.VarName for _, v in assignment.layer_decision_items()}
    partition |= {v.VarName for v in assignment.component_active.values()}
    partition |= {v.VarName for v in assignment.component_last.values()}
    placement = {v.VarName for v in assignment.component_worker.values()}
    priority = {
        v.VarName for v in compiled.variables.compute_queue.component_priority.values()
    }
    integer = {
        v.VarName
        for v in compiled.native_model.getVars()
        if v.VType in (GRB.BINARY, GRB.INTEGER)
    }
    primary = model | partition | placement | priority
    if any(
        v.VType not in (GRB.BINARY, GRB.INTEGER, GRB.CONTINUOUS)
        for v in compiled.native_model.getVars()
    ):
        raise ValueError("Unsupported native discrete domain")
    return {
        "model": model,
        "partition": partition,
        "placement": placement,
        "priority": priority,
        "auxiliary_integer": integer - primary,
        "all_integer": integer,
        "all_discrete_decisions": integer | primary,
        "implied_binary_continuous": primary - integer,
    }


def selected(compiled: CompactGurobiModel, ablation: str) -> set[str]:
    g = groups(compiled)
    cumulative = {
        "A": set(),
        "B": g["model"],
        "C": g["model"] | g["partition"],
        "D": g["model"] | g["partition"] | g["placement"],
    }
    cumulative["P"] = cumulative["D"] | g["priority"]
    cumulative["E"] = g["all_discrete_decisions"]
    return cumulative[ablation]


def reference_values(compiled: CompactGurobiModel) -> dict[str, float]:
    """Normalize explicit/implied binary decisions; keep physical values for starts."""
    native = compiled.native_model
    result = {}
    discrete = groups(compiled)["all_discrete_decisions"]
    for variable in native.getVars():
        value = variable.X
        if not math.isfinite(value):
            raise ValueError(f"Nonfinite reference: {variable.VarName}")
        if variable.VarName in discrete:
            if abs(value - round(value)) > native.Params.IntFeasTol:
                raise ValueError(f"Nonintegral reference: {variable.VarName}")
            value = float(round(value))
        result[variable.VarName] = value
    return result


def check_reference(compiled: CompactGurobiModel, reference: dict[str, Any]) -> None:
    native = compiled.native_model
    if reference["model_signature"] != signature(compiled):
        raise ValueError("Reference model signature mismatch")
    values = reference["values"]
    discrete = groups(compiled)["all_discrete_decisions"]
    if set(values) != {v.VarName for v in native.getVars()}:
        raise ValueError("Reference variable set mismatch")
    for variable in native.getVars():
        value = values[variable.VarName]
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Nonfinite reference value")
        if (
            value < variable.LB - native.Params.FeasibilityTol
            or value > variable.UB + native.Params.FeasibilityTol
        ):
            raise ValueError(f"Reference outside original bounds: {variable.VarName}")
        if variable.VarName in discrete and value != round(value):
            raise ValueError("Reference integer values must already be normalized")


def apply(
    compiled: CompactGurobiModel,
    ablation: str,
    reference: dict[str, Any] | None,
    *,
    warm: bool = False,
    model_choices: ModelChoices | None = None,
) -> list[str]:
    native = compiled.native_model
    fixed = selected(compiled, ablation)
    values = {}
    if model_choices is not None:
        if set(model_choices) != set(compiled.formulation.problem.flows):
            raise ValueError(
                "Predefined model choices must cover exactly the instance flows"
            )
        for flow, choice in model_choices.items():
            if choice not in compiled.formulation.configurations_by_flow[flow]:
                raise ValueError("Predefined model is outside the eligible candidates")
        values = {
            v.VarName: float(configuration == model_choices[flow])
            for (
                flow,
                configuration,
            ), v in compiled.variables.assignment.configuration_selected.items()
        }
    if reference is not None:
        check_reference(compiled, reference)
        if any(reference["values"][name] != value for name, value in values.items()):
            raise ValueError("Reference disagrees with predefined model choices")
        values = reference["values"]
    if not fixed <= values.keys() or (warm and reference is None):
        raise ValueError(
            "A validated common reference is required for dependent decisions or warm starts"
        )
    for variable in native.getVars():
        if variable.VarName in fixed:
            variable.LB = variable.UB = values[variable.VarName]
        if warm:
            variable.Start = values[variable.VarName]
    native.update()
    return sorted(fixed)
