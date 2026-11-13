from dataclasses import dataclass

import gurobipy as gp


@dataclass(frozen=True, slots=True)
class GurobiCompiledFormulation[FormulationT, FormulationVariablesT]:
    """Compiled model and formulation metadata, valid within the builder's context."""

    native_model: gp.Model
    variables: FormulationVariablesT
    formulation: FormulationT
    construction_time_s: float
