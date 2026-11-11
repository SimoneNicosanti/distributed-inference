from dataclasses import dataclass

import gurobipy as gp


@dataclass(frozen=True, slots=True)
class GurobiFormulationModel[FormulationDataT, FormulationVariablesT]:
    """Compiled model and formulation metadata, valid within the builder's context."""

    native_model: gp.Model
    variables: FormulationVariablesT
    data: FormulationDataT
    construction_time_s: float
