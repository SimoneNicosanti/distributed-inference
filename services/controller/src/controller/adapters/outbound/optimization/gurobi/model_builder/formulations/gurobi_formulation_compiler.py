"""Backend-local contract for compiling a formulation into a Gurobi model."""

from abc import ABC, abstractmethod

import gurobipy as gp


class GurobiFormulationCompiler[FormulationDataT, FormulationVariablesT](ABC):
    @abstractmethod
    def compile_formulation(
        self, model: gp.Model, data: FormulationDataT, /
    ) -> FormulationVariablesT: ...
