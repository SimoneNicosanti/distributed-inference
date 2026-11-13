"""Backend-local contract for compiling a formulation into a Gurobi model."""

from abc import ABC, abstractmethod

import gurobipy as gp


class GurobiModelBuilder[FormulationT, FormulationVariablesT](ABC):
    @abstractmethod
    def build_model_from_formulation(
        self, model: gp.Model, data: FormulationT, /
    ) -> FormulationVariablesT: ...
