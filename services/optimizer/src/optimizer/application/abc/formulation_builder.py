from abc import ABC, abstractmethod

from optimizer.domain.formulation.formulation import Formulation
from optimizer.domain.formulation.formulation_config import FormulationConfig
from optimizer.domain.problem.optimization_problem import OptimizationProblem


class FormulationBuilder[
    OptimizationProblemT: OptimizationProblem,
    FormulationT: Formulation,
    FormulationConfigT: FormulationConfig,
](ABC):
    @abstractmethod
    def build_formulation(
        self,
        problem: OptimizationProblemT,
        formulation_config: FormulationConfigT,
    ) -> FormulationT: ...
