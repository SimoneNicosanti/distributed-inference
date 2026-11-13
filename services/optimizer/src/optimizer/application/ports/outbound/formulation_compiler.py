from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager

from optimizer.domain.formulation.formulation import Formulation


class FormulationCompiler[FormulationT: Formulation, CompiledFormulationT](ABC):
    @abstractmethod
    def compile_formulation(
        self,
        formulation: FormulationT,
    ) -> AbstractAsyncContextManager[CompiledFormulationT]: ...
