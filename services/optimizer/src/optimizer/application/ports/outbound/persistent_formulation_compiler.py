from abc import abstractmethod
from contextlib import AbstractAsyncContextManager

from optimizer.application.ports.outbound.formulation_compiler import (
    FormulationCompiler,
)
from optimizer.domain.formulation.formulation import Formulation


class PersistentFormulationCompiler[FormulationT: Formulation, CompiledFormulationT](
    FormulationCompiler[FormulationT, CompiledFormulationT]
):
    """Keep one native model alive across successive compile/extend operations."""

    @abstractmethod
    def open_session(self) -> AbstractAsyncContextManager[None]: ...
