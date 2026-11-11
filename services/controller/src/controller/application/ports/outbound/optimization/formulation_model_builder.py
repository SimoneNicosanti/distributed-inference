from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager


class FormulationModelBuilder[FormulationDataT, FormulationModelT](ABC):
    @abstractmethod
    def build_formulation_model(
        self,
        data: FormulationDataT,
    ) -> AbstractAsyncContextManager[FormulationModelT]: ...
