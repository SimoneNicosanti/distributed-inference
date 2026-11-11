from abc import ABC, abstractmethod

from controller.domain.optimization.request.optimization_request import (
    OptimizationRequestsByFlow,
)
from controller.domain.optimization.solving.optimization_context import (
    OptimizationContext,
)


class FormulationDataBuilder[FormulationDataT, FormulationConfigT](ABC):
    @abstractmethod
    def build_formulation_data(
        self,
        requests_by_flow: OptimizationRequestsByFlow,
        context: OptimizationContext,
        formulation_config: FormulationConfigT,
    ) -> FormulationDataT: ...
