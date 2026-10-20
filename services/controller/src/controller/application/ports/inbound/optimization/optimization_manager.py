from abc import ABC, abstractmethod

from controller.domain.optimization.optimization_request import (
    OptimizationRequestsByFlow,
)


class OptimizationManager(ABC):
    @abstractmethod
    async def optimize_inference_distribution(
        self,
        requests_by_flow: OptimizationRequestsByFlow,
    ) -> None: ...
