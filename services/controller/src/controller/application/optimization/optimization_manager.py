from abc import ABC, abstractmethod

from controller.domain.optimization.optimization_request import OptimizationRequest


class OptimizationManager(ABC):
    @abstractmethod
    def optimize_inference_distribution(
        self, requests: list[OptimizationRequest]
    ) -> None: ...
